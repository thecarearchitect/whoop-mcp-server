import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { CallToolRequestSchema, ListToolsRequestSchema, ErrorCode, McpError } from '@modelcontextprotocol/sdk/types.js';
import express, { type Request, type Response } from 'express';
import { createHash } from 'node:crypto';
import { WhoopClient } from './whoop-client.js';
import { WhoopDatabase } from './database.js';
import { WhoopSync } from './sync.js';

interface ToolArguments {
	days?: number;
	full?: boolean;
}

const config = {
	clientId: process.env.WHOOP_CLIENT_ID ?? '',
	clientSecret: process.env.WHOOP_CLIENT_SECRET ?? '',
	redirectUri: process.env.WHOOP_REDIRECT_URI ?? 'http://localhost:3000/callback',
	dbPath: process.env.DB_PATH ?? './whoop.db',
	port: Number.parseInt(process.env.PORT ?? '3000', 10),
	mode: process.env.MCP_MODE ?? 'http',
};

const db = new WhoopDatabase(config.dbPath);
const client = new WhoopClient({
	clientId: config.clientId,
	clientSecret: config.clientSecret,
	redirectUri: config.redirectUri,
	onTokenRefresh: tokens => db.saveTokens(tokens),
});

const existingTokens = db.getTokens();
if (existingTokens) {
	client.setTokens(existingTokens);
}

const sync = new WhoopSync(client, db);

// ============================================================
// OAuth Proxy State Storage (for Claude.ai connector flow)
// ============================================================

interface PendingAuth {
	claudeRedirectUri: string;
	claudeState: string;
	codeChallenge?: string;
	codeChallengeMethod?: string;
	clientId?: string;
	createdAt: number;
}

interface IssuedCode {
	codeChallenge?: string;
	codeChallengeMethod?: string;
	redirectUri: string;
	createdAt: number;
}

const pendingAuths = new Map<string, PendingAuth>();
const issuedCodes = new Map<string, IssuedCode>();
const issuedTokens = new Set<string>();

// Cleanup expired entries every 5 minutes
setInterval(() => {
	const now = Date.now();
	const TEN_MINUTES = 10 * 60 * 1000;
	for (const [key, val] of pendingAuths) {
		if (now - val.createdAt > TEN_MINUTES) pendingAuths.delete(key);
	}
	for (const [key, val] of issuedCodes) {
		if (now - val.createdAt > TEN_MINUTES) issuedCodes.delete(key);
	}
}, 5 * 60 * 1000);

// ============================================================
// Helpers
// ============================================================

function getBaseUrl(req: Request): string {
	const proto = req.headers['x-forwarded-proto'] || 'https';
	const host = req.headers['x-forwarded-host'] || req.headers.host;
	return `${proto}://${host}`;
}

const SESSION_TTL_MS = 30 * 60 * 1000;
const transports = new Map<string, { transport: StreamableHTTPServerTransport; lastAccess: number }>();

function cleanupStaleSessions(): void {
	const now = Date.now();
	for (const [sessionId, session] of transports) {
		if (now - session.lastAccess > SESSION_TTL_MS) {
			session.transport.close().catch(() => {});
			transports.delete(sessionId);
		}
	}
}

setInterval(cleanupStaleSessions, 5 * 60 * 1000);

function formatDuration(millis: number | null): string {
	if (!millis) return 'N/A';
	const hours = Math.floor(millis / 3_600_000);
	const minutes = Math.floor((millis % 3_600_000) / 60_000);
	return `${hours}h ${minutes}m`;
}

function formatDate(isoString: string): string {
	return new Date(isoString).toLocaleDateString('en-US', {
		weekday: 'short',
		month: 'short',
		day: 'numeric',
	});
}

function getRecoveryZone(score: number): string {
	if (score >= 67) return 'Green (Well Recovered)';
	if (score >= 34) return 'Yellow (Moderate)';
	return 'Red (Needs Rest)';
}

function getStrainZone(strain: number): string {
	if (strain >= 18) return 'All Out (18-21)';
	if (strain >= 14) return 'High (14-17)';
	if (strain >= 10) return 'Moderate (10-13)';
	return 'Light (0-9)';
}

function validateDays(value: unknown): number {
	if (value === undefined || value === null) return 14;
	const num = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
	if (Number.isNaN(num) || num < 1) return 14;
	return Math.min(num, 90);
}

function validateBoolean(value: unknown): boolean {
	if (typeof value === 'boolean') return value;
	if (value === 'true') return true;
	return false;
}

// ============================================================
// MCP Server (unchanged tool definitions)
// ============================================================

function createMcpServer(): Server {
	const server = new Server(
		{ name: 'whoop-mcp-server', version: '1.0.0' },
		{ capabilities: { tools: {} } }
	);

	server.setRequestHandler(ListToolsRequestSchema, async () => ({
		tools: [
			{
				name: 'get_today',
				description: "Get today's Whoop data including recovery score, last night's sleep, and current strain.",
				inputSchema: { type: 'object', properties: {}, required: [] },
			},
			{
				name: 'get_recovery_trends',
				description: 'Get recovery score trends over time, including HRV and resting heart rate patterns.',
				inputSchema: {
					type: 'object',
					properties: { days: { type: 'number', description: 'Number of days to analyze (default: 14, max: 90)' } },
					required: [],
				},
			},
			{
				name: 'get_sleep_analysis',
				description: 'Get detailed sleep analysis including duration, stages, efficiency, and sleep debt.',
				inputSchema: {
					type: 'object',
					properties: { days: { type: 'number', description: 'Number of days to analyze (default: 14, max: 90)' } },
					required: [],
				},
			},
			{
				name: 'get_strain_history',
				description: 'Get training strain history and workout data.',
				inputSchema: {
					type: 'object',
					properties: { days: { type: 'number', description: 'Number of days to analyze (default: 14, max: 90)' } },
					required: [],
				},
			},
			{
				name: 'sync_data',
				description: 'Manually trigger a data sync from Whoop.',
				inputSchema: {
					type: 'object',
					properties: { full: { type: 'boolean', description: 'Force a full 90-day sync (default: false)' } },
					required: [],
				},
			},
			{
				name: 'get_auth_url',
				description: 'Get the Whoop authorization URL to connect your account.',
				inputSchema: { type: 'object', properties: {}, required: [] },
			},
		],
	}));

	server.setRequestHandler(CallToolRequestSchema, async request => {
		const { name, arguments: args } = request.params;
		const typedArgs = (args ?? {}) as ToolArguments;

		try {
			const dataTools = ['get_today', 'get_recovery_trends', 'get_sleep_analysis', 'get_strain_history'];
			if (dataTools.includes(name)) {
				const tokens = db.getTokens();
				if (!tokens) {
					return { content: [{ type: 'text', text: 'Not authenticated with Whoop. Use get_auth_url to authorize first.' }] };
				}
				client.setTokens(tokens);
				try {
					await sync.smartSync();
				} catch {
					// Continue with cached data
				}
			}

			switch (name) {
				case 'get_today': {
					const recovery = db.getLatestRecovery();
					const sleep = db.getLatestSleep();
					const cycle = db.getLatestCycle();

					if (!recovery && !sleep && !cycle) {
						return { content: [{ type: 'text', text: 'No data available. Try running sync_data first.' }] };
					}

					let response = "# Today's Whoop Summary\n\n";

					if (recovery) {
						response += `## Recovery: ${recovery.recovery_score ?? 'N/A'}% ${recovery.recovery_score ? getRecoveryZone(recovery.recovery_score) : ''}\n`;
						response += `- **HRV**: ${recovery.hrv_rmssd?.toFixed(1) ?? 'N/A'} ms\n`;
						response += `- **Resting HR**: ${recovery.resting_hr ?? 'N/A'} bpm\n`;
						if (recovery.spo2) response += `- **SpO2**: ${recovery.spo2.toFixed(1)}%\n`;
						if (recovery.skin_temp) response += `- **Skin Temp**: ${recovery.skin_temp.toFixed(1)}°C\n`;
						response += '\n';
					}

					if (sleep) {
						const totalSleep = (sleep.total_in_bed_milli ?? 0) - (sleep.total_awake_milli ?? 0);
						response += `## Last Night's Sleep\n`;
						response += `- **Total Sleep**: ${formatDuration(totalSleep)}\n`;
						response += `- **Performance**: ${sleep.sleep_performance?.toFixed(0) ?? 'N/A'}%\n`;
						response += `- **Efficiency**: ${sleep.sleep_efficiency?.toFixed(0) ?? 'N/A'}%\n`;
						response += `- **Stages**: Light ${formatDuration(sleep.total_light_milli)}, Deep ${formatDuration(sleep.total_deep_milli)}, REM ${formatDuration(sleep.total_rem_milli)}\n`;
						if (sleep.respiratory_rate) response += `- **Respiratory Rate**: ${sleep.respiratory_rate.toFixed(1)} breaths/min\n`;
						response += '\n';
					}

					if (cycle) {
						response += `## Current Strain\n`;
						response += `- **Day Strain**: ${cycle.strain?.toFixed(1) ?? 'N/A'} ${cycle.strain ? getStrainZone(cycle.strain) : ''}\n`;
						if (cycle.kilojoule) response += `- **Calories**: ${Math.round(cycle.kilojoule / 4.184)} kcal\n`;
						if (cycle.avg_hr) response += `- **Avg HR**: ${cycle.avg_hr} bpm\n`;
						if (cycle.max_hr) response += `- **Max HR**: ${cycle.max_hr} bpm\n`;
					}

					return { content: [{ type: 'text', text: response }] };
				}

				case 'get_recovery_trends': {
					const days = validateDays(typedArgs.days);
					const trends = db.getRecoveryTrends(days);

					if (trends.length === 0) {
						return { content: [{ type: 'text', text: 'No recovery data available for the requested period.' }] };
					}

					let response = `# Recovery Trends (Last ${days} Days)\n\n`;
					response += '| Date | Recovery | HRV | RHR |\n|------|----------|-----|-----|\n';

					for (const day of trends) {
						response += `| ${formatDate(day.date)} | ${day.recovery_score}% | ${day.hrv?.toFixed(1) ?? 'N/A'} ms | ${day.rhr ?? 'N/A'} bpm |\n`;
					}

					const avgRecovery = trends.reduce((sum, d) => sum + (d.recovery_score || 0), 0) / trends.length;
					const avgHrv = trends.reduce((sum, d) => sum + (d.hrv || 0), 0) / trends.length;
					const avgRhr = trends.reduce((sum, d) => sum + (d.rhr || 0), 0) / trends.length;

					response += `\n## Averages\n- **Recovery**: ${avgRecovery.toFixed(0)}%\n- **HRV**: ${avgHrv.toFixed(1)} ms\n- **RHR**: ${avgRhr.toFixed(0)} bpm\n`;

					return { content: [{ type: 'text', text: response }] };
				}

				case 'get_sleep_analysis': {
					const days = validateDays(typedArgs.days);
					const trends = db.getSleepTrends(days);

					if (trends.length === 0) {
						return { content: [{ type: 'text', text: 'No sleep data available for the requested period.' }] };
					}

					let response = `# Sleep Analysis (Last ${days} Days)\n\n`;
					response += '| Date | Duration | Performance | Efficiency |\n|------|----------|-------------|------------|\n';

					for (const day of trends) {
						response += `| ${formatDate(day.date)} | ${day.total_sleep_hours?.toFixed(1) ?? 'N/A'}h | ${day.performance?.toFixed(0) ?? 'N/A'}% | ${day.efficiency?.toFixed(0) ?? 'N/A'}% |\n`;
					}

					const avgDuration = trends.reduce((sum, d) => sum + (d.total_sleep_hours || 0), 0) / trends.length;
					const avgPerf = trends.reduce((sum, d) => sum + (d.performance || 0), 0) / trends.length;
					const avgEff = trends.reduce((sum, d) => sum + (d.efficiency || 0), 0) / trends.length;

					response += `\n## Averages\n- **Duration**: ${avgDuration.toFixed(1)} hours\n- **Performance**: ${avgPerf.toFixed(0)}%\n- **Efficiency**: ${avgEff.toFixed(0)}%\n`;

					return { content: [{ type: 'text', text: response }] };
				}

				case 'get_strain_history': {
					const days = validateDays(typedArgs.days);
					const trends = db.getStrainTrends(days);

					if (trends.length === 0) {
						return { content: [{ type: 'text', text: 'No strain data available for the requested period.' }] };
					}

					let response = `# Strain History (Last ${days} Days)\n\n`;
					response += '| Date | Strain | Calories |\n|------|--------|----------|\n';

					for (const day of trends) {
						response += `| ${formatDate(day.date)} | ${day.strain?.toFixed(1) ?? 'N/A'} | ${day.calories ?? 'N/A'} kcal |\n`;
					}

					const avgStrain = trends.reduce((sum, d) => sum + (d.strain || 0), 0) / trends.length;
					const avgCalories = trends.reduce((sum, d) => sum + (d.calories || 0), 0) / trends.length;

					response += `\n## Averages\n- **Daily Strain**: ${avgStrain.toFixed(1)}\n- **Daily Calories**: ${Math.round(avgCalories)} kcal\n`;

					return { content: [{ type: 'text', text: response }] };
				}

				case 'sync_data': {
					const tokens = db.getTokens();
					if (!tokens) {
						return { content: [{ type: 'text', text: 'Not authenticated with Whoop. Use get_auth_url to authorize first.' }] };
					}
					client.setTokens(tokens);

					const full = validateBoolean(typedArgs.full);
					let stats;

					if (full) {
						stats = await sync.syncDays(90);
					} else {
						const result = await sync.smartSync();
						if (result.type === 'skip') {
							return { content: [{ type: 'text', text: 'Data is already up to date (synced within the last hour).' }] };
						}
						stats = result.stats;
					}

					return {
						content: [{
							type: 'text',
							text: `Sync complete!\n- Cycles: ${stats?.cycles}\n- Recoveries: ${stats?.recoveries}\n- Sleeps: ${stats?.sleeps}\n- Workouts: ${stats?.workouts}`,
						}],
					};
				}

				case 'get_auth_url': {
					const scopes = ['read:profile', 'read:body_measurement', 'read:cycles', 'read:recovery', 'read:sleep', 'read:workout', 'offline'];
					const url = client.getAuthorizationUrl(scopes);
					return {
						content: [{
							type: 'text',
							text: `To authorize with Whoop:\n\n1. Visit: ${url}\n2. Log in and authorize\n3. You'll be redirected back automatically\n\nRedirect URI: ${config.redirectUri}`,
						}],
					};
				}

				default:
					throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error';
			return { content: [{ type: 'text', text: `Error: ${message}` }], isError: true };
		}
	});

	return server;
}

// ============================================================
// Main Server
// ============================================================

async function main(): Promise<void> {
	if (config.mode === 'stdio') {
		const server = createMcpServer();
		const transport = new StdioServerTransport();
		await server.connect(transport);
		process.stderr.write('Whoop MCP server running on stdio\n');
	} else {
		const app = express();

		// IMPORTANT: Do NOT apply express.json() globally!
		// The MCP SDK's StreamableHTTPServerTransport reads the raw body stream itself.
		// If express.json() consumes the stream first, handleRequest() gets an empty body → 400.
		// Instead, apply JSON parsing only to routes that need it (register, etc.).

		// ==================================================
		// OAuth Discovery Endpoints (NEW)
		// ==================================================

		// Protected Resource Metadata (RFC 9728)
		// Claude.ai fetches this to discover the authorization server
		app.get('/.well-known/oauth-protected-resource', (req: Request, res: Response) => {
			const baseUrl = getBaseUrl(req);
			res.json({
				resource: baseUrl,
				authorization_servers: [baseUrl],
				scopes_supported: ['whoop:read'],
				bearer_methods_supported: ['header'],
			});
		});

		// Also serve at path-specific location (fallback for Claude.ai)
		app.get('/.well-known/oauth-protected-resource/mcp', (req: Request, res: Response) => {
			const baseUrl = getBaseUrl(req);
			res.json({
				resource: `${baseUrl}/mcp`,
				authorization_servers: [baseUrl],
				scopes_supported: ['whoop:read'],
				bearer_methods_supported: ['header'],
			});
		});

		// OAuth Authorization Server Metadata (RFC 8414)
		// Describes this server's OAuth endpoints
		app.get('/.well-known/oauth-authorization-server', (req: Request, res: Response) => {
			const baseUrl = getBaseUrl(req);
			res.json({
				issuer: baseUrl,
				authorization_endpoint: `${baseUrl}/authorize`,
				token_endpoint: `${baseUrl}/token`,
				registration_endpoint: `${baseUrl}/register`,
				response_types_supported: ['code'],
				grant_types_supported: ['authorization_code', 'refresh_token'],
				code_challenge_methods_supported: ['S256'],
				token_endpoint_auth_methods_supported: ['client_secret_post', 'none'],
				scopes_supported: ['whoop:read'],
			});
		});

		// ==================================================
		// Dynamic Client Registration (NEW)
		// ==================================================

		// Claude.ai registers as an OAuth client
		app.post('/register', express.json(), (req: Request, res: Response) => {
			const { client_name, redirect_uris, grant_types, response_types, token_endpoint_auth_method, client_uri } = req.body;
			const clientId = crypto.randomUUID();
			res.status(201).json({
				client_id: clientId,
				client_name: client_name || 'Claude',
				redirect_uris: redirect_uris || [],
				grant_types: grant_types || ['authorization_code', 'refresh_token'],
				response_types: response_types || ['code'],
				token_endpoint_auth_method: token_endpoint_auth_method || 'none',
				client_uri: client_uri || '',
			});
		});

		// ==================================================
		// OAuth Authorization Endpoint (NEW)
		// Proxies Claude.ai's auth request to WHOOP
		// ==================================================

		app.get('/authorize', (req: Request, res: Response) => {
			const {
				response_type,
				client_id,
				redirect_uri,
				state,
				code_challenge,
				code_challenge_method,
			} = req.query;

			if (response_type !== 'code') {
				res.status(400).json({ error: 'unsupported_response_type' });
				return;
			}

			// Store Claude.ai's request params so we can redirect back after WHOOP auth
			const internalState = crypto.randomUUID();
			pendingAuths.set(internalState, {
				claudeRedirectUri: (redirect_uri as string) || '',
				claudeState: (state as string) || '',
				codeChallenge: code_challenge as string | undefined,
				codeChallengeMethod: code_challenge_method as string | undefined,
				clientId: client_id as string | undefined,
				createdAt: Date.now(),
			});

			// Redirect user to WHOOP's OAuth page
			const whoopScopes = [
				'read:profile', 'read:body_measurement', 'read:cycles',
				'read:recovery', 'read:sleep', 'read:workout', 'offline',
			];
			const params = new URLSearchParams({
				client_id: config.clientId,
				redirect_uri: config.redirectUri,
				response_type: 'code',
				scope: whoopScopes.join(' '),
				state: internalState,
			});

			res.redirect(`https://api.prod.whoop.com/oauth/oauth2/auth?${params}`);
		});

		// ==================================================
		// OAuth Callback (MODIFIED)
		// Now handles both direct auth AND Claude.ai proxy flow
		// ==================================================

		app.get('/callback', async (req: Request, res: Response) => {
			const code = req.query.code as string | undefined;
			const state = req.query.state as string | undefined;

			if (!code) {
				res.status(400).send('Missing authorization code');
				return;
			}

			try {
				// Exchange WHOOP's auth code for tokens
				const tokens = await client.exchangeCodeForTokens(code);
				db.saveTokens(tokens);

				// Start background sync of 90 days of data
				sync.syncDays(90).catch(() => {});

				// Check if this callback is part of Claude.ai's OAuth flow
				const pending = state ? pendingAuths.get(state) : undefined;

				if (pending && pending.claudeRedirectUri) {
					// This is a proxied flow from Claude.ai
					pendingAuths.delete(state!);

					// Generate a server-side authorization code for Claude.ai
					const serverCode = crypto.randomUUID();
					issuedCodes.set(serverCode, {
						codeChallenge: pending.codeChallenge,
						codeChallengeMethod: pending.codeChallengeMethod,
						redirectUri: pending.claudeRedirectUri,
						createdAt: Date.now(),
					});

					// Redirect back to Claude.ai with our server code
					const redirectUrl = new URL(pending.claudeRedirectUri);
					redirectUrl.searchParams.set('code', serverCode);
					if (pending.claudeState) {
						redirectUrl.searchParams.set('state', pending.claudeState);
					}

					res.redirect(redirectUrl.toString());
				} else {
					// Direct authorization (not via Claude.ai)
					res.send('Authorization successful! Your Whoop data is now syncing. You can close this window.');
				}
			} catch (error) {
				// Handle errors for proxied flow
				const pending = state ? pendingAuths.get(state) : undefined;
				if (pending && pending.claudeRedirectUri) {
					pendingAuths.delete(state!);
					const redirectUrl = new URL(pending.claudeRedirectUri);
					redirectUrl.searchParams.set('error', 'server_error');
					redirectUrl.searchParams.set('error_description', 'Failed to exchange WHOOP authorization code');
					if (pending.claudeState) {
						redirectUrl.searchParams.set('state', pending.claudeState);
					}
					res.redirect(redirectUrl.toString());
				} else {
					res.status(500).send('Authorization failed. Please try again.');
				}
			}
		});

		// ==================================================
		// OAuth Token Endpoint (NEW)
		// Claude.ai exchanges the server code for an access token
		// ==================================================

		app.post('/token', express.urlencoded({ extended: true }), (req: Request, res: Response) => {
			const { grant_type, code, code_verifier, refresh_token } = req.body;

			if (grant_type === 'authorization_code') {
				if (!code) {
					res.status(400).json({ error: 'invalid_request', error_description: 'Missing code' });
					return;
				}

				const issued = issuedCodes.get(code);
				if (!issued) {
					res.status(400).json({ error: 'invalid_grant', error_description: 'Invalid or expired authorization code' });
					return;
				}
				issuedCodes.delete(code);

				// Verify PKCE (required by Claude.ai)
				if (issued.codeChallenge && issued.codeChallengeMethod === 'S256') {
					if (!code_verifier) {
						res.status(400).json({ error: 'invalid_grant', error_description: 'Missing code_verifier for PKCE' });
						return;
					}
					const hash = createHash('sha256').update(code_verifier).digest('base64url');
					if (hash !== issued.codeChallenge) {
						res.status(400).json({ error: 'invalid_grant', error_description: 'Invalid code_verifier' });
						return;
					}
				}

				// Issue a session token to Claude.ai
				// (The actual WHOOP tokens are managed internally by the server)
				const accessToken = `whoop_session_${crypto.randomUUID()}`;
				const newRefreshToken = `whoop_refresh_${crypto.randomUUID()}`;
				issuedTokens.add(accessToken);

				res.json({
					access_token: accessToken,
					token_type: 'Bearer',
					expires_in: 86400, // 24 hours
					refresh_token: newRefreshToken,
				});

			} else if (grant_type === 'refresh_token') {
				// Claude.ai refreshes its session token
				const accessToken = `whoop_session_${crypto.randomUUID()}`;
				const newRefreshToken = `whoop_refresh_${crypto.randomUUID()}`;
				issuedTokens.add(accessToken);

				res.json({
					access_token: accessToken,
					token_type: 'Bearer',
					expires_in: 86400,
					refresh_token: newRefreshToken,
				});

			} else {
				res.status(400).json({ error: 'unsupported_grant_type' });
			}
		});

		// ==================================================
		// Health Endpoint (unchanged)
		// ==================================================

		app.get('/health', (_req: Request, res: Response) => {
			res.json({ status: 'ok', authenticated: Boolean(db.getTokens()) });
		});

		// ==================================================
		// MCP Endpoint (MODIFIED - added 401 for OAuth trigger)
		// ==================================================

		app.all('/mcp', async (req: Request, res: Response) => {
			// Check for Bearer token from Claude.ai
			// If no token, return 401 to trigger the OAuth flow
			const authHeader = req.headers.authorization;
			if (!authHeader || !authHeader.startsWith('Bearer ')) {
				const baseUrl = getBaseUrl(req);
				res.status(401)
					.set('WWW-Authenticate', `Bearer resource_metadata="${baseUrl}/.well-known/oauth-protected-resource"`)
					.json({ error: 'unauthorized', message: 'Bearer token required' });
				return;
			}

			// Handle MCP session management
			const sessionId = req.headers['mcp-session-id'] as string | undefined;

			if (req.method === 'DELETE' && sessionId && transports.has(sessionId)) {
				const session = transports.get(sessionId)!;
				await session.transport.close();
				transports.delete(sessionId);
				res.status(200).send('Session closed');
				return;
			}

			if (req.method === 'POST') {
				let transport: StreamableHTTPServerTransport;

				if (sessionId && transports.has(sessionId)) {
					const session = transports.get(sessionId)!;
					session.lastAccess = Date.now();
					transport = session.transport;
				} else {
					transport = new StreamableHTTPServerTransport({
						sessionIdGenerator: () => crypto.randomUUID(),
						onsessioninitialized: newSessionId => {
							transports.set(newSessionId, { transport, lastAccess: Date.now() });
						},
					});

					const server = createMcpServer();
					await server.connect(transport);
				}

				try {
					await transport.handleRequest(req, res);
				} catch (err) {
					console.error('[MCP] handleRequest error:', err);
					if (!res.headersSent) {
						res.status(500).json({ error: 'internal_error', message: String(err) });
					}
				}
				return;
			}

			res.status(405).send('Method not allowed');
		});

		// ==================================================
		// Deprecated SSE Endpoint (unchanged)
		// ==================================================

		app.get('/sse', (_req: Request, res: Response) => {
			res.status(410).send('SSE endpoint deprecated. Use /mcp with Streamable HTTP transport.');
		});

		// ==================================================
		// Start Server
		// ==================================================

		const server = app.listen(config.port, '0.0.0.0', () => {
			process.stdout.write(`Whoop MCP server running on http://0.0.0.0:${config.port}\n`);
			process.stdout.write(`OAuth endpoints active: /authorize, /token, /register\n`);
			process.stdout.write(`Metadata: /.well-known/oauth-protected-resource, /.well-known/oauth-authorization-server\n`);
		});

		const shutdown = (): void => {
			process.stdout.write('\nShutting down...\n');
			for (const [, session] of transports) {
				session.transport.close().catch(() => {});
			}
			transports.clear();
			db.close();
			server.close(() => process.exit(0));
		};

		process.on('SIGTERM', shutdown);
		process.on('SIGINT', shutdown);
	}
}

main().catch(error => {
	process.stderr.write(`Fatal error: ${error}\n`);
	process.exit(1);
});
