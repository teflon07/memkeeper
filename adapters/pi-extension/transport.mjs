import { spawn } from "node:child_process";

export const DEFAULT_TIMEOUT_MS = 5_000;
const PROTOCOL_VERSION = "memkeeper.v0.1";
const MAX_STDOUT_BYTES = 1_048_576;
const MAX_STDERR_BYTES = 65_536;
const READ_ONLY_COMMANDS = new Set([
	"doctor",
	"stats",
	"search",
	"entity-search",
	"graph-neighbors",
	"memory-list",
	"get",
	"history",
	"batch-search",
	"pack",
	"space-list",
	"silo-list",
]);

const stdioClients = new Map();
let nextRequestSequence = 0;

class MemkeeperTransportError extends Error {
	constructor(message, options = {}) {
		super(message);
		this.name = "MemkeeperTransportError";
		this.fallbackToCli = options.fallbackToCli ?? true;
		if (options.cause) this.cause = options.cause;
	}
}

export async function runMemkeeperViaTransport({
	binary,
	cwd,
	command,
	requestedStore,
	store,
	request,
	signal,
	timeoutMs = DEFAULT_TIMEOUT_MS,
	options = {},
	env = process.env,
}) {
	const invocation = buildMemkeeperInvocation({ command, requestedStore, store, request, options, cwd });
	const mode = resolveTransportMode(env);
	let envelope;

	if (mode === "stdio") {
		try {
			envelope = await runStdioInvocation({ binary, cwd, invocation, command, signal, timeoutMs, env });
		} catch (error) {
			if (!shouldFallbackToCli(error, env)) throw error;
			envelope = await runCliInvocation({ binary, cwd, invocation, command, signal, timeoutMs, env });
			envelope = addTransportFallbackWarning(envelope, error);
		}
	} else {
		envelope = await runCliInvocation({ binary, cwd, invocation, command, signal, timeoutMs, env });
	}

	return ensureOkEnvelope(envelope, command);
}

export function buildMemkeeperInvocation({ command, requestedStore, store, request, options = {}, cwd }) {
	const requestedStorePresent = firstNonEmpty(requestedStore) !== undefined;
	const passStore = command !== "doctor" || requestedStorePresent;
	const payload = command === "stats" || command === "doctor"
		? { include_indexes: Boolean(options.includeIndexes) }
		: (request ?? {});

	const cliArgs = [command];
	if (passStore) cliArgs.push("--store", store);
	if (command === "stats" || command === "doctor") {
		cliArgs.push("--json");
		cliArgs.push(options.includeIndexes ? "--include-indexes" : "--no-indexes");
	} else {
		cliArgs.push("--json", JSON.stringify(payload));
	}

	const serveRequest = cleanObject({
		protocol_version: PROTOCOL_VERSION,
		command,
		store_path: passStore ? store : undefined,
		cwd,
		payload,
	});

	return {
		cliArgs,
		serveRequest,
		readOnly: READ_ONLY_COMMANDS.has(command),
	};
}

export function resolveTransportMode(env = process.env) {
	const raw = firstNonEmpty(env.MEMKEEPER_TRANSPORT, env.PI_MEMKEEPER_TRANSPORT);
	if (!raw) return "cli";
	const value = raw.toLowerCase();
	if (["cli", "spawn", "process"].includes(value)) return "cli";
	if (["stdio", "serve", "serve-stdio"].includes(value)) return "stdio";
	throw new Error(`unsupported memkeeper transport ${JSON.stringify(raw)}; expected "cli" or "stdio"`);
}

export function shutdownStdioTransports() {
	for (const client of stdioClients.values()) {
		client.close();
	}
	stdioClients.clear();
}

async function runCliInvocation({ binary, cwd, invocation, command, signal, timeoutMs, env }) {
	const result = await execFileBounded(binary, invocation.cliArgs, {
		cwd,
		signal,
		timeoutMs,
		env,
	});
	return parseCliEnvelope(result, command);
}

async function runStdioInvocation({ binary, cwd, invocation, command, signal, timeoutMs, env }) {
	const client = getStdioClient(binary, cwd, env);
	return client.request(invocation.serveRequest, {
		command,
		signal,
		timeoutMs,
		fallbackToCliAfterWrite: invocation.readOnly,
	});
}

function getStdioClient(binary, cwd, env) {
	const key = stdioClientKey(binary, cwd, env);
	let client = stdioClients.get(key);
	if (!client) {
		client = new StdioMemkeeperClient(binary, cwd, env);
		stdioClients.set(key, client);
	}
	return client;
}

function stdioClientKey(binary, cwd, env) {
	return JSON.stringify([
		binary,
		cwd,
		env.MEMKEEPER_STORE ?? "",
		env.PI_MEMKEEPER_STORE ?? "",
	]);
}

class StdioMemkeeperClient {
	constructor(binary, cwd, env) {
		this.binary = binary;
		this.cwd = cwd;
		this.env = env;
		this.child = undefined;
		this.stdoutBuffer = "";
		this.stderr = "";
		this.pending = undefined;
		this.queue = Promise.resolve();
		this.closed = false;
	}

	request(serveRequest, options) {
		if (this.closed) {
			return Promise.reject(new MemkeeperTransportError("memkeeper stdio client is closed", { fallbackToCli: true }));
		}
		const run = this.queue.then(
			() => this.performRequest(serveRequest, options),
			() => this.performRequest(serveRequest, options),
		);
		this.queue = run.catch(() => {});
		return run;
	}

	performRequest(serveRequest, options) {
		if (options.signal?.aborted) return Promise.reject(abortError());

		let child;
		try {
			child = this.ensureChild();
		} catch (error) {
			return Promise.reject(
				new MemkeeperTransportError(`failed to start memkeeper stdio transport: ${errorMessage(error)}`, {
					fallbackToCli: true,
					cause: error,
				}),
			);
		}

		const requestId = nextRequestId();
		const line = `${JSON.stringify({ ...serveRequest, request_id: requestId })}\n`;

		return new Promise((resolve, reject) => {
			let settled = false;
			let written = false;
			let timeout;

			const settle = (error, responseLine) => {
				if (settled) return;
				settled = true;
				clearTimeout(timeout);
				if (options.signal) options.signal.removeEventListener("abort", onAbort);
				if (this.pending === pending) this.pending = undefined;
				if (error) {
					reject(error);
					return;
				}
				try {
					resolve(parseStdioEnvelope(responseLine, requestId, options.command, options.fallbackToCliAfterWrite));
				} catch (parseError) {
					this.killChild();
					reject(parseError);
				}
			};

			const fallbackForCurrentState = () => !written || options.fallbackToCliAfterWrite;
			const fail = (message, cause) => settle(new MemkeeperTransportError(message, {
				fallbackToCli: fallbackForCurrentState(),
				cause,
			}));
			const onAbort = () => {
				this.killChild();
				settle(abortError());
			};

			const pending = {
				requestId,
				settle,
				fail,
				fallbackToCli: fallbackForCurrentState,
			};
			this.pending = pending;

			timeout = setTimeout(() => {
				this.killChild();
				fail(`memkeeper stdio ${options.command} timed out after ${options.timeoutMs ?? DEFAULT_TIMEOUT_MS}ms`);
			}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);

			if (options.signal) options.signal.addEventListener("abort", onAbort, { once: true });

			try {
				written = true;
				child.stdin.write(line, "utf8", (error) => {
					if (error) {
						fail(`failed to write memkeeper stdio request: ${error.message}`, error);
					}
				});
			} catch (error) {
				written = false;
				fail(`failed to write memkeeper stdio request: ${errorMessage(error)}`, error);
			}
		});
	}

	ensureChild() {
		if (this.child && !this.child.killed) return this.child;
		this.stdoutBuffer = "";
		this.stderr = "";
		const child = spawn(this.binary, ["serve", "--stdio"], {
			cwd: this.cwd,
			stdio: ["pipe", "pipe", "pipe"],
			env: this.env,
		});
		this.child = child;

		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk) => this.handleStdout(chunk));
		child.stderr.on("data", (chunk) => this.handleStderr(chunk));
		child.on("error", (error) => this.handleChildFailure(child, `memkeeper stdio process error: ${error.message}`, error));
		child.on("close", (code, signal) => {
			const detail = signal ? `signal ${signal}` : `exit ${code}`;
			this.handleChildFailure(child, `memkeeper stdio process closed (${detail})`);
		});
		return child;
	}

	handleStdout(chunk) {
		this.stdoutBuffer += chunk;
		if (Buffer.byteLength(this.stdoutBuffer, "utf8") > MAX_STDOUT_BYTES) {
			this.killChild();
			this.pending?.fail("memkeeper stdio stdout exceeded adapter bound");
			return;
		}

		for (;;) {
			const newline = this.stdoutBuffer.indexOf("\n");
			if (newline < 0) return;
			const line = this.stdoutBuffer.slice(0, newline).trim();
			this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
			if (!line) continue;
			if (!this.pending) {
				this.killChild();
				return;
			}
			this.pending.settle(undefined, line);
		}
	}

	handleStderr(chunk) {
		this.stderr += chunk;
		if (Buffer.byteLength(this.stderr, "utf8") > MAX_STDERR_BYTES) {
			this.stderr = truncateText(this.stderr, MAX_STDERR_BYTES);
		}
	}

	handleChildFailure(child, message, cause) {
		if (this.child === child) this.child = undefined;
		this.stdoutBuffer = "";
		const stderr = this.stderr.trim();
		const suffix = stderr ? `; stderr: ${truncateText(stderr, 1000)}` : "";
		if (this.pending) {
			this.pending.fail(`${message}${suffix}`, cause);
		}
	}

	killChild() {
		const child = this.child;
		if (!child) return;
		this.child = undefined;
		this.stdoutBuffer = "";
		try {
			child.stdin.destroy();
		} catch (_error) {
			// Best-effort cleanup only.
		}
		if (!child.killed) child.kill("SIGTERM");
	}

	close() {
		this.closed = true;
		this.killChild();
	}
}

function parseCliEnvelope(result, command) {
	let envelope;
	try {
		envelope = JSON.parse(result.stdout.trim());
	} catch (_error) {
		throw new Error(
			`memkeeper ${command} emitted invalid JSON (exit ${result.code}). stderr: ${truncateText(result.stderr.trim(), 1000)}`,
		);
	}
	return ensureEnvelopeObject(envelope, command);
}

function parseStdioEnvelope(line, requestId, command, fallbackToCli) {
	let envelope;
	try {
		envelope = JSON.parse(line.trim());
	} catch (error) {
		throw new MemkeeperTransportError(
			`memkeeper stdio ${command} emitted invalid JSON: ${truncateText(line, 1000)}`,
			{ fallbackToCli, cause: error },
		);
	}
	envelope = ensureEnvelopeObject(envelope, command, fallbackToCli);
	if (envelope.request_id !== requestId) {
		throw new MemkeeperTransportError(
			`memkeeper stdio ${command} response request_id mismatch: expected ${requestId}, got ${JSON.stringify(envelope.request_id)}`,
			{ fallbackToCli },
		);
	}
	return envelope;
}

function ensureEnvelopeObject(envelope, command, fallbackToCli = false) {
	if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
		throw new MemkeeperTransportError(`memkeeper ${command} emitted a non-object JSON response`, { fallbackToCli });
	}
	return envelope;
}

function ensureOkEnvelope(envelope, command) {
	if (!envelope.ok) {
		const code = envelope.error?.code ?? "unknown";
		const message = envelope.error?.message ?? "unknown error";
		const hint = envelope.error?.hint ? ` Hint: ${envelope.error.hint}` : "";
		throw new Error(`memkeeper ${command} failed (${code}): ${message}${hint}`);
	}
	return envelope;
}

function addTransportFallbackWarning(envelope, error) {
	if (!envelope || typeof envelope !== "object") return envelope;
	const warning = `Pi adapter fell back to memkeeper CLI after stdio transport failure: ${truncateText(errorMessage(error), 500)}`;
	const warnings = Array.isArray(envelope.warnings) ? envelope.warnings : [];
	return { ...envelope, warnings: [...warnings, warning] };
}

function execFileBounded(command, args, options) {
	return new Promise((resolvePromise, reject) => {
		let stdout = "";
		let stderr = "";
		let settled = false;
		const child = spawn(command, args, {
			cwd: options.cwd,
			stdio: ["ignore", "pipe", "pipe"],
			env: options.env ?? process.env,
			signal: options.signal,
		});

		const timeout = setTimeout(() => {
			if (!settled) {
				settled = true;
				child.kill("SIGTERM");
				reject(new Error(`memkeeper command timed out after ${options.timeoutMs}ms`));
			}
		}, options.timeoutMs);

		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk) => {
			stdout += chunk;
			if (!settled && Buffer.byteLength(stdout, "utf8") > MAX_STDOUT_BYTES) {
				settled = true;
				clearTimeout(timeout);
				child.kill("SIGTERM");
				reject(new Error("memkeeper stdout exceeded adapter bound"));
			}
		});
		child.stderr.on("data", (chunk) => {
			stderr += chunk;
			if (Buffer.byteLength(stderr, "utf8") > MAX_STDERR_BYTES) {
				stderr = truncateText(stderr, MAX_STDERR_BYTES);
			}
		});
		child.on("error", (error) => {
			if (!settled) {
				settled = true;
				clearTimeout(timeout);
				reject(error);
			}
		});
		child.on("close", (code) => {
			if (!settled) {
				settled = true;
				clearTimeout(timeout);
				resolvePromise({ code, stdout, stderr });
			}
		});
	});
}

function shouldFallbackToCli(error, env) {
	return stdioFallbackEnabled(env) && error instanceof MemkeeperTransportError && error.fallbackToCli !== false;
}

function stdioFallbackEnabled(env) {
	const raw = firstNonEmpty(env.MEMKEEPER_STDIO_FALLBACK, env.PI_MEMKEEPER_STDIO_FALLBACK);
	if (!raw) return true;
	return !["0", "false", "off", "no"].includes(raw.toLowerCase());
}

function nextRequestId() {
	nextRequestSequence += 1;
	return `pi-${Date.now().toString(36)}-${nextRequestSequence.toString(36)}`;
}

function abortError() {
	const error = new Error("memkeeper command aborted");
	error.name = "AbortError";
	error.fallbackToCli = false;
	return error;
}

function cleanObject(value) {
	if (!value || typeof value !== "object" || Array.isArray(value)) return value;
	const output = {};
	for (const [key, child] of Object.entries(value)) {
		if (child === undefined) continue;
		if (Array.isArray(child) && child.length === 0) continue;
		if (child && typeof child === "object" && !Array.isArray(child)) {
			const cleaned = cleanObject(child);
			if (Object.keys(cleaned).length === 0) continue;
			output[key] = cleaned;
		} else {
			output[key] = child;
		}
	}
	return output;
}

function firstNonEmpty(...values) {
	for (const value of values) {
		if (typeof value === "string" && value.trim() !== "") return value.trim();
	}
	return undefined;
}

function errorMessage(error) {
	return error instanceof Error ? error.message : String(error);
}

function truncateText(value, maxChars) {
	const text = String(value);
	if ([...text].length <= maxChars) return text;
	return `${[...text].slice(0, maxChars).join("")}\n[memkeeper adapter output truncated]`;
}
