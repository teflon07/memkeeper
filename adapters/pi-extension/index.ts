import { constants, existsSync } from "node:fs";
import { access } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_TIMEOUT_MS, runMemkeeperViaTransport, shutdownStdioTransports } from "./transport.mjs";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const MEMKEEPER_ROOT = resolveMemkeeperRoot();
const DEFAULT_STORE_RELATIVE = ".memkeeper/store.sqlite";
const MAX_TOOL_TEXT_CHARS = 12_000;
const MAX_PREFIX_CAPTURE_CHARS = 4_096;
// Defaults tuned for the embed -> ANN -> cross-encoder rerank path on a warm
// serve child. The cross-encoder reranks ~rerank_candidates docs at ~100ms/doc
// on CPU, so the candidate pool and timeout are sized so warm retrieval of
// realistic (long) memories completes within budget; cold model load is masked
// by startup warmup. max_chars is large enough to inject several memories.
const DEFAULT_AUTO_RETRIEVE_TIMEOUT_MS = 4_000;
const DEFAULT_AUTO_RETRIEVE_MAX_MEMORIES = 5;
const DEFAULT_AUTO_RETRIEVE_MAX_CHARS = 3_000;
const DEFAULT_AUTO_RETRIEVE_RERANK_CANDIDATES = 12;
const DEFAULT_AUTO_RETRIEVE_MIN_PROMPT_CHARS = 20;
const DEFAULT_AUTO_RETRIEVE_MAX_QUERY_CHARS = 500;
const DEFAULT_AUTO_RETRIEVE_QUERY_EXPANSION = true;
const DEFAULT_AUTO_RETRIEVE_THREAD_EXPANSION = true;
const DEFAULT_AUTO_RETRIEVE_MAX_QUERY_VARIANTS = 8;
const DEFAULT_AUTO_RETRIEVE_MAX_THREAD_SEEDS = 3;
const DEFAULT_AUTO_RETRIEVE_MAX_THREAD_NEIGHBORS = 3;
// Option-3 gate defaults (see pack `cosine_gate`): inject when the embedding
// pool's top cosine clears the gate, OR the cross-encoder is confident.
const DEFAULT_AUTO_RETRIEVE_COSINE_GATE = 0.62;
const DEFAULT_AUTO_RETRIEVE_RERANK_CONFIDENCE = 0.05;

const PREFIX_CAPTURE_RULES = [
	{ prefix: "remember:", kind: "fact" },
	{ prefix: "fact:", kind: "fact" },
	{ prefix: "decision:", kind: "decision" },
	{ prefix: "preference:", kind: "preference" },
	{ prefix: "lesson:", kind: "lesson" },
	{ prefix: "action:", kind: "task" },
	{ prefix: "revert:", kind: "decision" },
];

const SECRET_PATTERNS = [
	/-----BEGIN [A-Z ]*PRIVATE KEY-----/i,
	/\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|client[_-]?secret|client[_-]?token|id[_-]?token|oauth[_-]?token|private[_-]?key|refresh[_-]?token|secret|session[_-]?token|signing[_-]?key|password|passwd|pwd)\s*[:=]\s*\S+/i,
	/\bauthorization\s*:\s*bearer\s+\S+/i,
	/\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/,
	/\bAIza[0-9A-Za-z_-]{35}\b/,
	/\bghp_[A-Za-z0-9_]{20,}\b/,
	/\bgithub_pat_[A-Za-z0-9_]{20,}\b/,
	/\bsk-[A-Za-z0-9_-]{20,}\b/,
	/\bxox[baprs]-[A-Za-z0-9-]{20,}\b/,
	/\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/,
];

const stringArraySchema = {
	type: "array",
	items: { type: "string" },
	maxItems: 64,
};

const searchFiltersSchema = {
	type: "object",
	additionalProperties: false,
	properties: {
		spaces: stringArraySchema,
		silos: stringArraySchema,
		scopes: stringArraySchema,
		projects: stringArraySchema,
		kinds: stringArraySchema,
		statuses: stringArraySchema,
		tags: stringArraySchema,
		entity_keys: stringArraySchema,
		claim_keys: stringArraySchema,
	},
};

const sourceSchema = {
	type: "object",
	additionalProperties: true,
	properties: {
		type: { type: "string" },
		adapter: { type: "string" },
		source_episode_id: { type: ["string", "null"] },
		session_id: { type: ["string", "null"] },
		cwd: { type: "string" },
		path: { type: "string" },
		source_description: { type: "string" },
	},
};

const storeProperty = {
	type: "string",
	description:
		"Optional memkeeper SQLite store path. Defaults to MEMKEEPER_STORE/PI_MEMKEEPER_STORE or .memkeeper/store.sqlite under the Pi cwd.",
};

const memorySearchParameters = {
	type: "object",
	additionalProperties: false,
	required: ["query"],
	properties: {
		store: storeProperty,
		query: { type: "string", minLength: 1, maxLength: 4096 },
		filters: searchFiltersSchema,
		limit: { type: "integer", minimum: 1, maximum: 50, default: 10 },
		offset: { type: "integer", minimum: 0, maximum: 1000, default: 0 },
		snippet_chars: { type: "integer", minimum: 0, maximum: 1000, default: 240 },
		include_content: { type: "boolean", default: false },
		include_source: { type: "boolean", default: false },
	},
};

const memoryReviewParameters = {
	type: "object",
	additionalProperties: false,
	properties: {
		store: storeProperty,
		filters: searchFiltersSchema,
		limit: { type: "integer", minimum: 1, maximum: 100, default: 20 },
		offset: { type: "integer", minimum: 0, maximum: 1000, default: 0 },
		snippet_chars: { type: "integer", minimum: 0, maximum: 1000, default: 240 },
		include_content: { type: "boolean", default: false },
		include_source: { type: "boolean", default: false },
		order: { type: "string", enum: ["updated_desc", "observed_desc", "created_desc"], default: "updated_desc" },
	},
};

const memoryEntitySearchParameters = {
	type: "object",
	additionalProperties: false,
	properties: {
		store: storeProperty,
		space: { type: "string" },
		query: { type: "string", minLength: 1, maxLength: 4096 },
		entity_key: { type: "string" },
		entity_types: stringArraySchema,
		statuses: stringArraySchema,
		limit: { type: "integer", minimum: 1, maximum: 50, default: 20 },
		offset: { type: "integer", minimum: 0, maximum: 1000, default: 0 },
		include_source: { type: "boolean", default: false },
	},
};

const memoryGraphNeighborsParameters = {
	type: "object",
	additionalProperties: false,
	properties: {
		store: storeProperty,
		space: { type: "string" },
		entity_id: { type: "string" },
		entity_key: { type: "string" },
		depth: { type: "integer", minimum: 1, maximum: 4, default: 1 },
		relation_types: stringArraySchema,
		statuses: stringArraySchema,
		max_edges: { type: "integer", minimum: 1, maximum: 200, default: 50 },
		include_tombstoned: { type: "boolean", default: false },
		include_source: { type: "boolean", default: false },
	},
};

const memoryRememberParameters = {
	type: "object",
	additionalProperties: false,
	required: ["content"],
	properties: {
		store: storeProperty,
		space: { type: "string" },
		silo: { type: "string" },
		scope: { type: "string", enum: ["global", "workspace", "project", "session", "custom"] },
		project: { type: "string" },
		kind: { type: "string" },
		content: { type: "string", minLength: 1, maxLength: 131072 },
		summary: { type: "string", maxLength: 8192 },
		tags: stringArraySchema,
		entity_key: { type: "string" },
		claim_key: { type: "string" },
		confidence: { type: "number", minimum: 0, maximum: 1, default: 1 },
		observed_at: { type: "string" },
		valid_from: { type: "string" },
		valid_to: { type: "string" },
		expires_at: { type: "string" },
		source: sourceSchema,
		pinned: { type: "boolean", default: false },
		supersedes: stringArraySchema,
		contradicts: stringArraySchema,
		dry_run: { type: "boolean", default: false },
	},
};

const memoryGetParameters = {
	type: "object",
	additionalProperties: false,
	required: ["id"],
	properties: {
		store: storeProperty,
		id: { type: "string", minLength: 1 },
		include_history: { type: "boolean", default: false },
		include_links: { type: "boolean", default: true },
		include_source: { type: "boolean", default: false },
	},
};

const memoryHistoryParameters = {
	type: "object",
	additionalProperties: false,
	required: ["id"],
	properties: {
		store: storeProperty,
		id: { type: "string", minLength: 1 },
		limit: { type: "integer", minimum: 1, maximum: 50, default: 20 },
		include_source: { type: "boolean", default: false },
	},
};

const memoryForgetParameters = {
	type: "object",
	additionalProperties: false,
	required: ["id"],
	properties: {
		store: storeProperty,
		id: { type: "string", minLength: 1 },
		reason: { type: "string", maxLength: 2048 },
		dry_run: { type: "boolean", default: false },
	},
};

const memoryDoctorParameters = {
	type: "object",
	additionalProperties: false,
	properties: {
		store: storeProperty,
		include_indexes: { type: "boolean", default: false },
	},
};

const memoryStatsParameters = {
	type: "object",
	additionalProperties: false,
	properties: {
		store: storeProperty,
		include_indexes: { type: "boolean", default: false },
	},
};

export default function memkeeperPiExtension(pi) {
	pi.on?.("session_shutdown", async () => {
		shutdownStdioTransports();
	});

	// Warm the serve child + ONNX models up front so the first retrieval is fast.
	void warmMemkeeper();

	let pendingRetrievalContent;

	pi.on?.("before_agent_start", async (event, ctx) => {
		pendingRetrievalContent = undefined;
		if (!autoRetrieveEnabled()) return;
		const query = promptToMemkeeperQuery(event?.prompt ?? "");
		if (!query) return;

		try {
			const envelope = await runMemkeeper(ctx, "pack", undefined, buildAutoRetrievePackRequest(query), ctx.signal, {
				timeoutMs: autoRetrieveTimeoutMs(),
			});
			pendingRetrievalContent = extractPackContent(envelope);
		} catch (_error) {
			// Fail open: prompt-time recall must never block normal Pi operation.
			pendingRetrievalContent = undefined;
		}
	});

	pi.on?.("context", async (event) => {
		if (!pendingRetrievalContent || !Array.isArray(event?.messages)) return undefined;
		const content = pendingRetrievalContent;
		pendingRetrievalContent = undefined;
		return {
			messages: insertBeforeLast(event.messages, memoryContextMessage(content, Date.now())),
		};
	});

	pi.on?.("input", async (event, ctx) => {
		if (!prefixCaptureEnabled()) return { action: "continue" };
		if (event.source === "extension") return { action: "continue" };

		const capture = parsePrefixCapture(event.text ?? "");
		if (!capture.matched) return { action: "continue" };
		if (capture.tooLong) {
			notify(ctx, `memkeeper prefix capture skipped: text exceeds ${MAX_PREFIX_CAPTURE_CHARS} characters`, "warning");
			return { action: "handled" };
		}
		if (!capture.content) {
			notify(ctx, "memkeeper prefix capture skipped: memory text is empty", "warning");
			return { action: "handled" };
		}
		if (Array.isArray(event.images) && event.images.length > 0) {
			notify(ctx, "memkeeper prefix capture skipped: text-only capture does not ingest images", "warning");
			return { action: "handled" };
		}
		if (looksLikeSecret(capture.content)) {
			notify(ctx, "memkeeper prefix capture skipped: text looks like it may contain a secret", "error");
			return { action: "handled" };
		}

		try {
			const request = cleanObject({
				kind: capture.kind,
				content: capture.content,
				tags: ["prefix-capture"],
				source: prefixCaptureSource(ctx, capture.prefix),
			});
			const envelope = await runMemkeeper(ctx, "remember", undefined, request, ctx.signal);
			const memory = envelope.result?.memory ?? {};
			notify(ctx, `memkeeper captured ${memory.id ?? "memory"} kind=${memory.kind ?? capture.kind}`, "info");
		} catch (error) {
			notify(ctx, `memkeeper prefix capture failed: ${truncateText(errorMessage(error), 1000)}`, "error");
		}
		return { action: "handled" };
	});

	pi.registerTool({
		name: "memory_search",
		label: "Memory Search",
		description:
			"Search memkeeper local memory with deterministic SQLite FTS/BM25. Source/provenance is hidden by default.",
		promptSnippet: "Search memkeeper local memory by query and metadata filters.",
		promptGuidelines: [
			"Use memory_search when prior local memory may help answer the user's current request.",
			"memory_search must leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
			"memory_search is read-only and must not be treated as semantic/LLM recall; it uses deterministic local memkeeper search.",
		],
		parameters: memorySearchParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				query: params.query,
				filters: params.filters ?? {},
				limit: params.limit ?? 10,
				offset: params.offset ?? 0,
				snippet_chars: params.snippet_chars ?? 240,
				include_content: params.include_content ?? false,
				include_source: params.include_source ?? false,
				semantic_fallback: "disabled",
			});
			const envelope = await runMemkeeper(ctx, "search", params.store, request, signal);
			return toolResult(formatSearch(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_entity_search",
		label: "Memory Entity Search",
		description:
			"Search memkeeper projected graph entities. Source/provenance is hidden by default.",
		promptSnippet: "Search memkeeper graph entities by key, name, alias, type, or status.",
		promptGuidelines: [
			"Use memory_entity_search when the user asks about explicit memkeeper entity keys or graph anchors.",
			"memory_entity_search must leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
			"Graph entities are rebuildable projections; memories remain the source of truth.",
		],
		parameters: memoryEntitySearchParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				space: params.space,
				query: params.query,
				entity_key: params.entity_key,
				entity_types: params.entity_types,
				statuses: params.statuses,
				limit: params.limit ?? 20,
				offset: params.offset ?? 0,
				include_source: params.include_source ?? false,
			});
			const envelope = await runMemkeeper(ctx, "entity-search", params.store, request, signal);
			return toolResult(formatEntitySearch(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_graph_neighbors",
		label: "Memory Graph Neighbors",
		description:
			"Traverse memkeeper projected graph neighbors with bounded SQLite recursion. Source/provenance is hidden by default.",
		promptSnippet: "Traverse memkeeper graph neighbors from one entity id or entity key.",
		promptGuidelines: [
			"Use memory_graph_neighbors when the user asks for connected memkeeper graph entities around a known seed.",
			"Provide exactly one of entity_id or entity_key.",
			"Leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
		],
		parameters: memoryGraphNeighborsParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				space: params.space,
				entity_id: params.entity_id,
				entity_key: params.entity_key,
				depth: params.depth ?? 1,
				relation_types: params.relation_types,
				statuses: params.statuses,
				max_edges: params.max_edges ?? 50,
				include_tombstoned: params.include_tombstoned ?? false,
				include_source: params.include_source ?? false,
			});
			const envelope = await runMemkeeper(ctx, "graph-neighbors", params.store, request, signal);
			return toolResult(formatGraphNeighbors(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_review",
		label: "Memory Review",
		description:
			"List recent memkeeper memories for review and cleanup. Source/provenance and full content are hidden by default.",
		promptSnippet: "Review recent memkeeper memories with ids for inspection or cleanup.",
		promptGuidelines: [
			"Use memory_review when the user asks what is stored in memory, wants to audit memories, or needs memory ids for cleanup.",
			"memory_review must leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
			"Use memory_forget only after the user explicitly asks to tombstone a specific memory id.",
		],
		parameters: memoryReviewParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				filters: params.filters ?? {},
				limit: params.limit ?? 20,
				offset: params.offset ?? 0,
				snippet_chars: params.snippet_chars ?? 240,
				include_content: params.include_content ?? false,
				include_source: params.include_source ?? false,
				order: params.order ?? "updated_desc",
			});
			const envelope = await runMemkeeper(ctx, "memory-list", params.store, request, signal);
			return toolResult(formatReview(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_remember",
		label: "Memory Remember",
		description:
			"Explicitly store one user-approved fact, decision, preference, lesson, or project note in memkeeper. Does not auto-ingest sessions.",
		promptSnippet: "Store one explicit memory in memkeeper when the user asks or clearly approves it.",
		promptGuidelines: [
			"Use memory_remember only for explicit durable facts, decisions, preferences, lessons, or notes the user asked to save.",
			"memory_remember must not store secrets, private credentials, or broad transcript dumps.",
			"memory_remember should prefer concise content with useful kind, tags, entity_key, or claim_key metadata when available.",
		],
		parameters: memoryRememberParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				space: params.space,
				silo: params.silo,
				scope: params.scope,
				project: params.project,
				kind: params.kind,
				content: params.content,
				summary: params.summary,
				tags: params.tags,
				entity_key: params.entity_key,
				claim_key: params.claim_key,
				confidence: params.confidence,
				observed_at: params.observed_at,
				valid_from: params.valid_from,
				valid_to: params.valid_to,
				expires_at: params.expires_at,
				source: params.source,
				pinned: params.pinned,
				supersedes: params.supersedes,
				contradicts: params.contradicts,
				dry_run: params.dry_run ?? false,
			});
			const envelope = await runMemkeeper(ctx, "remember", params.store, request, signal);
			return toolResult(formatRemember(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_get",
		label: "Memory Get",
		description: "Retrieve one memkeeper memory by id. History and source/provenance are hidden by default.",
		promptSnippet: "Retrieve one memkeeper memory by id.",
		promptGuidelines: [
			"Use memory_get after memory_search when an exact memory id needs full content or status details.",
			"memory_get must leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
		],
		parameters: memoryGetParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				id: params.id,
				include_history: params.include_history ?? false,
				include_links: params.include_links ?? true,
				include_source: params.include_source ?? false,
			});
			const envelope = await runMemkeeper(ctx, "get", params.store, request, signal);
			return toolResult(formatGet(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_history",
		label: "Memory History",
		description: "Retrieve one memkeeper memory's audit events and versions. Source/provenance is hidden by default.",
		promptSnippet: "Inspect audit history for one memkeeper memory id.",
		promptGuidelines: [
			"Use memory_history after memory_review or memory_get when a user wants to inspect why a memory exists or how it changed.",
			"memory_history must leave include_source false unless the user explicitly asks to inspect provenance/source metadata.",
			"Use memory_forget only after the user explicitly asks to tombstone a specific memory id; history is audit-preserving, not a hard-delete log.",
		],
		parameters: memoryHistoryParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				id: params.id,
				limit: params.limit ?? 20,
				include_source: params.include_source ?? false,
			});
			const envelope = await runMemkeeper(ctx, "history", params.store, request, signal);
			return toolResult(formatHistory(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_forget",
		label: "Memory Forget",
		description:
			"Tombstone one memkeeper memory by id. v0.1 forget is audit-preserving tombstone, not hard delete.",
		promptSnippet: "Tombstone one memkeeper memory by id when the user asks to forget it.",
		promptGuidelines: [
			"Use memory_forget only when the user explicitly asks to forget/tombstone a specific memory id.",
			"Prefer memory_review and memory_history first when the user is auditing or deciding what to clean up.",
			"Set dry_run=true when the user asks to preview a forget operation before committing it.",
			"memory_forget preserves audit history; do not promise hard deletion.",
		],
		parameters: memoryForgetParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const request = cleanObject({
				id: params.id,
				reason: params.reason,
				mode: "tombstone",
				dry_run: params.dry_run ?? false,
			});
			const envelope = await runMemkeeper(ctx, "forget", params.store, request, signal);
			return toolResult(formatForget(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_doctor",
		label: "Memory Doctor",
		description: "Diagnose memkeeper binary, adapter configuration, and local store readiness without mutating the store.",
		promptSnippet: "Run memkeeper setup diagnostics before relying on memory tools.",
		promptGuidelines: [
			"Use memory_doctor when memkeeper setup, binary resolution, store initialization, or Pi adapter configuration may be wrong.",
			"memory_doctor is read-only and must not be treated as initialization or repair.",
		],
		parameters: memoryDoctorParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const envelope = await runMemkeeper(ctx, "doctor", params.store, undefined, signal, {
				includeIndexes: params.include_indexes ?? false,
			});
			return toolResult(formatDoctor(envelope), envelope);
		},
	});

	pi.registerTool({
		name: "memory_stats",
		label: "Memory Stats",
		description: "Show memkeeper local store statistics and initialization health.",
		promptSnippet: "Inspect memkeeper store health and counts.",
		promptGuidelines: [
			"Use memory_stats to check whether the memkeeper store is initialized before relying on memory tools.",
		],
		parameters: memoryStatsParameters,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const envelope = await runMemkeeper(ctx, "stats", params.store, undefined, signal, {
				includeIndexes: params.include_indexes ?? false,
			});
			return toolResult(formatStats(envelope), envelope);
		},
	});
}

function autoRetrieveEnabled() {
	const value = firstNonEmpty(process.env.MEMKEEPER_AUTO_RETRIEVE, process.env.PI_MEMKEEPER_AUTO_RETRIEVE);
	if (!value) return true;
	return !["0", "false", "off", "no"].includes(value.trim().toLowerCase());
}

export function promptToMemkeeperQuery(prompt) {
	const chars = Array.from(String(prompt ?? ""));
	const minChars = envInt("MEMKEEPER_HOOK_MIN_PROMPT_CHARS", DEFAULT_AUTO_RETRIEVE_MIN_PROMPT_CHARS, 1, 10_000);
	if (chars.length < minChars) return undefined;
	const maxChars = envInt("MEMKEEPER_HOOK_MAX_QUERY_CHARS", DEFAULT_AUTO_RETRIEVE_MAX_QUERY_CHARS, 20, 10_000);
	return chars.slice(0, maxChars).join("");
}

function buildAutoRetrievePackRequest(query) {
		return cleanObject({
			title: "pi-auto-retrieve",
			queries: [query],
			max_memories: envInt("MEMKEEPER_HOOK_MAX_MEMORIES", DEFAULT_AUTO_RETRIEVE_MAX_MEMORIES, 1, 20),
			max_chars: envInt("MEMKEEPER_HOOK_MAX_CHARS", DEFAULT_AUTO_RETRIEVE_MAX_CHARS, 200, 20_000),
			format: "markdown",
			rerank_candidates: envInt("MEMKEEPER_HOOK_RERANK_CANDIDATES", DEFAULT_AUTO_RETRIEVE_RERANK_CANDIDATES, 0, 50),
			query_expansion: envBool("MEMKEEPER_HOOK_QUERY_EXPANSION", DEFAULT_AUTO_RETRIEVE_QUERY_EXPANSION),
			thread_expansion: envBool("MEMKEEPER_HOOK_THREAD_EXPANSION", DEFAULT_AUTO_RETRIEVE_THREAD_EXPANSION),
			max_query_variants: envInt(
				"MEMKEEPER_HOOK_MAX_QUERY_VARIANTS",
				DEFAULT_AUTO_RETRIEVE_MAX_QUERY_VARIANTS,
				1,
				64,
			),
			max_thread_seeds: envInt(
				"MEMKEEPER_HOOK_MAX_THREAD_SEEDS",
				DEFAULT_AUTO_RETRIEVE_MAX_THREAD_SEEDS,
				0,
				20,
			),
			max_thread_neighbors: envInt(
				"MEMKEEPER_HOOK_MAX_THREAD_NEIGHBORS",
				DEFAULT_AUTO_RETRIEVE_MAX_THREAD_NEIGHBORS,
				0,
				20,
			),
			// NOTE: the `pack` command does not accept `include_source` (it errors
			// `unknown field`). pack output is source-hidden by design, so we must not
		// send it here — doing so makes every auto-retrieve fail (fail-open => no
		// injection). See memory_search/pack: include_source is search-only.
		//
		// Option-3 gating (see pack `cosine_gate`): the query-level cosine gate
		// decides whether to inject at all; `min_score` is the cross-encoder
		// OR-confidence (a memory with rerank >= min_score forces injection even
		// when cos_top is below the gate). It is NOT a per-item floor on this path.
		cosine_gate: envFloat("MEMKEEPER_HOOK_COSINE_GATE", DEFAULT_AUTO_RETRIEVE_COSINE_GATE, 0, 1),
		min_score: envFloat("MEMKEEPER_HOOK_MIN_SCORE", DEFAULT_AUTO_RETRIEVE_RERANK_CONFIDENCE, 0, 1),
	});
}

// The CLI emits this marker line when pack finds nothing (notably once a
// min_score floor is set). Treat it as empty so we never inject a useless
// "no memories" context message.
const PACK_NO_MATCH_MARKER = "No matching active memories";

function extractPackContent(envelope) {
	const content = envelope?.result?.pack?.content;
	if (typeof content !== "string") return undefined;
	const trimmed = content.trim();
	if (!trimmed) return undefined;
	if (trimmed.includes(PACK_NO_MATCH_MARKER)) return undefined;
	return trimmed;
}

function memoryContextMessage(content, timestamp) {
	return {
		role: "user",
		content: `[Relevant local memkeeper memories — source-hidden, read-only context]\n${content}`,
		timestamp,
	};
}

function insertBeforeLast(messages, message) {
	if (!Array.isArray(messages) || messages.length === 0) return [message];
	return [...messages.slice(0, -1), message, messages[messages.length - 1]];
}

function autoRetrieveTimeoutMs() {
	return envInt("MEMKEEPER_HOOK_TIMEOUT_MS", DEFAULT_AUTO_RETRIEVE_TIMEOUT_MS, 100, 30_000);
}

function envInt(name, fallback, min, max) {
	const raw = process.env[name] ?? process.env[`PI_${name}`];
	if (!raw) return fallback;
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed)) return fallback;
	return Math.max(min, Math.min(max, parsed));
}

function envFloat(name, fallback, min, max) {
	const raw = process.env[name] ?? process.env[`PI_${name}`];
	if (!raw) return fallback;
	const parsed = Number.parseFloat(raw);
	if (!Number.isFinite(parsed)) return fallback;
	return Math.max(min, Math.min(max, parsed));
}

function envBool(name, fallback) {
	const raw = process.env[name] ?? process.env[`PI_${name}`];
	if (!raw) return fallback;
	const normalized = raw.trim().toLowerCase();
	if (["1", "true", "on", "yes"].includes(normalized)) return true;
	if (["0", "false", "off", "no"].includes(normalized)) return false;
	return fallback;
}

async function runMemkeeper(ctx, command, requestedStore, request, signal, options = {}) {
	const store = resolveStore(requestedStore, ctx.cwd);
	const memkeeper = await resolveMemkeeperBinary();
	return runMemkeeperViaTransport({
		binary: memkeeper,
		cwd: ctx.cwd,
		command,
		requestedStore,
		store,
		request,
		signal,
		timeoutMs: options.timeoutMs ?? (Number.parseInt(process.env.MEMKEEPER_TIMEOUT_MS ?? "", 10) || DEFAULT_TIMEOUT_MS),
		options,
		env: memkeeperProcessEnv(),
	});
}

// Augment the environment passed to the spawned memkeeper process so retrieval
// runs on the embed -> ANN -> cross-encoder rerank tier instead of the weaker
// BM25 fallback. memkeeper only loads the local ONNX models when
// MEMKEEPER_EMBED_MODEL_DIR / MEMKEEPER_RERANK_MODEL_DIR point at them, so we
// self-provision those from the bundled model dirs when present, and prefer the
// warm stdio serve transport (the CLI transport reloads the models on every
// call, ~5s; a persistent serve child loads them once per session). Computed
// once and reused so the stdio client is shared across calls.
let cachedMemkeeperEnv;
function memkeeperProcessEnv() {
	if (cachedMemkeeperEnv) return cachedMemkeeperEnv;
	const env = { ...process.env };
	defaultModelDir(env, "MEMKEEPER_EMBED_MODEL_DIR", "mxbai-embed-large");
	defaultModelDir(env, "MEMKEEPER_RERANK_MODEL_DIR", "mxbai-rerank-base");
	if (env.MEMKEEPER_EMBED_MODEL_DIR && !firstNonEmpty(env.MEMKEEPER_TRANSPORT, env.PI_MEMKEEPER_TRANSPORT)) {
		env.MEMKEEPER_TRANSPORT = "stdio";
	}
	cachedMemkeeperEnv = env;
	return cachedMemkeeperEnv;
}

function defaultModelDir(env, varName, dirName) {
	// Respect an explicit override (including the PI_-prefixed form), but copy it
	// onto the bare var the Rust binary actually reads.
	const existing = firstNonEmpty(env[varName], env[`PI_${varName}`]);
	if (existing) {
		env[varName] = existing;
		return;
	}
	const candidate = resolve(MEMKEEPER_ROOT, "models", dirName);
	if (existsSync(candidate)) env[varName] = candidate;
}

// Best-effort: spawn the persistent serve child at startup so its ONNX models
// load before the first prompt-time retrieval (cold load ~5s would otherwise
// blow the auto-retrieve timeout and fall back to no injection). Fail-open.
async function warmMemkeeper() {
	const env = memkeeperProcessEnv();
	if (env.MEMKEEPER_TRANSPORT !== "stdio") return;
	try {
		const binary = await resolveMemkeeperBinary();
		const cwd = process.cwd();
		await runMemkeeperViaTransport({
			binary,
			cwd,
			command: "stats",
			store: resolveStore(undefined, cwd),
			options: { includeIndexes: false },
			timeoutMs: 30_000,
			env,
		});
	} catch (_error) {
		// Warmup is best-effort; never block or surface errors.
	}
}

function resolveStore(requestedStore, cwd) {
	const raw = firstNonEmpty(requestedStore, process.env.MEMKEEPER_STORE, process.env.PI_MEMKEEPER_STORE);
	const store = stripAtPrefix(raw ?? DEFAULT_STORE_RELATIVE);
	return isAbsolute(store) ? store : resolve(cwd, store);
}

function resolveMemkeeperRoot() {
	const configured = firstNonEmpty(process.env.MEMKEEPER_ROOT, process.env.PI_MEMKEEPER_ROOT);
	return configured ? resolve(configured) : resolve(EXTENSION_DIR, "../..");
}

async function resolveMemkeeperBinary() {
	const envBin = firstNonEmpty(process.env.MEMKEEPER_BIN, process.env.PI_MEMKEEPER_BIN);
	const candidates = [
		envBin,
		resolve(MEMKEEPER_ROOT, "target/release/memkeeper"),
		resolve(MEMKEEPER_ROOT, "target/debug/memkeeper"),
	].filter(Boolean);
	for (const candidate of candidates) {
		if (await isExecutable(candidate)) return candidate;
	}
	return "memkeeper";
}

async function isExecutable(path) {
	try {
		await access(path, constants.X_OK);
		return true;
	} catch (_error) {
		return false;
	}
}

function toolResult(text, envelope) {
	return {
		content: [{ type: "text", text: truncateText(text, MAX_TOOL_TEXT_CHARS) }],
		details: envelope,
	};
}

function prefixCaptureEnabled() {
	const values = [process.env.MEMKEEPER_PREFIX_CAPTURE, process.env.PI_MEMKEEPER_PREFIX_CAPTURE]
		.filter((value) => typeof value === "string" && value.trim() !== "")
		.map((value) => value.trim().toLowerCase());
	return !values.some((value) => ["0", "false", "off", "no"].includes(value));
}

function parsePrefixCapture(text) {
	const raw = String(text);
	const start = firstNonWhitespaceIndex(raw);
	if (start === undefined) return { matched: false };
	const maxPrefixLength = Math.max(...PREFIX_CAPTURE_RULES.map((rule) => rule.prefix.length));
	const head = raw.slice(start, start + maxPrefixLength);
	for (const rule of PREFIX_CAPTURE_RULES) {
		if (head.startsWith(rule.prefix)) {
			const analysis = analyzeBoundedCapture(raw, start, start + rule.prefix.length);
			return {
				matched: true,
				prefix: rule.prefix,
				kind: rule.kind,
				content: analysis.tooLong || !analysis.hasBody ? "" : raw.slice(start).trimEnd(),
				tooLong: analysis.tooLong,
			};
		}
	}
	return { matched: false };
}

function firstNonWhitespaceIndex(text) {
	const maxLeadingWhitespace = 1024;
	let scanned = 0;
	for (let index = 0; index < text.length && scanned <= maxLeadingWhitespace; index += 1) {
		if (!isAsciiWhitespace(text[index])) return index;
		scanned += 1;
	}
	return undefined;
}

function analyzeBoundedCapture(text, contentStart, bodyStart) {
	let count = 0;
	let hasBody = false;
	for (let index = contentStart; index < text.length; ) {
		const codePoint = text.codePointAt(index);
		const char = String.fromCodePoint(codePoint);
		count += 1;
		if (count > MAX_PREFIX_CAPTURE_CHARS) return { tooLong: true, hasBody };
		if (index >= bodyStart && !isAsciiWhitespace(char)) hasBody = true;
		index += codePoint > 0xffff ? 2 : 1;
	}
	return { tooLong: false, hasBody };
}

function isAsciiWhitespace(char) {
	return char === " " || char === "\n" || char === "\r" || char === "\t" || char === "\f" || char === "\v";
}

function looksLikeSecret(text) {
	return SECRET_PATTERNS.some((pattern) => pattern.test(text));
}

function prefixCaptureSource(ctx, prefix) {
	const sessionFile = ctx?.sessionManager?.getSessionFile?.();
	return cleanObject({
		type: "host_input",
		adapter: "memkeeper-pi-extension",
		session_id: typeof sessionFile === "string" ? sessionFile : undefined,
		cwd: ctx?.cwd,
		source_description: `explicit prefix capture (${prefix})`,
	});
}

function notify(ctx, message, level = "info") {
	if (ctx?.ui?.notify) ctx.ui.notify(message, level);
}

function errorMessage(error) {
	return error instanceof Error ? error.message : String(error);
}

function formatReview(envelope) {
	const result = envelope.result ?? {};
	const review = result.review ?? {};
	const rows = Array.isArray(result.results) ? result.results : [];
	const lines = [
		`memkeeper review: ${rows.length} memory/memories, total_estimate=${review.total_estimate ?? rows.length}, truncated=${Boolean(review.truncated)}`,
	];
	for (const row of rows) {
		const text = row.summary || row.snippet || row.content || "";
		const tags = Array.isArray(row.tags) && row.tags.length > 0 ? ` tags=${row.tags.join(",")}` : "";
		lines.push(
			`${row.rank}. ${row.memory_id} kind=${row.kind} status=${row.status} updated=${row.updated_at ?? "?"}${tags}\n   ${collapseWhitespace(text)}`,
		);
		const sourceLine = formatSourceLine(row);
		if (sourceLine) lines.push(`   ${sourceLine}`);
	}
	return lines.join("\n");
}

function formatSearch(envelope) {
	const result = envelope.result ?? {};
	const search = result.search ?? {};
	const rows = Array.isArray(result.results) ? result.results : [];
	const lines = [
		`memkeeper search: ${rows.length} result(s), total_estimate=${search.total_estimate ?? rows.length}, truncated=${Boolean(search.truncated)}`,
	];
	for (const row of rows) {
		const text = row.summary || row.snippet || row.content || "";
		const tags = Array.isArray(row.tags) && row.tags.length > 0 ? ` tags=${row.tags.join(",")}` : "";
		lines.push(
			`${row.rank}. ${row.memory_id} score=${formatScore(row.score)} kind=${row.kind} status=${row.status}${tags}\n   ${collapseWhitespace(text)}`,
		);
		const sourceLine = formatSourceLine(row);
		if (sourceLine) lines.push(`   ${sourceLine}`);
	}
	return lines.join("\n");
}

function formatEntitySearch(envelope) {
	const result = envelope.result ?? {};
	const search = result.entity_search ?? {};
	const rows = Array.isArray(result.results) ? result.results : [];
	const lines = [
		`memkeeper entity-search: ${rows.length} result(s), total_estimate=${search.total_estimate ?? rows.length}, truncated=${Boolean(search.truncated)}`,
	];
	for (const row of rows) {
		const entity = row.entity ?? {};
		const aliases = Array.isArray(entity.aliases) && entity.aliases.length > 0 ? ` aliases=${entity.aliases.join(",")}` : "";
		const matched = Array.isArray(row.matched_aliases) && row.matched_aliases.length > 0 ? ` matched_aliases=${row.matched_aliases.join(",")}` : "";
		lines.push(
			`${row.rank}. ${entity.id ?? "?"} key=${entity.entity_key ?? "?"} type=${entity.entity_type ?? "?"} status=${entity.status ?? "?"}${aliases}${matched}\n   ${collapseWhitespace(entity.canonical_name ?? "")}`,
		);
		const sourceLine = formatSourceLine(entity);
		if (sourceLine) lines.push(`   ${sourceLine}`);
	}
	return lines.join("\n");
}

function formatGraphNeighbors(envelope) {
	const result = envelope.result ?? {};
	const graph = result.graph_neighbors ?? {};
	const seed = result.seed ?? {};
	const entities = Array.isArray(result.entities) ? result.entities : [];
	const relationships = Array.isArray(result.relationships) ? result.relationships : [];
	const lines = [
		`memkeeper graph-neighbors: seed=${seed.entity_key ?? seed.id ?? "?"} depth=${graph.depth ?? "?"} entities=${entities.length} relationships=${relationships.length} truncated=${Boolean(graph.truncated)}`,
	];
	for (const item of entities) {
		const entity = item.entity ?? {};
		lines.push(`entity depth=${item.depth ?? "?"} ${entity.id ?? "?"} key=${entity.entity_key ?? "?"} type=${entity.entity_type ?? "?"} status=${entity.status ?? "?"}`);
	}
	for (const item of relationships) {
		const relationship = item.relationship ?? {};
		const memory = relationship.memory_id ? ` memory=${relationship.memory_id}` : "";
		lines.push(
			`edge ${relationship.id ?? "?"}: ${relationship.subject_entity_id ?? "?"} -[${relationship.relation_type ?? "?"}]-> ${relationship.object_entity_id ?? "?"} status=${relationship.status ?? "?"}${memory}`,
		);
		const sourceLine = formatSourceLine(relationship);
		if (sourceLine) lines.push(`   ${sourceLine}`);
	}
	return lines.join("\n");
}

function formatRemember(envelope) {
	const memory = envelope.result?.memory ?? {};
	const dryRun = envelope.result?.dry_run ? " dry_run=true" : "";
	const lines = [
		`memkeeper remembered${dryRun}: ${memory.id ?? "<unknown>"} kind=${memory.kind ?? "unknown"} status=${memory.status ?? "unknown"}`,
		collapseWhitespace(memory.summary || memory.content || ""),
	];
	const candidates = Array.isArray(envelope.result?.candidates) ? envelope.result.candidates : [];
	if (candidates.length > 0) {
		lines.push(`candidate memories (${candidates.length}${envelope.result?.candidates_truncated ? "+" : ""}):`);
		for (const candidate of candidates.slice(0, 5)) {
			const matched = Array.isArray(candidate.matched_on) && candidate.matched_on.length > 0 ? ` via ${candidate.matched_on.join(",")}` : "";
			lines.push(`- ${candidate.memory_id} ${candidate.relationship ?? "candidate"} score=${formatScore(candidate.score)}${matched}: ${truncateText(collapseWhitespace(candidate.summary || candidate.snippet || ""), 240)}`);
		}
	}
	return lines.filter(Boolean).join("\n");
}

function formatGet(envelope) {
	const memory = envelope.result?.memory ?? {};
	const lines = [
		`memkeeper memory: ${memory.id ?? "<unknown>"} status=${memory.status ?? "unknown"} kind=${memory.kind ?? "unknown"} confidence=${formatScore(memory.confidence)}`,
		`space=${memory.space ?? "?"} silo=${memory.silo ?? "?"} scope=${memory.scope ?? "?"} observed_at=${memory.observed_at ?? "?"}`,
	];
	if (memory.summary) lines.push(`summary: ${collapseWhitespace(memory.summary)}`);
	if (memory.content) lines.push(`content: ${memory.content}`);
	if (Array.isArray(memory.tags) && memory.tags.length > 0) lines.push(`tags: ${memory.tags.join(",")}`);
	if (memory.entity_key) lines.push(`entity_key: ${memory.entity_key}`);
	if (memory.claim_key) lines.push(`claim_key: ${memory.claim_key}`);
	const sourceLine = formatSourceLine(memory);
	if (sourceLine) lines.push(sourceLine);
	return lines.join("\n");
}

function formatSourceLine(value) {
	if (!value || typeof value !== "object") return undefined;
	const source = value.source && typeof value.source === "object" ? value.source : undefined;
	const parts = [];
	if (value.source_episode_id) parts.push(`episode=${value.source_episode_id}`);
	if (source?.type) parts.push(`type=${source.type}`);
	if (source?.adapter) parts.push(`adapter=${source.adapter}`);
	if (source?.path) parts.push(`path=${source.path}`);
	if (source?.source_description) parts.push(`description=${collapseWhitespace(source.source_description)}`);
	return parts.length > 0 ? `source: ${truncateText(parts.join(" "), 1000)}` : undefined;
}

function formatHistory(envelope) {
	const result = envelope.result ?? {};
	const events = Array.isArray(result.events) ? result.events : [];
	const versions = Array.isArray(result.versions) ? result.versions : [];
	const lines = [
		`memkeeper history: ${result.memory_id ?? "<unknown>"} status=${result.current_status ?? "?"} events=${events.length} versions=${versions.length} truncated=${Boolean(result.truncated)}`,
	];
	for (const event of events) {
		const transition = event.old_status || event.new_status ? ` ${event.old_status ?? "?"}->${event.new_status ?? "?"}` : "";
		const reason = event.reason ? ` reason=${truncateText(collapseWhitespace(event.reason), 500)}` : "";
		lines.push(`event ${event.id ?? "?"}: ${event.type ?? "?"}${transition} at=${event.created_at ?? "?"}${reason}`);
	}
	for (const version of versions) {
		const text = version.summary || version.content || "";
		lines.push(
			`version ${version.version_num ?? "?"} ${version.id ?? "?"} created=${version.created_at ?? "?"}\n   ${truncateText(collapseWhitespace(text), 500)}`,
		);
		const sourceLine = formatSourceLine(version);
		if (sourceLine) lines.push(`   ${sourceLine}`);
	}
	return lines.join("\n");
}

function formatForget(envelope) {
	const result = envelope.result ?? {};
	if (result.dry_run) {
		return `memkeeper forget dry_run=true: would tombstone ${result.memory_id ?? "<unknown>"} ${result.old_status ?? "?"} -> ${result.new_status ?? "?"}; no changes committed`;
	}
	return `memkeeper tombstoned: ${result.memory_id ?? "<unknown>"} ${result.old_status ?? "?"} -> ${result.new_status ?? "?"}; audit history preserved`;
}

function formatDoctor(envelope) {
	const result = envelope.result ?? {};
	const doctor = result.doctor ?? {};
	const store = result.store ?? {};
	const checks = Array.isArray(result.checks) ? result.checks : [];
	const lines = [
		`memkeeper doctor: status=${doctor.status ?? "?"} mutating=${Boolean(doctor.mutating)}`,
		`store=${store.path ?? "?"} state=${store.state ?? "?"} exists=${Boolean(store.exists)} source=${store.path_source ?? "?"}`,
	];
	for (const check of checks) {
		lines.push(`check ${check.name ?? "?"}: ${check.status ?? "?"} - ${check.message ?? ""}`.trim());
	}
	if (store.error) {
		lines.push(`store error: ${store.error.code ?? "?"}: ${store.error.message ?? "?"}`);
	}
	return lines.join("\n");
}

function formatStats(envelope) {
	const result = envelope.result ?? {};
	return [
		`memkeeper stats: memories=${result.memory_count ?? "?"} active=${result.active_count ?? "?"} spaces=${result.space_count ?? "?"} silos=${result.silo_count ?? "?"}`,
		`schema=${result.schema_version ?? "?"} protocol=${result.protocol_version ?? "?"} journal=${result.journal_mode ?? "?"} bytes=${result.database_bytes ?? "?"}`,
	]
		.filter(Boolean)
		.join("\n");
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

function stripAtPrefix(value) {
	return value.startsWith("@") ? value.slice(1) : value;
}

function collapseWhitespace(value) {
	return String(value).replace(/\s+/g, " ").trim();
}

function formatScore(value) {
	return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "?";
}

function truncateText(value, maxChars) {
	const text = String(value);
	if ([...text].length <= maxChars) return text;
	return `${[...text].slice(0, maxChars).join("")}\n[memkeeper adapter output truncated]`;
}
