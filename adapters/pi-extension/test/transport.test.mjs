import assert from "node:assert/strict";
import { chmod, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { afterEach } from "node:test";

import {
	buildMemkeeperInvocation,
	runMemkeeperViaTransport,
	shutdownStdioTransports,
} from "../transport.mjs";

afterEach(() => {
	shutdownStdioTransports();
});

test("buildMemkeeperInvocation preserves doctor store-path behavior", () => {
	const withoutRequestedStore = buildMemkeeperInvocation({
		command: "doctor",
		requestedStore: undefined,
		store: "/tmp/store.sqlite",
		cwd: "/tmp/workspace",
		options: { includeIndexes: false },
	});
	assert.deepEqual(withoutRequestedStore.cliArgs, ["doctor", "--json", "--no-indexes"]);
	assert.equal(Object.hasOwn(withoutRequestedStore.serveRequest, "store_path"), false);
	assert.deepEqual(withoutRequestedStore.serveRequest.payload, { include_indexes: false });

	const withRequestedStore = buildMemkeeperInvocation({
		command: "doctor",
		requestedStore: "custom.sqlite",
		store: "/tmp/custom.sqlite",
		cwd: "/tmp/workspace",
		options: { includeIndexes: true },
	});
	assert.deepEqual(withRequestedStore.cliArgs, [
		"doctor",
		"--store",
		"/tmp/custom.sqlite",
		"--json",
		"--include-indexes",
	]);
	assert.equal(withRequestedStore.serveRequest.store_path, "/tmp/custom.sqlite");
	assert.deepEqual(withRequestedStore.serveRequest.payload, { include_indexes: true });
});

test("default transport keeps CLI behavior", async () => {
	const fixture = await createMockMemkeeper();
	const envelope = await runMemkeeperViaTransport({
		binary: fixture.binary,
		cwd: fixture.cwd,
		command: "stats",
		store: join(fixture.cwd, "store.sqlite"),
		options: { includeIndexes: false },
		timeoutMs: 1000,
		env: fixture.env,
	});

	assert.equal(envelope.result.transport, "cli");
	assert.deepEqual(envelope.result.args, [
		"stats",
		"--store",
		join(fixture.cwd, "store.sqlite"),
		"--json",
		"--no-indexes",
	]);
});

test("stdio transport reuses one serve process and echoes request ids", async () => {
	const fixture = await createMockMemkeeper({ MEMKEEPER_TRANSPORT: "stdio" });
	const first = await runMemkeeperViaTransport({
		binary: fixture.binary,
		cwd: fixture.cwd,
		command: "stats",
		store: join(fixture.cwd, "store.sqlite"),
		options: { includeIndexes: false },
		timeoutMs: 1000,
		env: fixture.env,
	});
	const second = await runMemkeeperViaTransport({
		binary: fixture.binary,
		cwd: fixture.cwd,
		command: "search",
		store: join(fixture.cwd, "store.sqlite"),
		request: { query: "deterministic" },
		timeoutMs: 1000,
		env: fixture.env,
	});

	assert.equal(first.result.transport, "stdio");
	assert.equal(second.result.transport, "stdio");
	assert.equal(first.result.pid, second.result.pid);
	assert.equal(typeof first.request_id, "string");
	assert.equal(typeof second.request_id, "string");
	assert.notEqual(first.request_id, second.request_id);
	assert.equal(second.result.payload.query, "deterministic");
});

test("stdio request-id mismatch falls back to CLI for read-only commands", async () => {
	const fixture = await createMockMemkeeper({
		MEMKEEPER_TRANSPORT: "stdio",
		MOCK_MEMKEEPER_STDIO_MODE: "mismatch",
	});
	const envelope = await runMemkeeperViaTransport({
		binary: fixture.binary,
		cwd: fixture.cwd,
		command: "stats",
		store: join(fixture.cwd, "store.sqlite"),
		options: { includeIndexes: false },
		timeoutMs: 1000,
		env: fixture.env,
	});

	assert.equal(envelope.result.transport, "cli");
	assert.match(envelope.warnings.at(-1), /fell back to memkeeper CLI/);
});

test("stdio valid error envelopes do not fall back to CLI", async () => {
	const fixture = await createMockMemkeeper({
		MEMKEEPER_TRANSPORT: "stdio",
		MOCK_MEMKEEPER_STDIO_MODE: "error-envelope",
	});
	await assert.rejects(
		runMemkeeperViaTransport({
			binary: fixture.binary,
			cwd: fixture.cwd,
			command: "stats",
			store: join(fixture.cwd, "store.sqlite"),
			options: { includeIndexes: false },
			timeoutMs: 1000,
			env: fixture.env,
		}),
		/memkeeper stats failed \(invalid_request\): mock validation failure/,
	);
});

test("stdio transport does not retry possibly-mutating written requests", async () => {
	const fixture = await createMockMemkeeper({
		MEMKEEPER_TRANSPORT: "stdio",
		MOCK_MEMKEEPER_STDIO_MODE: "mismatch",
	});
	await assert.rejects(
		runMemkeeperViaTransport({
			binary: fixture.binary,
			cwd: fixture.cwd,
			command: "remember",
			store: join(fixture.cwd, "store.sqlite"),
			request: { content: "remember me" },
			timeoutMs: 1000,
			env: fixture.env,
		}),
		/response request_id mismatch/,
	);
});

async function createMockMemkeeper(extraEnv = {}) {
	const cwd = await mkdtemp(join(tmpdir(), "memkeeper-pi-transport-test-"));
	const binary = join(cwd, "mock-memkeeper.mjs");
	await writeFile(binary, mockMemkeeperScript(), "utf8");
	await chmod(binary, 0o755);
	return {
		cwd,
		binary,
		env: {
			...process.env,
			...extraEnv,
		},
	};
}

function mockMemkeeperScript() {
	return `#!/usr/bin/env node
const args = process.argv.slice(2);
const command = args[0];

if (command === "serve" && args.includes("--stdio")) {
  process.stdin.setEncoding("utf8");
  let buffer = "";
  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf("\\n");
      if (newline < 0) return;
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      handleServeLine(line);
    }
  });
} else {
  emit({
    protocol_version: "memkeeper.v0.1",
    request_id: null,
    ok: true,
    command,
    store: { path: storeArg(args), schema_version: 1 },
    result: { transport: "cli", pid: process.pid, args },
    warnings: [],
    elapsed_ms: 0,
  });
}

function handleServeLine(line) {
  if (process.env.MOCK_MEMKEEPER_STDIO_MODE === "malformed") {
    console.log("not json");
    return;
  }
  const request = JSON.parse(line);
  if (process.env.MOCK_MEMKEEPER_STDIO_MODE === "mismatch") {
    emit(successEnvelope({ ...request, request_id: "wrong-request-id" }));
    return;
  }
  if (process.env.MOCK_MEMKEEPER_STDIO_MODE === "error-envelope") {
    emit({
      protocol_version: "memkeeper.v0.1",
      request_id: request.request_id,
      ok: false,
      command: request.command,
      error: {
        code: "invalid_request",
        message: "mock validation failure",
        details: {},
        retryable: false,
        hint: null,
      },
      warnings: [],
      elapsed_ms: 0,
    });
    return;
  }
  emit(successEnvelope(request));
}

function successEnvelope(request) {
  return {
    protocol_version: "memkeeper.v0.1",
    request_id: request.request_id,
    ok: true,
    command: request.command,
    store: { path: request.store_path ?? "diagnostic", schema_version: 1 },
    result: {
      transport: "stdio",
      pid: process.pid,
      payload: request.payload ?? {},
      store_path_present: Object.hasOwn(request, "store_path"),
      cwd: request.cwd ?? null,
    },
    warnings: [],
    elapsed_ms: 0,
  };
}

function storeArg(values) {
  const index = values.indexOf("--store");
  return index >= 0 ? values[index + 1] : "diagnostic";
}

function emit(value) {
  console.log(JSON.stringify(value));
}
`;
}
