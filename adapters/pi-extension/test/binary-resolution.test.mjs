import assert from "node:assert/strict";
import { chmod, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { resolveMemkeeperBinary } from "../index.ts";

async function executable(path) {
	await mkdir(join(path, ".."), { recursive: true });
	await writeFile(path, "#!/bin/sh\nexit 0\n", "utf8");
	await chmod(path, 0o755);
}

test("installed binary wins over workspace build output", async () => {
	const root = await mkdtemp(join(tmpdir(), "memkeeper-binary-resolution-"));
	const installed = join(root, "installed", "memkeeper");
	const release = join(root, "source", "target", "release", "memkeeper");
	await executable(installed);
	await executable(release);

	const resolved = await resolveMemkeeperBinary({
		env: {},
		installedBin: installed,
		memkeeperRoot: join(root, "source"),
	});

	assert.equal(resolved, installed);
});

test("explicit MEMKEEPER_BIN still has highest priority", async () => {
	const root = await mkdtemp(join(tmpdir(), "memkeeper-binary-resolution-"));
	const explicit = join(root, "explicit", "memkeeper");
	const installed = join(root, "installed", "memkeeper");
	await executable(explicit);
	await executable(installed);

	const resolved = await resolveMemkeeperBinary({
		env: { MEMKEEPER_BIN: explicit },
		installedBin: installed,
		memkeeperRoot: join(root, "source"),
	});

	assert.equal(resolved, explicit);
});
