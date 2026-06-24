# Building memkeeper on Windows (experimental)

> **Status: experimental, community-supported.** Windows is not a first-class
> memkeeper platform. The project is developed and tested on macOS and Linux, and
> there are no prebuilt Windows binaries — you build from source. CI compiles the
> codebase on Windows to catch breakage, but Windows fixes are best-effort and
> prioritized behind the Unix platforms. Issues and pull requests are welcome.

## What works, what doesn't

- **Works:** the full CLI (`init`, `remember`, `search`, `pack`, `reindex`, …), the
  HTTP dashboard (`serve --http`), and `serve --stdio` for MCP / editor integrations.
- **Both retrieval backends work:**
  - **Local, on-device semantic (default).** `cargo build --release` (default
    features) plus `memkeeper pull-models` gives fully on-device embeddings +
    reranking — verified on Windows (1024-dim embeddings, semantic-primary +
    reranked, warm-daemon HTTP). Needs the MSVC C++ build tools and the ONNX models
    (~2 GB).
  - **Off-device / lexical (`--no-default-features --features api`).** A lighter
    build with no ONNX runtime and no model download: deterministic lexical
    (BM25/FTS) out of the box, or off-device semantic with an embeddings API key
    (see "Off-device semantic" in the main [README](../README.md)).
- **Not available on Windows:** the Unix-socket warm-daemon mode (`serve --socket`);
  use the HTTP (`serve --http`) or stdio transports instead.

## Prerequisites

1. **Rust**, via [rustup](https://rustup.rs).
2. **The MSVC C++ build tools.** The default `x86_64-pc-windows-msvc` toolchain
   links with the Visual C++ linker (`link.exe`). Install the **"Desktop development
   with C++"** workload from the
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   — the standalone Build Tools installer is enough; you don't need full Visual
   Studio. Then open a fresh terminal so the toolchain is on `PATH`.

## Build

For fully on-device semantic (the default), build with default features and fetch
the models:

```powershell
git clone https://github.com/teflon07/memkeeper.git
cd memkeeper
cargo build --release
.\target\release\memkeeper.exe pull-models   # ~2 GB; prints PowerShell env lines to set
```

For a lighter build with no ONNX runtime and no model download (lexical, or
off-device semantic with an API key), use the `api` feature instead:

```powershell
cargo build --release --no-default-features --features api
```

Either way, the binary lands at `target\release\memkeeper.exe`.

## Quick check

```powershell
$bin   = ".\target\release\memkeeper.exe"
$store = "$env:USERPROFILE\.memkeeper\store.sqlite"   # also the default when --store is omitted

& $bin init     --store $store --json
& $bin remember --store $store --json '{"content":"windows test memory"}'
& $bin search   --store $store --json '{"query":"test","limit":1}'
& $bin serve --http      # dashboard at http://127.0.0.1:7777 (Ctrl+C to stop)
```

(Inline JSON can be fragile in **Windows PowerShell 5.1** — if you see
`invalid JSON: key must be a string`, pass the payload from a file or stdin instead:
`--json @payload.json` or `... | memkeeper remember --json -`. A brand-new store's
**Graph** tab is expected to be empty until entities and relationships accrue — see
"The dashboard" in the main README.)

## Troubleshooting

The first two are **generic Rust-on-Windows** issues (not memkeeper-specific) but
the most likely to block a first build; the rest are memkeeper/Windows specifics.

- **`error: linker \`link.exe\` not found`** — the MSVC C++ build tools aren't
  installed, or aren't visible in this shell. Install "Desktop development with C++"
  (see Prerequisites) and retry in a fresh terminal. `rustc` locates the Visual
  Studio install automatically (via `vswhere`), so no manual `PATH` setup is needed.

- **cargo fails to fetch crates with a TLS error such as `SEC_E_NO_CREDENTIALS`** —
  cargo's Windows TLS stack (Schannel) is failing a certificate-revocation check it
  can't complete. This is common on locked-down or freshly-provisioned machines that
  can't reach the revocation servers, and is unrelated to your general connectivity
  (other tools such as Node may download fine because they use a different TLS
  stack). Disable the revocation check for cargo:

  ```powershell
  $env:CARGO_HTTP_CHECK_REVOKE = "false"
  # to persist across shells:  setx CARGO_HTTP_CHECK_REVOKE false   (then reopen the shell)
  ```

  If it still fails, also try `$env:CARGO_NET_GIT_FETCH_WITH_CLI = "true"`.

- **Existing memories don't appear in semantic search after you configure models** —
  embeddings are computed at write time, so anything stored before the models were
  present stays lexical-only until backfilled. Run `memkeeper reindex --embed` once.

- **Pointing the dashboard at a store** — `serve --http` accepts `--store <path>`
  (or set `MEMKEEPER_STORE`); otherwise it uses the default
  `%USERPROFILE%\.memkeeper\store.sqlite`.

- **Inline JSON errors in Windows PowerShell 5.1** (`invalid JSON: key must be a
  string`) — 5.1 mangles quotes in `--json '{...}'`. Pass the payload from a file or
  stdin: `--json @payload.json`, or `... | memkeeper remember --json -`.

- **`Access is denied` when rebuilding** (cargo can't replace `memkeeper.exe`) — a
  running dashboard/daemon is holding the binary open. Stop it first, then rebuild.

- **git reports "dubious ownership"** (sandboxed or admin-shifted setups) — run
  `git config --global --add safe.directory <repo path>`.

## Reporting issues

Windows is best-effort. If a build or command fails after the prerequisites above
are in place, please open an issue with: the exact command, the full error output,
your Windows version, and `rustc -Vv`. Pull requests that improve Windows support
are welcome.
