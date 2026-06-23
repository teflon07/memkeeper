# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
suspected vulnerability.

Use GitHub's private vulnerability reporting: on this repository, go to the
**Security** tab → **Report a vulnerability**. This opens a private advisory
visible only to the maintainers.

If you cannot use GitHub's reporting flow, email **security@memkeeper.ai**
instead.

Please include:
- a description of the issue and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected version / commit.

We aim to acknowledge a report within **7 days** and to provide a remediation
timeline after triage. Coordinated disclosure is appreciated: please give us a
reasonable window to ship a fix before any public discussion.

## Supported versions

memkeeper is pre-1.0 and under active development. Security fixes are applied to
the latest release and `main`; older pre-1.0 versions are not separately
maintained.

## Scope

memkeeper is a **local-first** engine: by default it stores data in a local
SQLite file and binds its daemon to a Unix socket / loopback only. Network egress
happens only for a user-configured embedding/rerank endpoint. Reports that
require an attacker to already control the host, the config, or the data
directory are generally out of scope; reports about unintended network exposure,
data disclosure, or memory-safety issues are in scope.
