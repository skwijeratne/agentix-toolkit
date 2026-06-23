# Security Policy

agentix is a toolkit for building agents, with a security/governance subsystem
(trust boundary, permission tiers, PII/injection guards). We take security
issues seriously — both vulnerabilities in the library itself and weaknesses in
the guards that could let an agent be misused.

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues or pull requests.**

Instead, report privately via either:

- GitHub's **[Private vulnerability reporting](https://github.com/skwijeratne/agentix-toolkit/security/advisories/new)**
  (Security tab → "Report a vulnerability"), or
- Email **skwijeratne@gmail.com** with the subject `agentix security`.

Please include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal proof of concept if possible).
- Affected version(s).

We'll acknowledge your report within a few days, keep you updated on progress,
and credit you in the release notes unless you prefer to remain anonymous.

## In scope

- A guard that can be trivially bypassed (e.g. injection content that defeats
  `InjectionGuard`, PII that slips past the redaction/URL guards).
- The trust boundary failing to mark tool output as untrusted.
- Confirmation/tier enforcement not being applied where the policy says it must.
- Any path that lets tool output reach an outbound endpoint without the
  configured guards running.

## Out of scope

- Misconfiguration by the user (e.g. running with no guards, or marking a
  dangerous tool as `auto_ok`).
- The behavior of the underlying model or a third-party tool/MCP server.
- Hardening that the docs explicitly leave to the caller (e.g. sandboxing the
  tool executor, a shared store for multi-process fleets).

## Supported versions

agentix is pre-1.0; security fixes are made against the latest release.
