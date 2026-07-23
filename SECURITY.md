# Security Policy

## Supported Versions

Security fixes are applied to the `main` branch only. There are currently no versioned release branches.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability — for example, a prompt injection bypass that circumvents the guardrail, an API key exposure risk, or a dependency with a known CVE — please report it privately by contacting the maintainers directly.

When reporting, please include:

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce the issue.
- Any suggested mitigations, if you have them.

You can expect an acknowledgement within **72 hours** and a resolution or status update within **14 days**.

## Scope

The following are within scope for security reports:

- Guardrail bypass vulnerabilities (prompt injection, jailbreaks that evade the `OUT_OF_SCOPE` classifier)
- API key leakage through logs, error messages, or the Streamlit UI
- Dependency vulnerabilities (`requirements.txt`) with a CVSS score of High or Critical
- Unauthorized access to the ChromaDB vector store or SQLite cache

The following are **out of scope**:

- Vulnerabilities in third-party services (OpenAI, Brave Search) — report those directly to the respective vendors.
- Denial-of-service via extremely long queries (the system has no rate limiting by design; this is a known limitation).

## Dependency Updates

Dependencies are pinned in `requirements.txt`. If you identify a dependency with a known CVE, please open a security report rather than a public issue.
