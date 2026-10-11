# Security policy

## Supported versions

This is a portfolio and training project with no releases. Only the latest
commit on `main` is maintained; fixes land there and nowhere else.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting instead: the **Security** tab of this repository,
then **Report a vulnerability**. If that button is not available, open an issue
titled "Security contact" with no details, and a private channel will be
arranged.

Include what you found, how to reproduce it, and what an attacker gains. You can
expect a first answer within seven days; a confirmed issue is fixed on `main`
and credited in the fix unless you prefer otherwise.

## Secrets

Every key and token is read from a local `.env` that git ignores; `.env.example`
lists the names with empty values. A credential that appears in this
repository's history is a vulnerability in its own right: report it the same
way, and it will be revoked rather than merely deleted.

## What is in scope here

- Side effects (saving a report) pause the agent with `interrupt()` and resume
  only after approval; the approval endpoint is protected by `RR_APPROVE_TOKEN`.
- In scope: an approval without the token, an agent run that escapes its step
  or token budget, a tool call outside the declared schema, and prompt
  injection through retrieved documents.
