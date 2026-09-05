# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/StarTrail-org/LEANN/security/advisories/new)
(repository → **Security** → **Advisories** → *Report a vulnerability*).

That opens a private advisory visible only to you and the maintainers. It needs
no infrastructure on either side, and it gives us a place to discuss a fix and
issue a CVE if one is warranted.

If the advisory form is unavailable to you, open a public issue containing only
that you have a security report and how to reach you — no details — and a
maintainer will arrange a private channel.

## What to include

The more of this you can provide, the faster a fix lands:

- what an attacker can achieve, and what access they need to start
- affected versions, and the platform and Python version you saw it on
- a reproduction — a script, or the commands you ran
- anything you already know about the cause or a possible fix

## Scope

LEANN runs locally and is usually pointed at a user's own data, so the
interesting boundaries are the ones where it stops being local:

- the backend embedding servers, which speak an **unauthenticated ZMQ REP
  protocol** — they bind `127.0.0.1` by default, and anything that can reach
  the port can request embeddings (see
  [Embedding Server Bind Address](docs/configuration-guide.md))
- the MCP server and its stdio transport
- document readers and parsers, which handle untrusted input by design
- index and metadata files written to disk, and anything that reads a path out
  of them
- credentials for third-party providers — API keys, endpoints, tokens

Out of scope: findings against a deployment you have deliberately exposed to a
network (for example `LEANN_EMBEDDING_SERVER_HOST=0.0.0.0` reachable from the
internet), and vulnerabilities in third-party dependencies that are already
public — please report those upstream, though we do want to know if LEANN
pins an affected version.

## Supported versions

Fixes land on `main` and ship in the next release. If you are on an older
release, please confirm the issue reproduces on current `main` where you can.

## Disclosure

We will confirm receipt, keep you updated as we investigate, and credit you in
the advisory unless you would rather stay anonymous. Please give us a
reasonable window to ship a fix before disclosing publicly.
