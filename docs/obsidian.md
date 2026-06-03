# Obsidian Vaults

LEANN can index an Obsidian vault as a local Markdown knowledge base. The Obsidian example keeps
the normal LEANN RAG flow while preserving note-specific metadata for filters and downstream tools.

## Usage

```bash
python -m apps.obsidian_rag \
  --vault-dir ~/Notes \
  --query "What did I learn about local-first retrieval?"
```

Run without `--query` to start the shared interactive RAG loop:

```bash
python -m apps.obsidian_rag --vault-dir ~/Notes
```

By default, LEANN indexes Markdown files under the vault and skips Obsidian internals such as
`.obsidian/`, `.trash/`, and `.git/`. Hidden folders are skipped unless `--include-hidden` is set.

## Captured Metadata

Each indexed note carries metadata that can be used by search result renderers, agents, or metadata
filters:

- `obsidian_note`: marks the passage as coming from an Obsidian vault.
- `obsidian_vault_path`: absolute path to the vault.
- `obsidian_relative_path`: note path relative to the vault root.
- `obsidian_title`: frontmatter `title` when present, otherwise the Markdown file stem.
- `obsidian_aliases`: aliases parsed from frontmatter.
- `obsidian_tags`: tags from frontmatter and inline `#tags`, without the leading `#`.
- `obsidian_links`: wiki-link targets from `[[Target]]` and `[[Target|Alias]]`.
- `obsidian_embeds`: embedded wiki targets from `![[Attachment]]`.
- `obsidian_frontmatter`: parsed frontmatter values.

## Current Scope

This is a Markdown vault ingestion foundation. It does not install an Obsidian plugin, read
Obsidian's internal workspace state, index binary attachments, or run live vault synchronization.
For always-current local indexes, combine this with the normal LEANN rebuild or watch workflows as
they become available for app-created indexes.
