# Changelog

## 0.2.0 — 2026-11-14

- Unify streamed and non-streamed agent execution.
- Preserve provider-specific reasoning state and reject incomplete/error responses.
- Select provider/key/endpoint as a coherent configuration; require an explicit model.
- Add SQLite conversation continuity and session/workspace/model binding.
- Move write approval to CLI/TUI; keep Core responsive and cancel pending approvals on disconnect.
- Add hash-checked, diff-reviewed atomic writes and exact-text patching.
- Add paginated file reads, literal search, glob and incremental reuse in local TF-IDF retrieval.
- Apply a shared filesystem boundary to direct tools, traversal and retrieval.
- Isolate run budgets and use UUID run IDs.
- Load the built-in repo-review workflow explicitly with --skill.
- Add installable command entry points, locked test dependencies, CI and offline demo.

Migration: select AGENT_ASSISTANT_PROVIDER and a model in .env. Existing explicit Anthropic-compatible endpoint variables remain supported, but native Anthropic keys now default to Anthropic's endpoint. For a gateway, prefer provider=custom with an explicit URL and protocol. Existing file replacements require the old sha256 returned by read_file. Runtime state now lives in .agent-state (or AGENT_ASSISTANT_STATE_DIR), not the target workspace.

This release persists completed conversations. It does not replay an interrupted task or provide MCP/shell execution.
