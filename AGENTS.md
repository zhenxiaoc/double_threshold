# Workspace Instructions

## Local Paper-Explainer Skill

When the user asks to use `paper-explainer`, `$paper-explainer`, or "the paper-explainer skill", use the session-provided skill if it appears in the active skill list.

If `paper-explainer` is missing from the active skill list, do not conclude that the skill is unavailable until checking the local installed skill file:

`%USERPROFILE%\.codex\skills\paper-explainer\SKILL.md`

If that file exists, read it and follow it as the authoritative `paper-explainer` workflow for the turn. The local skill build source is also kept in:

`.codex-skill-build\paper-explainer`

Prefer the installed skill path under `.codex\skills` when both exist.

For this JMP workspace, the default `paper-explainer` profile writes Obsidian literature notes to:

`JMP Notes\Understanding Literature\`

and maintains:

`JMP Notes\Understanding Literature\Reference List - Summarized Papers.md`
