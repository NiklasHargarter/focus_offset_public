# Focus Offset — Agent Rules

## Always
- Use `uv run` for all Python execution. Never use `pip` or `python` directly.
- Only run on `--tier small` during development. Never `--tier normal` or `--tier full` without explicit user instruction.
- Read `AGENT_EXECUTION_GUIDE.md` and the relevant phase plan before starting any phase work.
- Never modify files in directories containing a `LEGACY.md` marker file. Those directories are reference-only.
- Never proceed to the next phase autonomously. Stop after producing the phase completion artifact and wait for the user.

## Known Gotchas
- VSI files produce noisy stderr warnings. Use the `suppress_stderr` helper from `focus_offset.utils.io_utils` when reading `.vsi` files with `slideio`.

## Self-Governance
- If you discover a persistent environment quirk or mistake pattern, append it to `## Known Gotchas` in this file.
- Scientific discoveries (confirmed z-step values, dataset anomalies, experimental findings) belong in `research.md`, not here.
