# Pattern Library (reference, not an allowlist)

Known mechanism families for terminal-agent scaffolds, with concrete
examples and generic engineering traps. You may adopt, adapt, ignore, or
invent formats not listed here.

## 1. Waiting / polling mechanics

The #1 terminal-agent tax: fixed `duration` guesses on long-running
commands (builds, installs, downloads) waste steps or truncate work.
Mechanisms: detect a still-running foreground process and poll with empty
keystrokes; teach (via template) `&& echo DONE_MARKER` sentinels; convert
one long wait into bounded re-checks.

Generic trap: polling forever on a genuinely hung command — always pair a
polling mechanism with a stuck-state escape (bounded retries, then C-c).

## 2. Output plumbing (truncation, protection, post-processing)

Terminal output is truncated (head+tail) before entering context.
Mechanisms: smarter truncation that preserves error lines / last command;
extracting and pinning compiler-error summaries; collapsing progress-bar
spam; protecting the task description and key computed values from
summarization.

Generic trap: expanding caps wholesale blows the context budget on every
long task — prefer selective preservation over bigger limits.

## 3. Workflow/state-machine mechanisms (no persistence)

Examples: a verification pass before the second task_complete
confirmation ("run the artifact/tests you were asked to produce once,
report what you see"); a corrective bounce when N consecutive episodes
produce identical keystrokes (loop detector); structured re-read of the
task description before declaring completion; a scratchpad of
discovered facts the loop re-injects each episode.

Generic trap: unbounded "check again" loops — see hard rule 8; every
bounce needs a max-fires counter.

## 4. Parse-error recovery

The policy replies in strict JSON; malformed replies burn episodes.
Mechanisms: better corrective prompts (show the exact parse error and a
minimal valid example); salvage of truncated responses; auto-repair of
common JSON slips (trailing commas, unescaped newlines in keystrokes)
before rejecting.

Generic trap: silently "fixing" JSON can change intended keystrokes —
salvage conservatively, reject ambiguous repairs.

## 5. Context-budget mechanisms

Examples: summarization that preserves an explicit state block (task goal,
what's done, what's pending, key file paths, error messages); episode
pruning that keeps first-and-recent instead of recent-only; token-aware
truncation of old terminal outputs before whole-history summarization.

Generic trap: "context collapse" — a summary of a summary loses the task;
always regenerate the state block from pinned originals, not from the
previous summary.

## 6. Extra policy calls (bounded sub-calls)

Examples: pre-completion verifier brief ("list concrete evidence each task
requirement is met; answer only MET/UNMET per requirement"); a one-shot
plan critique at episode 1 for complex tasks; best-of-2 command selection
when the last episode errored.

Cost note: every sub-call adds latency on the same frozen policy; keep
briefs tight and outputs short. Never delegate the whole task.
