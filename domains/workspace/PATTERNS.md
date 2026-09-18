# Pattern Library (reference, not an allowlist)

Known mechanism families for agent scaffolds, with concrete format examples
and generic engineering traps. You may adopt, adapt, ignore, or invent
formats not listed here.

## 1. ACE-style playbook (incremental curated context)

Format: a single text artifact of numbered bullets, each with an id and
helpful/harmful counters; updated INCREMENTALLY (add/increment/demote one
bullet at a time), never rewritten wholesale.

```
[pb-014] (helpful=6, harmful=1) When a rubric asks for a recommendation,
state the recommendation in the first sentence, then support it.
```

Generic trap: "context collapse" — regenerating the whole playbook each
update shrinks and blurs it. Only incremental edits.

## 2. Text reference skills (triggered prose snippets): tend to wash out

WARNING: a scaffold carrying ~20 of these typically sees a zero-to-negative
net effect on a fixed policy, and some actively cause new failures (a forced
re-check that picks the wrong branch). Treat this family as a last resort;
prefer family 3 (executable skills).

Format: files with frontmatter (name, trigger conditions) + prose body;
injected when trigger matches the task.

```
---
name: financial-model-sanity
trigger: task mentions valuation, DCF, or model outputs
---
Before finalizing, re-derive the headline number from raw inputs...
```

Generic trap: a skill that is injected but never changes behavior is dead
weight in the context window — bind skills to checkable steps where
possible.

## 3. Executable skills (Voyager-style)

Format: each skill = a runnable function/script the agent can invoke via
code_exec, plus a one-line docstring used for retrieval. Skills DO work
(compute, parse, verify) instead of advising.

```python
def verify_totals(sheet_path):
    """Recompute column totals and compare against stated summary values."""
```

Cost note: code must run inside the sandbox via code_exec, so keep skills
dependency-free and defensive.

## 4. Per-task episodic memory (append-only JSONL)

Format: one record per completed task, filtered on read.

```json
{"domain": "Law", "lesson": "jurisdiction questions: check the specialized
court's exclusive jurisdiction before answering venue", "evidence": "lost a
criterion by answering venue from the general rule"}
```

Iron rule: a write path without a read path that changes behavior is dead
storage. Filter reads (by domain or trigger), never dump the whole file
into context. Keep records entity-free.

## 5. Workflow/state-machine mechanisms (no persistence)

Examples: forced finalize-checklist before final_answer; a todo scratchpad
the loop re-injects each step; mandatory verification pass when the answer
contains computed numbers; structured re-read of the task prompt before
finalizing. These change control flow only — no state dir needed.

## 6. Context-budget mechanisms

Examples: smarter compression triggers, tool-result summarization tuned to
what later steps actually cite, protecting key artifacts (task prompt,
todo, computed values) from compaction. Note: compression-path changes
affect every long task, so test their effect broadly.
