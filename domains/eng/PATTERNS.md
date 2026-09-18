# Pattern Library (reference, not an allowlist)

Shapes that mechanisms tend to take, with what is already known about each.
**This is not a menu to pick from.** It exists so you do not spend a round
rediscovering something already measured elsewhere, and so you can see what a
mechanism at each lever actually looks like. Inventing something absent from
this file is the preferred outcome.

Every entry is subject to the litmus test in SKILL.md: *would this help a
competent engineer working on MANY unfamiliar design problems across unrelated
disciplines?*

---

## 1. Submission-contract compliance

The graded artefact is a pure-literal `PAYLOAD` dict matching a pydantic schema
that is sitting in the workspace. A design can be entirely correct and still
score zero because a field was named wrong, nested at the wrong level, left as a
numpy scalar, or written as an expression instead of a value.

This is the class where a MECHANISM beats prose by the widest margin, because
the check is fully mechanical and needs no domain knowledge: parse the written
file with `ast.literal_eval`, load the task's own `output_structure.py`,
validate, and hand back what is missing. That is a client tool (lever 9) or a
control-flow hook at finalize (lever 2).

**Not this:** restating the contract more emphatically in the prompt. The
contract is already stated in the task wrapper; a policy that violated it did
not need to be told again, it needed to be shown its own output.

## 2. Does the agent ever evaluate its own design?

The environment gives the agent a full scientific Python. Nothing in v0 makes it
simulate, sanity-check, or re-derive the design it is about to submit, and
nothing tells it when to.

Before proposing a mechanism here, check the traces for which half of the
question is true: does the agent compute a candidate and never test it, or does
it test it, see it fall short, and submit anyway? Those want different fixes,
and only the first is a missing-mechanism story.

Bounded is mandatory (hard rule 8): "at most one targeted verification pass" is
a mechanism, "check your work carefully" is a wish and will wash out.

### Two design dimensions, judged separately

A mechanism in this family has TWO independent design dimensions, and a single
edit-history entry only speaks to one of them. Conflating them will cost you rounds.

* **What it checks** — a semantic question ("does this design meet the target?")
  versus a mechanical one ("is what I wrote consistent with what I computed?").
* **What it costs** — whether it fires on EVERY trial regardless of evidence, or
  only when something in the run itself says it should.

The expensive corner of this space is an ALWAYS-ON, SEMANTIC pre-finalize gate:
it typically adds a third or more tokens per trial, rarely raises the pass
count, and can multiply the invalid rate by pushing the agent to rework designs
that were already valid. If the edit history already rejects that combination,
re-skinning it is barred.

That does NOT settle the same family at a different cost shape: a CHEAP or a
CONDITIONAL check is a different candidate. Note also that the behaviour itself
pays when it happens: runs that check something before packaging tend to score.
So the evidence reads "this behaviour helps, and one expensive way of forcing
it does not pay for itself", not "checking is refuted".

The general principle, which applies to every family in this file: **a mechanism
that is refuted at one cost is not thereby refuted at another.** When the edit history
rejects something for its token growth rather than its direction, the open
question is whether the same direction can be reached for a fraction of the
budget — by firing on a trigger the run itself provides, by acting on data the
run already produced rather than generating more, or by moving the intervention
from the end of the run into the middle of it. Say explicitly which of the two
dimensions above your candidate changes relative to the refuted one.

## 3. `code_exec` failure handling

Every design run is a chain of code executions. What happens when one errors --
a missing import, a shape mismatch, a solver that will not converge -- is
currently nothing in particular: the error text goes into context and the policy
improvises. Repeated identical failures, and the step budget they burn, are
worth measuring in the traces before assuming.

Watch for the specific trap: the jail has NO NETWORK, so any recovery path the
policy invents that involves installing a package hangs and then fails. A
mechanism that detects and short-circuits that is legitimate and general.

## 4. Termination and the step budget

A run that never writes the payload scores zero regardless of how good the
engineering was. Look for runs that end at the step cap or the wall clock with
work in progress. The lever is control flow as the budget runs down: something
that ensures a best-effort submission exists early and is refined, rather than
one written at the end or not at all.

Be careful about the reverse failure: a mechanism that makes the agent submit
early and stop working trades UNDER TARGET for a worse score. Writing early is
only safe if the agent keeps improving afterwards.

## 5. Output plumbing and the carried number

The recurring shape: the agent computes the right value in a `code_exec` result,
the result is long, and by the time it writes the payload the number it needs
has been truncated or compressed away. This is lever 4 (tool_result.py) or
lever 5 (resum.py) -- routing the right content in -- and it is invisible if you
read only the policy's reasoning rather than what actually entered its context.

## 6. Context budget on long runs

`resum.py` decides what survives when context fills. On a long design run the
constraint list from the problem statement, the schema, and the numbers already
computed are exactly what later steps need. Whether they survive is checkable in
the traces, and protecting the right things is a legitimate mechanism.
Compression-path changes affect every long task, so reason about breadth.

## 7. Progressive-disclosure skills

`skills/<name>/SKILL.md` (inside the harness package) plus a `skill_use` meta-tool. Suitable for a
recurring WRONG-METHOD mode -- a procedure the policy would follow correctly if
it knew it. PRIOR: purely textual reference skills tend to WASH OUT; a scaffold can carry
twenty of them for a net effect of zero to negative. Only worth it if the procedure is concrete and checkable,
or is backed by an executable helper the skill tells the model to run.

## 8. Bounded extra policy calls

`mechanisms.subcall` gives one tightly-briefed extra call to the same frozen
policy. Suitable for a narrow extraction or selection ("here is the schema and
the value I computed; return ONLY the literal dict"). Not suitable for
open-ended reasoning or delegating a task. PRIOR: sub-calls tend to hurt on
smaller policies (early exit, added latency without payoff); the policy here is
strong, which makes this less certain, but it is still a cost the gain has to
cover.

## 9. Memory across trials

`mechanisms.Memory`, fcntl-locked JSONL. Entity-free general lessons ONLY.
Iron rule: a write path with no read path that changes
behaviour is dead storage, and dumping the whole file into context is worse than
not having it. Filter reads by trigger. Anything task-specific is memorization
and the critic rejects it -- and here that is not a technicality, since the
harness is evolved and scored on the same 61 tasks.
