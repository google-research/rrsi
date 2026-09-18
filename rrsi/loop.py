# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One round of RRSI: Algorithm 1 (proposal side) then Algorithm 2 (selection
side), for round t = 0..T-1 of one domain.

    run.round(t)
      1. F_t   <- Analyze(H_t, D_evolve)                     analyst over the incumbent's own eval
      2. b_t   <- annealed edit budget                         schedule.edit_budget
      3. sigma_t, T_t, U_t, E_t, B_t                            history.stall_flag / History
      4. for each of m variants (own worktree + branch):
             C_t^(v) ~ P_reg(. | H_t, F_t, L_t, b_t, E_t)      propose.propose
             tag edits (component, hypothesis, diff)
             Critic(H_t, C_t^(v)) with bounded repair          critic.review
             liveness smoke (compile / construct / few tasks)   domain.smoke
      5. Evaluate(H', D_evolve, k) for the screened set        evaluate.evaluate (parallel optional)
      6. admissibility, argmax, S*, history records, B_t       select.select_round
      7. fast-forward evolve/<domain> to the winner, update frontier

Everything written under runs/<domain>/ is resume-safe: a round that crashed
after evaluating variant A reuses A's eval.json when re-run.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import gitops as G
from .analyst import analyze, load_digests
from .calibrate import calibrate as _calibrate, write as _write_cal
from .components import normalize
from .config import RRSIConfig
from .critic import review
from .evaluate import EvalResult, evaluate
from .history import History, exploration, stall_flag
from .propose import propose
from .schedule import edit_budget
from .selection import Candidate, select_round

VARIANT_LABELS = "ABCDEFGH"


def log(domain: str, msg: str) -> None:
    print(f"[rrsi:{domain}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class Run:
    """Paths, state and git plumbing of one domain's evolution."""

    def __init__(self, domain, cfg: RRSIConfig, repo: Path, runs_root: Path):
        self.domain, self.cfg, self.repo = domain, cfg, Path(repo)
        self.runs = Path(runs_root) / domain.name
        self.jobs = self.runs / "jobs"
        self.wt_root = self.runs / "wt"
        self.frontier_path = self.runs / "frontier.json"
        self.calibration_path = self.runs / "calibration.json"
        self.global_analysis = self.runs / "global_analysis.json"
        self.attribution_path = self.runs / "attribution.jsonl"
        self.history = History(self.runs / "history.jsonl")
        self.branch = f"evolve/{domain.name}"
        self.harness_rel = os.path.normpath(f"domains/{domain.name}/{domain.harness_path}")
        for d in (self.runs, self.jobs, self.wt_root, self.runs / "logs"):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ state ---
    def frontier(self) -> dict:
        if not self.frontier_path.exists():
            raise SystemExit(f"no {self.frontier_path}; run `baseline` first")
        return json.loads(self.frontier_path.read_text())

    def save_frontier(self, fr: dict) -> None:
        self.frontier_path.write_text(json.dumps(fr, indent=1))

    def delta(self) -> float:
        if self.cfg.delta is not None:
            return float(self.cfg.delta)
        if self.calibration_path.exists():
            return float(json.loads(self.calibration_path.read_text())["delta"])
        raise SystemExit("no noise band: set cfg.delta or run `calibrate`")

    def eval_path(self, job: str) -> Path:
        return self.jobs / job / "eval.json"

    def incumbent_eval(self, fr: dict) -> EvalResult:
        return EvalResult.load(self.eval_path(fr["incumbent"]["job"]))

    # -------------------------------------------------------------- git ---
    def ensure_branch(self) -> None:
        G.ensure_branch(self.repo, self.branch, "HEAD")

    def checkout(self, name: str, start: str, branch: str | None = None) -> Path:
        """Worktree `name` at `start`; on a new branch when `branch` is given,
        detached otherwise."""
        path = self.wt_root / name
        if branch:
            return G.worktree_add(self.repo, path, branch, start)
        G.worktree_remove(self.repo, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        G.git(self.repo, "worktree", "add", "--detach", str(path), start, check=True)
        return path

    def harness_tree(self, ref: str) -> str:
        return G.tree_hash(self.repo, ref, self.harness_rel)

    # --------------------------------------------------------- evidence ---
    def build_traces(self, job: str, per_task: dict) -> dict:
        """Worst trial of the lowest-scoring tasks plus the best trial of the
        highest-scoring ones (success habits the proposer must not break)."""
        ranked = sorted(per_task.items(), key=lambda kv: kv[1].mean)
        fails = [t for t, _ in ranked[:self.cfg.n_fail_traces]]
        wins = [t for t, _ in ranked[-self.cfg.n_success_traces:] if t not in fails]
        traces = {}
        for tid in fails + wins:
            tr = per_task[tid]
            if not tr.rewards:
                continue
            idx = (tr.rewards.index(min(tr.rewards)) if tid in fails
                   else tr.rewards.index(max(tr.rewards)))
            rec = self.domain.load_trial(self.runs, job, tid, idx)
            if rec is not None:
                traces[tid] = rec
        return traces

    def scoreboard(self) -> list:
        if not self.attribution_path.exists():
            return []
        rows = [json.loads(l) for l in self.attribution_path.read_text().splitlines()
                if l.strip()]
        return rows[-20:]

    def attribute(self, t: int, variant: str, edits: list, inc: EvalResult,
                  cand: EvalResult) -> list:
        thr = self.domain.regression_threshold(self.cfg.k)
        rows = []
        for e in edits:
            pred = [str(p) for p in (e.get("predicted_affected") or [])
                    if str(p) in inc.per_task and str(p) in cand.per_task]
            hits = [p for p in pred if cand.per_task[p].mean > inc.per_task[p].mean]
            drops = [x for x in inc.per_task if x in cand.per_task
                     and cand.per_task[x].mean - inc.per_task[x].mean <= -thr]
            rows.append({"t": t, "variant": variant, "edit_id": e.get("id"),
                         "component": e.get("component"),
                         "hypothesis": str(e.get("hypothesis"))[:160],
                         "n_predicted": len(pred), "predicted_hit": hits,
                         "hit_rate": round(len(hits) / len(pred), 2) if pred else None,
                         "unpredicted_regressions": [d for d in drops if d not in pred][:12]})
        with open(self.attribution_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows

    # ----------------------------------------------------------- baseline --
    def baseline(self, job: str = "base") -> EvalResult:
        """Evaluate H_0 (the tip of evolve/<domain>) and seed the frontier."""
        self.ensure_branch()
        wt = self.checkout("incumbent", self.branch)
        ids = self.domain.evolve_ids()
        log(self.domain.name, f"baseline: {len(ids)} tasks x k={self.cfg.k} -> job {job}")
        ev = evaluate(self.domain, wt, self.runs, job, ids, self.cfg.k,
                      log_prefix=f"{job}")
        ev.save(self.eval_path(job))
        if ev.missing > self.cfg.invalid_missing_frac * ev.n_expected:
            raise SystemExit(f"baseline invalid: {ev.missing}/{ev.n_expected} trials missing")
        commit = G.rev(self.repo, self.branch)
        fr = {"domain": self.domain.name,
              "incumbent": {"t": 0, "commit": commit,
                            "harness_tree": self.harness_tree(self.branch),
                            "job": job, "S": ev.S, "C": ev.C, "extra": ev.extra},
              "S_star": ev.S,
              "trajectory": [{"t": 0, "S": ev.S, "C": ev.C, "commit": commit, "job": job}],
              "config": self.cfg.dump()}
        self.save_frontier(fr)
        self.history.append({"t": 0, "variant": "-", "edit_id": None, "component": None,
                             "hypothesis": "H_0 baseline", "outcome": "BASELINE",
                             "S": ev.S, "C": ev.C, "accepted": True,
                             "delta_S": None, "delta_C": None, "bundle": 0})
        log(self.domain.name, f"baseline S={ev.S:.4f} C={ev.C} missing={ev.missing}"
            f"/{ev.n_expected} {json.dumps(ev.extra)[:200]}")
        return ev

    def calibrate(self, jobs: list[str]) -> dict:
        evals = [EvalResult.load(self.eval_path(j)) for j in jobs]
        cal = _calibrate(evals, z=self.cfg.delta_z)
        cal["jobs"] = jobs
        _write_cal(self.calibration_path, cal)
        log(self.domain.name, f"calibrated delta={cal['delta']:.5f} "
            f"(sd_null {cal['sd_null']:.5f}, z={cal['z']}, {cal['method']})")
        return cal

    def heldout(self, label: str, ids: list[str], ref: str | None = None,
                k: int | None = None) -> EvalResult:
        ref = ref or self.branch
        wt = self.checkout(f"heldout_{label}", ref)
        job = f"heldout_{label}"
        ev = evaluate(self.domain, wt, self.runs, job, ids, k or self.cfg.k,
                      log_prefix=job)
        ev.save(self.eval_path(job))
        log(self.domain.name, f"heldout {label} @ {G.rev(self.repo, ref)}: S={ev.S:.4f} "
            f"C={ev.C} missing={ev.missing}/{ev.n_expected} {json.dumps(ev.extra)[:200]}")
        return ev

    # -------------------------------------------------------------- round --
    def round(self, t: int, dry_run: bool = False) -> None:
        d, cfg = self.domain, self.cfg
        fr = self.frontier()
        inc = fr["incumbent"]
        if inc["harness_tree"] != self.harness_tree(self.branch):
            raise SystemExit(f"harness tree of {self.branch} != frontier "
                             f"({self.harness_tree(self.branch)} vs {inc['harness_tree']})")
        if len(fr["trajectory"]) < t + 1:
            raise SystemExit(f"round {t} needs trajectory up to t={t}; have "
                             f"{len(fr['trajectory'])} entries (run earlier rounds)")
        delta = self.delta()
        rdir = self.runs / f"r{t}"
        rdir.mkdir(exist_ok=True)
        ids = d.evolve_ids()
        inc_ev = self.incumbent_eval(fr)
        log(d.name, f"=== round {t}/{cfg.T} incumbent t={inc['t']} {inc['commit']} "
            f"S={inc_ev.S:.4f} C={inc_ev.C} S*={fr['S_star']:.4f} delta={delta:.5f} ===")

        # 1) F_t <- Analyze(H_t, D_evolve)
        traces = self.build_traces(inc["job"], inc_ev.per_task)
        if len(traces) < 0.5 * min(len(ids), cfg.n_fail_traces + cfg.n_success_traces):
            raise SystemExit(f"only {len(traces)} traces available from {inc['job']}")
        report_path = rdir / "analysis_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            log(d.name, "reusing analysis_report.json")
        else:
            prior = (json.loads(self.global_analysis.read_text())
                     if self.global_analysis.exists() else {})
            report = analyze(d, traces, inc_ev.per_task, rdir,
                             prior_modes=prior.get("failure_modes"),
                             prior_habits=prior.get("success_habits"),
                             model=cfg.analyst_model)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        self.global_analysis.write_text(json.dumps(
            {"failure_modes": report.get("failure_modes"),
             "success_habits": report.get("success_habits")}, ensure_ascii=False, indent=1))
        log(d.name, f"top modes: {[m.get('mode') for m in report.get('failure_modes', [])[:5]]}")
        if dry_run:
            log(d.name, "dry-run: stopping before propose")
            return

        # 2-3) b_t, sigma_t, T_t, U_t, E_t, B_t
        budget = edit_budget(t, cfg.T, cfg.b_min, cfg.b_max)
        traj = [x["S"] for x in fr["trajectory"]]
        sigma = stall_flag(traj, t, cfg.w, delta)
        tried = self.history.tried()
        explore = exploration(t, sigma, tried, cfg.m_draft)
        prune = self.history.prune_set(t, cfg.n_prune)
        hist_rows = self.history.render()
        digests = load_digests(rdir)
        skill_md, patterns_md = d.constitution(self.repo)
        log(d.name, f"b_t={budget} sigma_t={sigma} untried={explore['untried']} "
            f"prune={[p['component'] for p in prune]}")
        (rdir / "directives.json").write_text(json.dumps(
            {"t": t, "b_t": budget, "sigma_t": sigma, "tried": sorted(tried),
             "explore": explore, "prune_set": prune, "delta": delta,
             "S_star": fr["S_star"]}, indent=1))

        # 4) draw and screen m candidates
        cands: list[Candidate] = []
        for v in range(cfg.m):
            vid = VARIANT_LABELS[v]
            reserved = bool(sigma and explore["untried"] and v >= cfg.m - cfg.m_draft)
            cands.append(self._draft(t, vid, rdir, report, digests, traces, inc_ev,
                                     budget, explore, reserved, prune, hist_rows,
                                     skill_md, patterns_md))

        # 5) Evaluate(H', D_evolve, k) for the screened set, in parallel
        live = [c for c in cands if c.gate_failure is None]
        with ThreadPoolExecutor(max_workers=max(1, cfg.eval_parallel)) as ex:
            list(ex.map(lambda c: self._evaluate(t, c, rdir, ids), live))

        # 6) Algorithm 2: admissibility, argmax, S*, history
        counts = self.history.incumbent_component_counts()
        winner, decisions = select_round(cands, inc_ev, fr["S_star"], delta, cfg,
                                         counts, guard_fn=d.guards)
        (rdir / "decisions.json").write_text(json.dumps(
            [x.to_json() for x in decisions], indent=1))
        for c, dec in zip(cands, decisions):
            if self.history.has(t, c.variant):
                continue                      # resumed round: already recorded
            if c.ev is None:
                outcome = c.gate_failure or "not_evaluated"
                self.history.append_candidate(t, c.variant, c.edits, outcome, None, None,
                                              False, None, None, c.diff_path, c.detail)
                continue
            outcome = ("ACCEPTED" if c is winner else
                       ("LOST" if dec.admissible else "REJECTED"))
            self.history.append_candidate(t, c.variant, c.edits, outcome, dec.delta_S,
                                          dec.delta_C, c is winner, dec.S, dec.C,
                                          c.diff_path, dec.reason)
            self.attribute(t, c.variant, c.edits, inc_ev, c.ev)
            log(d.name, f"{c.variant}: S={dec.S:.4f} dS={dec.delta_S:+.4f} "
                f"dC={dec.delta_C:+.3f} nu={dec.novelty} -> {outcome}: {dec.reason}")

        # 7) H_{t+1}
        if winner is not None:
            G.fast_forward(self.repo, self.branch, winner.commit)
            new_S, new_C = winner.ev.S, winner.ev.C
            fr["incumbent"] = {"t": t + 1, "commit": winner.commit,
                               "harness_tree": self.harness_tree(winner.commit),
                               "job": winner.ev.job, "S": new_S, "C": new_C,
                               "extra": winner.ev.extra, "variant": winner.variant}
            fr["S_star"] = max(fr["S_star"], new_S)
            log(d.name, f"ACCEPTED r{t}{winner.variant} -> {winner.commit} "
                f"S={new_S:.4f} S*={fr['S_star']:.4f}")
        else:
            new_S, new_C = inc_ev.S, inc_ev.C
            log(d.name, f"no admissible candidate; H_{t+1} = H_{t}")
        traj_entry = {"t": t + 1, "S": new_S, "C": new_C,
                      "commit": fr["incumbent"]["commit"], "job": fr["incumbent"]["job"]}
        fr["trajectory"] = [x for x in fr["trajectory"] if x["t"] <= t] + [traj_entry]
        self.save_frontier(fr)
        for c in cands:
            G.worktree_remove(self.repo, self.wt_root / f"r{t}{c.variant}")

    def readjudicate(self, t: int) -> None:
        """Re-run Algorithm 2 on the STORED measurements of round t, after a
        change of delta or of the acceptance weights. No new evaluation is
        spent: every candidate keeps the (S', C') it was measured at against
        the incumbent H_t it was drafted from. Rounds after t must have been
        removed first, since their candidates were drafted from the old H_{t+1}."""
        d, cfg = self.domain, self.cfg
        fr = self.frontier()
        if len(fr["trajectory"]) > t + 2:
            raise SystemExit(f"rounds after {t} exist in the frontier; remove them first")
        inc_entry = fr["trajectory"][t]
        inc_job = inc_entry.get("job") or ("base" if t == 0 else None)
        if not inc_job:
            raise SystemExit(f"trajectory[{t}] has no job; cannot locate H_{t}'s evaluation")
        inc_ev = EvalResult.load(self.eval_path(inc_job))
        S_star = max(x["S"] for x in fr["trajectory"][:t + 1])
        delta = self.delta()
        rdir = self.runs / f"r{t}"
        cands = []
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir() and len(p.name) == 1):
            prep = (json.loads((vdir / "prep.json").read_text())
                    if (vdir / "prep.json").exists() else {})
            c = Candidate(vdir.name, prep.get("edits") or [], diff_path=prep.get("diff_path"),
                          branch=prep.get("branch"), commit=prep.get("commit"),
                          gate_failure=prep.get("gate_failure"), detail=prep.get("detail", ""))
            if (vdir / "eval.json").exists():
                c.ev = EvalResult.load(vdir / "eval.json")
            elif not c.gate_failure:
                c.gate_failure = "eval_invalid"
            cands.append(c)
        counts = {k: 0 for k in self.history.incumbent_component_counts()}
        for r in self.history.records():
            if r.get("accepted") and r.get("t", 10**9) < t and r.get("component") in counts:
                counts[r["component"]] += 1
        winner, decisions = select_round(cands, inc_ev, S_star, delta, cfg, counts,
                                         guard_fn=d.guards)
        (rdir / "decisions.json").write_text(json.dumps(
            [x.to_json() for x in decisions], indent=1))
        self.history.replace_round(t)
        for c, dec in zip(cands, decisions):
            if c.ev is None:
                self.history.append_candidate(t, c.variant, c.edits, c.gate_failure or "not_evaluated",
                                              None, None, False, None, None, c.diff_path, c.detail)
                continue
            outcome = ("ACCEPTED" if c is winner else ("LOST" if dec.admissible else "REJECTED"))
            self.history.append_candidate(t, c.variant, c.edits, outcome, dec.delta_S, dec.delta_C,
                                          c is winner, dec.S, dec.C, c.diff_path,
                                          f"[re-adjudicated delta={delta:.5f}] " + dec.reason)
            log(d.name, f"r{t}{c.variant}: S={dec.S:.4f} dS={dec.delta_S:+.4f} dC={dec.delta_C:+.3f} "
                f"-> {outcome}: {dec.reason}")
        if winner is not None:
            r = G.git(self.repo, "merge-base", "--is-ancestor", inc_entry["commit"], winner.commit)
            if r.returncode != 0:
                raise SystemExit(f"{winner.commit} does not descend from H_{t} {inc_entry['commit']}")
            G.git(self.repo, "update-ref", f"refs/heads/{self.branch}", winner.commit, check=True)
            new = {"t": t + 1, "commit": winner.commit, "harness_tree": self.harness_tree(winner.commit),
                   "job": winner.ev.job, "S": winner.ev.S, "C": winner.ev.C,
                   "extra": winner.ev.extra, "variant": winner.variant}
            fr["incumbent"] = new
            fr["S_star"] = max(S_star, winner.ev.S)
            entry = {"t": t + 1, "S": winner.ev.S, "C": winner.ev.C, "commit": winner.commit,
                     "job": winner.ev.job}
            log(d.name, f"re-adjudicated r{t}: ACCEPTED {winner.variant} -> {winner.commit} "
                f"S={winner.ev.S:.4f} S*={fr['S_star']:.4f}")
        else:
            G.git(self.repo, "update-ref", f"refs/heads/{self.branch}", inc_entry["commit"], check=True)
            fr["incumbent"] = {"t": t + 1, "commit": inc_entry["commit"],
                               "harness_tree": self.harness_tree(inc_entry["commit"]),
                               "job": inc_job, "S": inc_ev.S, "C": inc_ev.C, "extra": inc_ev.extra}
            fr["S_star"] = S_star
            entry = {"t": t + 1, "S": inc_ev.S, "C": inc_ev.C, "commit": inc_entry["commit"],
                     "job": inc_job}
            log(d.name, f"re-adjudicated r{t}: no admissible candidate; H_{t+1} = H_{t}")
        fr["trajectory"] = fr["trajectory"][:t + 1] + [entry]
        self.save_frontier(fr)

    def reevaluate(self, t: int, variants: list[str] | None = None) -> None:
        """Re-measure the stored (committed) candidates of round t whose evaluation
        was invalid or corrupted by an infrastructure failure, then re-adjudicate
        the round. Rounds after t must have been removed first."""
        d = self.domain
        rdir = self.runs / f"r{t}"
        ids = d.evolve_ids()
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir() and len(p.name) == 1):
            if variants and vdir.name not in variants:
                continue
            prep = (json.loads((vdir / "prep.json").read_text())
                    if (vdir / "prep.json").exists() else {})
            if not prep.get("commit") or not G.branch_exists(self.repo, prep.get("branch", "")):
                log(d.name, f"r{t}{vdir.name}: no committed candidate to re-evaluate")
                continue
            (vdir / "eval.json").unlink(missing_ok=True)
            wt = self.wt_root / f"r{t}{vdir.name}"
            G.worktree_remove(self.repo, wt)
            G.git(self.repo, "worktree", "add", str(wt), prep["branch"], check=True)
            c = Candidate(vdir.name, prep.get("edits") or [], diff_path=prep.get("diff_path"),
                          branch=prep["branch"], commit=prep["commit"])
            self._evaluate(t, c, rdir, ids)
            log(d.name, f"r{t}{vdir.name}: re-evaluated -> "
                + (f"S={c.ev.S:.4f} C={c.ev.C} missing={c.ev.missing}" if c.ev else
                   f"{c.gate_failure}: {c.detail}"))
            G.worktree_remove(self.repo, wt)
        self.readjudicate(t)

    # ------------------------------------------------------------ helpers --
    def _draft(self, t, vid, rdir, report, digests, traces, inc_ev, budget, explore,
               reserved, prune, hist_rows, skill_md, patterns_md) -> Candidate:
        d, cfg = self.domain, self.cfg
        vdir = rdir / vid
        vdir.mkdir(exist_ok=True)
        branch = f"{d.name}/r{t}{vid}"
        wt = self.wt_root / f"r{t}{vid}"
        prep_path = vdir / "prep.json"
        # resume: candidate already drafted and committed in an earlier attempt
        if prep_path.exists():
            prep = json.loads(prep_path.read_text())
            if prep.get("gate_failure"):
                return Candidate(vid, prep.get("edits") or [], gate_failure=prep["gate_failure"],
                                 detail=prep.get("detail", ""),
                                 diff_path=prep.get("diff_path"))
            if prep.get("commit") and G.branch_exists(self.repo, branch):
                G.worktree_remove(self.repo, wt)
                G.git(self.repo, "worktree", "add", str(wt), branch, check=True)
                log(d.name, f"{vid}: resuming committed candidate {prep['commit']}")
                return Candidate(vid, prep["edits"], diff_path=prep["diff_path"],
                                 branch=branch, commit=prep["commit"])
        G.worktree_add(self.repo, wt, branch, self.branch)
        hdir = d.harness_dir(wt)
        variant_brief = (f"You are variant {vid} of round {t}. {cfg.m} variants are "
                         f"drafted independently from the same incumbent this round "
                         f"and each is evaluated on the full evolve set; the best "
                         f"admissible one becomes H_{t + 1}.")

        def finish(gate_failure: str, detail: str = "", edits=None, diff_path=None):
            prep_path.write_text(json.dumps(
                {"gate_failure": gate_failure, "detail": detail, "edits": edits or [],
                 "diff_path": diff_path}, ensure_ascii=False, indent=1))
            G.worktree_remove(self.repo, wt, branch)
            log(d.name, f"{vid}: {gate_failure} {detail[:200]}")
            return Candidate(vid, edits or [], gate_failure=gate_failure, detail=detail,
                             diff_path=diff_path)

        prop = propose(d, hdir, report, hist_rows, skill_md, patterns_md, budget,
                       explore, reserved, prune, traces=traces, per_task=inc_ev.per_task,
                       findings=digests, scoreboard=self.scoreboard(),
                       variant_brief=variant_brief, model=cfg.proposer_model)
        (vdir / "proposal.json").write_text(json.dumps(prop, ensure_ascii=False, indent=1))
        if prop["status"] != "done" or prop["n_edits"] == 0:
            return finish("no_proposal", str(prop.get("reason") or prop["status"]))

        # Critic(H_t, H') with bounded repair
        verdict, diff = None, ""
        for attempt in range(1 + cfg.repair_rounds):
            diff = G.diff_with_new_files(wt, self.harness_rel)
            (vdir / "diff.patch").write_text(diff)
            verdict = review(d, diff, prop.get("mechanism") or "",
                             prop.get("targets_mode") or "", edits=prop.get("edits"),
                             model=cfg.critic_model)
            if verdict.get("verdict") == "accept":
                # tag validation (l' must be evidenced by d'): a declared structural
                # component with nothing in the diff is re-tagged from the diff, and
                # a reserved exploration slot must still be honoured afterwards.
                tagged = [normalize(e.get("component"), diff, d.component_signals)
                          for e in (prop.get("edits") or [])]
                # A declared tag without diff evidence is simply re-tagged to what
                # the diff reads as (that is what the history records and what the
                # novelty term counts); it is bounced only when the re-tag breaks a
                # reserved exploration slot, since arguing labels with the proposer
                # costs a candidate and buys nothing.
                retagged = [(e.get("id"), str(e.get("component", "")).lower(), c)
                            for e, c in zip(prop.get("edits") or [], tagged)
                            if c != str(e.get("component", "")).lower()]
                for eid, decl, real in retagged:
                    log(d.name, f"{vid}: edit {eid} declared '{decl}' but the diff reads as "
                        f"'{real}'; re-tagged")
                if reserved and explore.get("untried") and not any(
                        c in explore["untried"] for c in tagged):
                    verdict = {"verdict": "reject",
                               "reasons": [f"this variant holds a RESERVED EXPLORATION SLOT: at "
                                           f"least one edit must be on a never-exercised component "
                                           f"from {explore['untried']}, judged by the DIFF, and "
                                           f"none is"],
                               "risk_notes": verdict.get("risk_notes")}
            (vdir / f"critic_a{attempt}.json").write_text(json.dumps(verdict, indent=1))
            if verdict.get("verdict") == "accept" or attempt >= cfg.repair_rounds:
                break
            log(d.name, f"{vid}: critic objections (repair {attempt + 1}/"
                f"{cfg.repair_rounds}): {str(verdict.get('reasons'))[:200]}")
            prop = propose(d, hdir, report, hist_rows, skill_md, patterns_md, budget,
                           explore, reserved, prune, traces=traces,
                           per_task=inc_ev.per_task, findings=digests,
                           scoreboard=self.scoreboard(), variant_brief=variant_brief,
                           repair_brief={"reasons": verdict.get("reasons"),
                                         "risk_notes": verdict.get("risk_notes"),
                                         "your_declared_edits": prop.get("edits")},
                           model=cfg.proposer_model)
            (vdir / f"proposal_r{attempt + 1}.json").write_text(
                json.dumps(prop, ensure_ascii=False, indent=1))
            if prop["status"] != "done":
                break
        (vdir / "critic.json").write_text(json.dumps(verdict, indent=1))
        edits = prop.get("edits") or []
        if not verdict or verdict.get("verdict") != "accept":
            return finish("critic_reject", str((verdict or {}).get("reasons"))[:600],
                          edits, str(vdir / "diff.patch"))

        # tag (l', h', d'): validate the declared component against the diff
        diff = G.diff_with_new_files(wt, self.harness_rel)
        (vdir / "diff.patch").write_text(diff)
        for e in edits:
            e["component"] = normalize(e.get("component"), diff, d.component_signals)
        commit = G.commit_path(wt, self.harness_rel,
                               f"r{t}{vid}: {prop.get('mechanism')}")

        # liveness smoke: does the candidate run at all (not a selection rule)
        ok, detail = d.smoke(wt, self.runs, f"r{t}{vid}_smoke",
                             d.smoke_ids(inc_ev.per_task))
        (vdir / "smoke.json").write_text(json.dumps({"ok": ok, **detail}, indent=1))
        if not ok:
            return finish("smoke_fail", json.dumps(detail)[:600], edits,
                          str(vdir / "diff.patch"))
        prep_path.write_text(json.dumps(
            {"commit": commit, "branch": branch, "edits": edits,
             "diff_path": str(vdir / "diff.patch"), "mechanism": prop.get("mechanism")},
            ensure_ascii=False, indent=1))
        log(d.name, f"{vid}: {len(edits)} edit(s) on "
            f"{[e['component'] for e in edits]} -> {commit}")
        return Candidate(vid, edits, diff_path=str(vdir / "diff.patch"),
                         branch=branch, commit=commit)

    def _evaluate(self, t: int, c: Candidate, rdir: Path, ids: list[str]) -> None:
        d, cfg = self.domain, self.cfg
        job = f"r{t}{c.variant}"
        ep = rdir / c.variant / "eval.json"
        if ep.exists():
            c.ev = EvalResult.load(ep)
            log(d.name, f"{c.variant}: reusing eval.json")
            return
        wt = self.wt_root / job
        ev = None
        for attempt in range(2):
            try:
                ev = evaluate(d, wt, self.runs, job, ids, cfg.k, log_prefix=job)
            except Exception as e:  # noqa: BLE001
                c.gate_failure, c.detail = "eval_invalid", f"evaluation crashed: {e!r}"[:600]
                log(d.name, f"{c.variant}: {c.detail}")
                return
            if ev.missing <= cfg.invalid_missing_frac * ev.n_expected:
                break
            log(d.name, f"{c.variant}: {ev.missing}/{ev.n_expected} trials missing "
                f"(infrastructure){'; retrying once' if attempt == 0 else ''}")
        if ev.missing > cfg.invalid_missing_frac * ev.n_expected:
            c.gate_failure = "eval_invalid"
            c.detail = f"{ev.missing}/{ev.n_expected} trials missing (infrastructure) after a retry"
            return
        ev.save(ep)
        ev.save(self.eval_path(job))
        c.ev = ev
