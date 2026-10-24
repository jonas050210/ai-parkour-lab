"""Harness²-style *restricted* research assistant.

The assistant is an optional convenience layer around the experiments.  It can look at a
finished or in-progress run, describe what looks wrong, and propose a small number of
bounded configuration changes.  It is intentionally weak in exactly the ways that matter:

* it cannot simulate physics, drive the humanoid, edit a policy, promote a checkpoint or
  change level geometry;
* every change it proposes must name a knob from :data:`KNOB_BOUNDS` and is clamped into
  that knob's range before it is returned;
* it never applies a change itself --- proposals are data for a human to inspect and pass
  back to ``parkour train``;
* network access is off unless explicitly enabled, and even then the response is treated
  as untrusted text and validated against the same bounds.

The default ``plan`` path is a transparent rule set over the run's own metrics.  It exists
so that a researcher gets a second opinion that is reproducible, not so that a model can
quietly take over the experiment.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import EvalConfig, env_config_from_checkpoint, evaluate_policy
from .levels import LevelCatalog
from .policies import ActorCritic
from .storage import DEFAULT_RUNS_DIR, RunStore

#: The only settings the assistant may touch.  Everything else is out of scope by design.
KNOB_BOUNDS: dict[str, tuple[float, float]] = {
    "reward.progress_per_metre": (1.0, 12.0),
    "reward.alive_per_second": (0.0, 2.5),
    "reward.upright_per_second": (0.0, 2.5),
    "reward.action_rate_penalty": (1e-4, 0.2),
    "reward.fall_penalty": (1.0, 25.0),
    "ppo.learning_rate": (1e-4, 1.5e-3),
    "ppo.entropy_coefficient": (1e-5, 2e-2),
    "ppo.gamma": (0.95, 0.999),
    "action_scale": (0.08, 0.6),
    "curriculum.weakness_scale": (0.5, 8.0),
    "curriculum.novelty_bonus": (0.0, 8.0),
    "curriculum.mastered_success_threshold": (0.4, 0.95),
}

FORBIDDEN_PREFIXES = (
    "humanoid",
    "level",
    "levels",
    "course",
    "physics",
    "observation",
    "action_mode",
    "actuation",
    "policy",
    "network",
    "checkpoint",
)


class ProposalError(ValueError):
    """Raised when a proposal tries to leave the allowed edit surface."""


@dataclass(frozen=True)
class AssistantConfig:
    min_improvement: float = 0.02
    max_proposals: int = 3
    allow_network: bool = False
    endpoint: str | None = None
    timeout_seconds: float = 20.0
    probe_levels: tuple[str, ...] = ()
    probe_seeds: tuple[int, ...] = (11,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_improvement": self.min_improvement,
            "max_proposals": self.max_proposals,
            "allow_network": self.allow_network,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout_seconds,
            "probe_levels": list(self.probe_levels),
            "probe_seeds": list(self.probe_seeds),
        }


@dataclass(frozen=True)
class Proposal:
    """One bounded suggestion. It is never applied automatically."""

    title: str
    rationale: str
    changes: dict[str, float]
    expected_gain: float
    risk: str = "low"
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"
    proposal_id: str = field(default_factory=lambda: f"proposal-{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "rationale": self.rationale,
            "changes": self.changes,
            "expected_gain": self.expected_gain,
            "risk": self.risk,
            "evidence": self.evidence,
            "source": self.source,
            "applied": False,
            "how_to_apply": "inspect, then pass these values to `parkour train` explicitly",
        }


def validate_changes(changes: Mapping[str, Any]) -> dict[str, float]:
    """Validate and clamp a proposed change set; reject anything off the edit surface."""

    if not isinstance(changes, Mapping):
        raise ProposalError("changes must be a mapping")
    result: dict[str, float] = {}
    for name, value in changes.items():
        if not isinstance(name, str):
            raise ProposalError("knob names must be strings")
        lowered = name.lower()
        if any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise ProposalError(f"{name!r} is outside the assistant's edit surface")
        if name not in KNOB_BOUNDS:
            raise ProposalError(f"unknown knob {name!r}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ProposalError(f"{name!r} must be numeric") from exc
        if not math.isfinite(number):
            raise ProposalError(f"{name!r} must be finite")
        low, high = KNOB_BOUNDS[name]
        result[name] = float(min(max(number, low), high))
    if not result:
        raise ProposalError("empty change set")
    return result


@dataclass(frozen=True)
class Report:
    run_id: str
    checkpoint: str
    levels: list[dict[str, Any]]
    aggregate: dict[str, Any]
    weak_levels: list[str]
    notes: list[str]
    report_path: str
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint": self.checkpoint,
            "levels": self.levels,
            "aggregate": self.aggregate,
            "weak_levels": self.weak_levels,
            "notes": self.notes,
            "report_path": self.report_path,
            "ok": self.ok,
            "restrictions": {
                "physics": False,
                "control": False,
                "weight_edits": False,
                "checkpoint_promotion": False,
            },
        }


def _latest_evaluation(store: RunStore) -> dict[str, Any] | None:
    evaluations = store.evaluations()
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda record: int(record.get("total_steps", 0)),
    )


def _fall_rate(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    falls = sum(
        1
        for record in records
        if record.get("termination_reason") in ("fell_over", "collapsed", "void_fall")
    )
    return falls / len(records)


class ResearchAssistant:
    """Restricted planner/evaluator. See the module docstring for the contract."""

    def __init__(
        self, catalog: LevelCatalog, config: AssistantConfig | None = None
    ) -> None:
        self.catalog = catalog
        self.config = config or AssistantConfig()

    # -- planning -----------------------------------------------------------------------
    def plan(
        self,
        *,
        run_dir: Path | str | None = None,
        target_levels: Sequence[str] | None = None,
    ) -> list[Proposal]:
        if target_levels:
            for level_id in target_levels:
                self.catalog[level_id]
        proposals: list[Proposal] = []
        if run_dir is None:
            proposals.append(
                Proposal(
                    title="Start with one level, then widen",
                    rationale=(
                        "A single-level run gives a clean learning signal and a stable "
                        "baseline before the adaptive sampler is introduced."
                    ),
                    changes={"curriculum.weakness_scale": 3.0},
                    expected_gain=0.0,
                    evidence={"source": "no run supplied"},
                )
            )
            proposals.append(
                Proposal(
                    title="Record deterministic evaluations every 20 iterations",
                    rationale=(
                        "Evaluation is the only signal used to compare runs; without "
                        "periodic deterministic rollouts a run cannot be ranked."
                    ),
                    changes={"curriculum.mastered_success_threshold": 0.7},
                    expected_gain=0.0,
                    evidence={"source": "no run supplied"},
                )
            )
            return proposals[: self.config.max_proposals]

        store = RunStore.load(Path(run_dir))
        config = store.read_config()
        summary = store.read_summary()
        evaluation = _latest_evaluation(store)
        episodes = list(store.episodes())[-400:]
        aggregate = (evaluation or {}).get("aggregate", {}).get("overall", {})
        success = float(aggregate.get("success_rate", 0.0))
        completion = float(aggregate.get("mean_completion", 0.0))
        falls = _fall_rate(episodes)
        training = store.training_rows()
        last = training[-1] if training else {}
        approx_kl = float(last.get("approx_kl", 0.0) or 0.0)
        clip_fraction = float(last.get("clip_fraction", 0.0) or 0.0)
        recent_success = [
            float(row.get("eval_success_rate") or 0.0)
            for row in training
            if row.get("eval_success_rate") not in (None, "")
        ][-5:]
        trend = (
            (recent_success[-1] - recent_success[0]) if len(recent_success) > 1 else 0.0
        )
        reward = dict(config.get("reward", {}) or {})
        ppo = dict(config.get("ppo", {}) or {})
        evidence = {
            "run_id": store.paths.run_id,
            "eval_success": success,
            "eval_completion": completion,
            "fall_rate": falls,
            "episodes_inspected": len(episodes),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "success_trend_over_last_evals": trend,
            "total_steps": summary.get("total_steps", 0),
        }

        # 1) Instability: the update is stepping outside the trust region.
        if config.get("ppo") and (approx_kl > 0.03 or clip_fraction > 0.35):
            current = float(ppo.get("learning_rate", 3e-4))
            proposals.append(
                Proposal(
                    title="Reduce the PPO learning rate",
                    rationale=(
                        f"approx_kl={approx_kl:.3f} and clip_fraction={clip_fraction:.2f} "
                        "suggest updates are leaving the trust region, which makes "
                        "learning unstable before it makes it slow."
                    ),
                    changes={"ppo.learning_rate": current * 0.5},
                    expected_gain=0.03,
                    risk="low",
                    evidence=evidence,
                )
            )

        # 2) Falls dominate: balance shaping is under-weighted.
        if falls > 0.55 and completion < 0.6:
            upright = float(reward.get("upright_per_second", 0.5))
            proposals.append(
                Proposal(
                    title="Strengthen upright shaping",
                    rationale=(
                        f"{falls * 100:.0f}% of recent episodes ended in a fall at only "
                        f"{completion * 100:.0f}% completion. Raising the upright term "
                        "rewards staying on the feet while the policy learns a gait."
                    ),
                    changes={
                        "reward.upright_per_second": upright * 1.6,
                        "reward.alive_per_second": float(
                            reward.get("alive_per_second", 1.0)
                        )
                        * 1.2,
                    },
                    expected_gain=0.04,
                    risk="medium",
                    evidence=evidence,
                )
            )

        # 3) Stagnation with little exploration: widen the search.
        if len(recent_success) >= 3 and abs(trend) < self.config.min_improvement:
            entropy = float(ppo.get("entropy_coefficient", 0.003))
            proposals.append(
                Proposal(
                    title="Increase exploration slightly",
                    rationale=(
                        "Evaluation success has been flat across the last evaluations. "
                        "A small entropy increase gives the policy a chance to leave the "
                        "current behaviour without discarding it."
                    ),
                    changes={"ppo.entropy_coefficient": entropy * 1.8},
                    expected_gain=0.02,
                    risk="low",
                    evidence=evidence,
                )
            )

        # 4) Multi-level run with weak levels: focus the sampler.
        levels = [
            row
            for row in (evaluation or {}).get("records", [])
            if row.get("level_id")
        ]
        by_level: dict[str, list[Mapping[str, Any]]] = {}
        for record in levels:
            by_level.setdefault(str(record["level_id"]), []).append(record)
        weak = [
            level_id
            for level_id, records in by_level.items()
            if sum(1 for record in records if record.get("success")) / len(records) < 0.5
        ]
        if len(by_level) > 1 and weak:
            curriculum = dict(config.get("curriculum_config", {}) or {})
            proposals.append(
                Proposal(
                    title="Focus the adaptive curriculum on weak levels",
                    rationale=(
                        f"Levels below 50% success in the latest evaluation: "
                        f"{', '.join(sorted(weak))}. Raising the weakness exponent "
                        "samples them more often while keeping mastered levels in the mix."
                    ),
                    changes={
                        "curriculum.weakness_scale": float(
                            curriculum.get("weakness_scale", 3.0)
                        )
                        * 1.5,
                        "curriculum.mastered_success_threshold": float(
                            curriculum.get("mastered_success_threshold", 0.75)
                        ),
                    },
                    expected_gain=0.03,
                    risk="low",
                    evidence={**evidence, "weak_levels": sorted(weak)},
                )
            )

        # 5) Nothing is progressing at all: the action range may be too coarse/fine.
        if completion < 0.15 and success == 0.0 and len(training) > 20:
            proposals.append(
                Proposal(
                    title="Reduce the action range",
                    rationale=(
                        "The policy reaches under 15% of the course and never succeeds; a "
                        "smaller action range keeps the first learned behaviours close to "
                        "the stable standing pose."
                    ),
                    changes={"action_scale": max(0.1, float(config.get("action_scale", 0.25)) * 0.7)},
                    expected_gain=0.02,
                    risk="medium",
                    evidence=evidence,
                )
            )

        if self.config.allow_network and self.config.endpoint:
            try:
                proposals.extend(self._network_proposals(evidence))
            except Exception as exc:  # network is optional; never break planning
                proposals.append(
                    Proposal(
                        title="Remote planner unavailable",
                        rationale=f"Falling back to the local rule set ({type(exc).__name__}).",
                        changes={"curriculum.weakness_scale": 3.0},
                        expected_gain=0.0,
                        evidence={"error": str(exc)},
                    )
                )

        # Validate everything that leaves the assistant.
        cleaned: list[Proposal] = []
        for proposal in proposals:
            changes = validate_changes(proposal.changes)
            cleaned.append(
                Proposal(
                    title=proposal.title,
                    rationale=proposal.rationale,
                    changes=changes,
                    expected_gain=proposal.expected_gain,
                    risk=proposal.risk,
                    evidence=proposal.evidence,
                    source=proposal.source,
                )
            )
        cleaned.sort(key=lambda proposal: proposal.expected_gain, reverse=True)
        return cleaned[: self.config.max_proposals]

    # -- probing ------------------------------------------------------------------------
    def probe(
        self,
        run_dir: Path | str,
        *,
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        checkpoint: Path | str | None = None,
    ) -> Report:
        """Evaluate a saved checkpoint across the catalogue and write a report."""

        store = RunStore.load(Path(run_dir))
        config = store.read_config()
        checkpoint_path = Path(checkpoint) if checkpoint else store.paths.best
        if not checkpoint_path.exists():
            checkpoint_path = store.paths.latest
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"run {store.paths.run_id!r} has no checkpoint to probe")
        policy, extra = ActorCritic.load(checkpoint_path)
        level_ids = list(self.config.probe_levels) or [
            level.id for level in self.catalog
        ]
        env_config = (
            env_config_from_checkpoint(extra, level_ids[0]) if extra.get("config") else None
        )
        evaluation = evaluate_policy(
            policy,
            self.catalog,
            level_ids,
            config=EvalConfig(seeds=self.config.probe_seeds, deterministic=True),
            env_config=env_config,
        )
        aggregate = evaluation["aggregate"]["overall"]
        by_level: dict[str, list[dict[str, Any]]] = {}
        for record in evaluation["records"]:
            by_level.setdefault(str(record["level_id"]), []).append(record)
        level_rows = []
        weak_levels = []
        for level_id, records in by_level.items():
            success = sum(1 for record in records if record.get("success")) / len(records)
            completion = sum(
                float(record.get("completion_fraction", 0.0)) for record in records
            ) / len(records)
            level_rows.append(
                {
                    "level_id": level_id,
                    "episodes": len(records),
                    "success_rate": success,
                    "mean_completion": completion,
                    "mean_return": sum(
                        float(record.get("episode_return", 0.0)) for record in records
                    )
                    / len(records),
                    "terminations": sorted(
                        {str(record.get("termination_reason")) for record in records}
                    ),
                }
            )
            if success < 0.5:
                weak_levels.append(level_id)
        notes = [
            "deterministic evaluation, assist-free",
            f"checkpoint: {checkpoint_path}",
            f"policy hidden size: {extra.get('hidden_dim', getattr(policy, 'hidden_dim', '?'))}",
        ]
        report_dir = store.paths.root / "harness2"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = report_dir / f"probe-{stamp}.json"
        payload = {
            "run_id": store.paths.run_id,
            "checkpoint": str(checkpoint_path),
            "config_action_scale": config.get("action_scale"),
            "aggregate": aggregate,
            "levels": level_rows,
            "weak_levels": weak_levels,
            "notes": notes,
            "restrictions": {
                "physics": "untouched",
                "controller": "untouched",
                "learning_algorithm": "untouched",
            },
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        del runs_dir
        return Report(
            run_id=store.paths.run_id,
            checkpoint=str(checkpoint_path),
            levels=level_rows,
            aggregate=aggregate,
            weak_levels=weak_levels,
            notes=notes,
            report_path=str(report_path),
        )

    # -- optional network planner -------------------------------------------------------
    def _network_proposals(self, evidence: Mapping[str, Any]) -> list[Proposal]:
        """Ask a configured endpoint for proposals and validate them strictly."""

        assert self.config.endpoint
        body = json.dumps(
            {
                "task": "propose bounded PPO/reward settings",
                "allowed_knobs": {name: list(bounds) for name, bounds in KNOB_BOUNDS.items()},
                "evidence": dict(evidence),
                "max_proposals": self.config.max_proposals,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        proposals = []
        for raw in payload.get("proposals", [])[: self.config.max_proposals]:
            proposals.append(
                Proposal(
                    title=str(raw.get("title", "remote proposal"))[:120],
                    rationale=str(raw.get("rationale", ""))[:800],
                    changes=validate_changes(raw.get("changes", {})),
                    expected_gain=float(raw.get("expected_gain", 0.0)),
                    risk=str(raw.get("risk", "medium"))[:20],
                    evidence={"source": "network", "knobs": list(raw.get("changes", {}))},
                    source="network",
                )
            )
        return proposals


__all__ = [
    "AssistantConfig",
    "KNOB_BOUNDS",
    "Proposal",
    "ProposalError",
    "Report",
    "ResearchAssistant",
    "validate_changes",
]
