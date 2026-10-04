"""CurriculumPlanner: a learning goal + the knowledge base's concept graph + the learner model -> a curriculum.

Deterministic. The scope is the goal's target concepts plus every prerequisite they rely on (from the knowledge
base, never from model text), ordered prerequisites first (the graph's stable topological order). Per objective:

- role: `target` for a goal concept, `prerequisite` for a concept the goal relies on;
- mode: `maintain` when the concept is already mastered (target mastery and enough evidence), otherwise `learn`;
- priority: `target_priority` / `prerequisite_priority`; `remediation_priority` after repeated failure, which also
  adds `remediation_evidence_bonus` to the evidence required and lists the prerequisites to revisit first;
- target mastery: the configured mastery target; evidence required: `evidence_required`.

A model may word the objectives afterwards (`finalize` takes a validated proposal); it never changes the structure.
Feasibility of a target date is estimated against an explicit `as_of` and reported as a warning, never "fixed".
"""

from __future__ import annotations

import math
from datetime import datetime

from app.curriculum.progress import objective_status
from app.curriculum.validation import CurriculumValidationError, review_proposal, validate_curriculum
from app.pedagogy.graph import ConceptGraph
from app.schemas.concepts import Concept
from app.schemas.curriculum import (
    BriefConcept,
    CurriculumBrief,
    CurriculumConfig,
    CurriculumDraft,
    CurriculumObjective,
    CurriculumPlan,
    CurriculumProposal,
    CurriculumVersion,
    CurriculumWarning,
    GoalSpec,
    ObjectiveStatus,
    ReplanReason,
    content_hash,
    curriculum_id_for,
    version_id_for,
)
from app.schemas.learner import LearningGoal
from app.schemas.pedagogy import LearnerModel


def targets_for_level(concepts: list[Concept], level: str, levels: list[str]) -> list[str]:
    """The concepts of a domain up to and including `level` of the level framework (`levels` in order). Concepts
    without a level, or with a level the framework does not have, are not targets."""
    if level not in levels:
        raise CurriculumValidationError([f"level {level!r} is not one of {levels}"])
    top = levels.index(level)
    return [c.concept_id for c in concepts if c.level in levels and levels.index(c.level) <= top]


class CurriculumPlanner:
    def __init__(self, config: CurriculumConfig | None = None) -> None:
        self.config = config or CurriculumConfig()

    # --- deterministic structure ------------------------------------------------------------------------------

    def draft(self, goal: LearningGoal, graph: ConceptGraph, model: LearnerModel, *, as_of: datetime,
              language: str = "en") -> CurriculumDraft:
        cfg = self.config
        if goal.learner_id != model.learner_id or goal.domain != model.domain:
            raise CurriculumValidationError([f"goal {goal.goal_id} does not belong to this learner and domain"])
        unknown = [c for c in goal.target_concepts if c not in graph]
        if unknown:
            raise CurriculumValidationError([f"goal targets concepts the knowledge base does not know: {unknown}"])
        scope = graph.closure(goal.target_concepts)
        if len(scope) > cfg.max_objectives:
            raise CurriculumValidationError([f"the goal needs {len(scope)} objectives; the limit is "
                                             f"{cfg.max_objectives}"])
        targets = set(goal.target_concepts)
        objectives = []
        for order, cid in enumerate(scope, start=1):
            concept = graph.concept(cid)
            objectives.append(self._objective(goal, concept, order, cid in targets, model))
        brief = CurriculumBrief(
            domain=goal.domain, goal_title=goal.title or goal.description or goal.domain,
            goal_description=goal.description, target_level=goal.target_level, language_of_instruction=language,
            concepts=[BriefConcept(concept_id=o.concept_id, name=o.name, level=graph.concept(o.concept_id).level,
                                   description=graph.concept(o.concept_id).description,
                                   prerequisites=o.prerequisites, role=o.role, state=self._band(o, model))
                      for o in objectives])
        return CurriculumDraft(curriculum_id=curriculum_id_for(goal.goal_id), learner_id=goal.learner_id, goal=goal,
                               objectives=objectives, brief=brief,
                               warnings=self.feasibility(goal, objectives, as_of),
                               config_fingerprint=cfg.fingerprint())

    def _objective(self, goal: LearningGoal, concept: Concept, order: int, is_target: bool,
                   model: LearnerModel) -> CurriculumObjective:
        cfg = self.config
        cid = concept.concept_id
        state = model.state(cid)
        prerequisites = list(concept.prerequisites)
        streak = state.incorrect_streak if state is not None else 0
        base = CurriculumObjective(
            objective_id=CurriculumObjective.id_for(goal.goal_id, cid), goal_id=goal.goal_id, concept_id=cid,
            name=concept.name, description=f"Use {concept.name} accurately and independently", order=order,
            role="target" if is_target else "prerequisite", target_mastery=cfg.mastery_target,
            priority=cfg.target_priority if is_target else cfg.prerequisite_priority,
            prerequisites=sorted(prerequisites), evidence_required=cfg.evidence_required,
            current_mastery=model.mastery_of(cid))
        status, _ = objective_status(base, model, cfg)
        update: dict = {"status": status}
        if status == ObjectiveStatus.MASTERED:
            update |= {"mode": "maintain", "description": f"Keep {concept.name} secure through spaced review"}
        elif streak >= cfg.repeated_failure_streak:
            update |= {"remediated": True, "priority": cfg.remediation_priority,
                       "evidence_required": cfg.evidence_required + cfg.remediation_evidence_bonus,
                       "remediation": sorted(prerequisites),
                       "description": f"Relearn {concept.name} step by step"
                                      + (", revisiting its prerequisites first" if prerequisites else "")}
        return base.model_copy(update=update)

    def _band(self, objective: CurriculumObjective, model: LearnerModel) -> str:
        if objective.mode == "maintain":
            return "mastered"
        return "developing" if model.has_evidence(objective.concept_id) else "new"

    def feasibility(self, goal: LearningGoal, objectives: list[CurriculumObjective],
                    as_of: datetime) -> list[CurriculumWarning]:
        """Whether the remaining objectives can reasonably fit before the target date. Reported, never adjusted."""
        if goal.target_date is None:
            return []
        f = self.config.feasibility
        sessions = sum(math.ceil(max(0.0, o.target_mastery - o.current_mastery) / f.mastery_gain_per_session)
                       for o in objectives if o.mode == "learn")
        needed_days = math.ceil(sessions / f.sessions_per_week * 7)
        available_days = (goal.target_date - as_of).total_seconds() / 86400
        details = {"sessions_needed": sessions, "days_needed": needed_days,
                   "days_available": round(available_days, 1), "sessions_per_week": f.sessions_per_week,
                   "target_date": goal.target_date.isoformat()}
        if available_days <= 0:
            return [CurriculumWarning(code="deadline_passed", details=details,
                                      message=f"the target date {goal.target_date.date()} has passed; "
                                              f"{sessions} sessions of work remain")]
        if needed_days > available_days:
            return [CurriculumWarning(
                code="deadline_infeasible", details=details,
                message=(f"about {sessions} sessions ({needed_days} days at {f.sessions_per_week:g} a week) remain "
                         f"but only {available_days:.0f} days are left before {goal.target_date.date()}; the "
                         "schedule is not compressed: move the date or narrow the goal"))]
        return []

    # --- validated plan ------------------------------------------------------------------------------------------

    def finalize(self, draft: CurriculumDraft, graph: ConceptGraph, *, proposal: CurriculumProposal | None = None,
                 carried: dict[str, str] | None = None, current: CurriculumVersion | None = None,
                 reasons: list[ReplanReason] | None = None) -> CurriculumPlan:
        """Apply the validated wording (a model's proposal, else wording carried over from the current version,
        else the template), validate everything, and decide whether this is a new version."""
        descriptions, review, warnings = review_proposal(proposal, draft.objectives, set(graph.ids))
        carried = carried or {}
        objectives = []
        for o in draft.objectives:
            text = descriptions.get(o.concept_id) or carried.get(o.objective_id)
            objectives.append(o.model_copy(update={"description": text[:400]}) if text else o)
        goal = draft.goal
        validate_curriculum(learner_id=draft.learner_id, goal=goal, objectives=objectives, graph=graph,
                            config=self.config)
        spec = GoalSpec.of(goal)
        digest = content_hash(spec, objectives, draft.config_fingerprint)
        if current is not None and current.content_hash == digest:
            version, version_id, changed = current.version, current.version_id, False
        else:
            version = current.version + 1 if current is not None else 1
            version_id, changed = version_id_for(draft.curriculum_id, version, digest), True
        why = reasons or ([ReplanReason.INITIAL] if current is None else [ReplanReason.REBUILD])
        return CurriculumPlan(
            curriculum_id=draft.curriculum_id, learner_id=draft.learner_id, goal_id=goal.goal_id, title=spec.title,
            goal=spec, objectives=objectives, content_hash=digest, version=version, version_id=version_id,
            base_version=current.version if current is not None else None, changed=changed, reasons=why,
            rationale=proposal.explanation if proposal is not None else (current.rationale if current else ""),
            proposal_review=review, warnings=[*draft.warnings, *warnings],
            config_fingerprint=draft.config_fingerprint)
