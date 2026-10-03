"""KnowledgeGapAnalyzer: deterministic gap detection and prioritisation against a learning goal.

No model is involved. A model may later explain the gaps in words (`KnowledgeGapSet.explanation`), but which
concepts are gaps, how urgent they are and what to do about them is computed here.
"""

from __future__ import annotations

from app.pedagogy.graph import ConceptGraph
from app.schemas.learner import LearningGoal, stable_id
from app.schemas.pedagogy import (
    GapAction,
    KnowledgeGap,
    KnowledgeGapSet,
    LearnerModel,
    PedagogyConfig,
    PriorityFactors,
)


class GapAnalysisError(ValueError):
    pass


class KnowledgeGapAnalyzer:
    def __init__(self, config: PedagogyConfig | None = None) -> None:
        self.config = config or PedagogyConfig()

    def analyze(self, model: LearnerModel, graph: ConceptGraph, goal: LearningGoal) -> KnowledgeGapSet:
        if goal.learner_id != model.learner_id or goal.domain != model.domain:
            raise GapAnalysisError("the goal must belong to the learner and the model's domain")
        missing = [c for c in goal.target_concepts if c not in graph]
        if missing:
            raise GapAnalysisError(f"goal {goal.goal_id} targets concepts the knowledge base does not know: {missing}")
        cfg = self.config
        scope = graph.closure(goal.target_concepts)
        targets = set(goal.target_concepts)
        gaps: list[KnowledgeGap] = []
        mastered: list[str] = []
        for cid in scope:
            mastery = model.mastery_of(cid)
            if mastery >= cfg.mastery_target:
                mastered.append(cid)
                continue
            gaps.append(self._gap(model, graph, cid, mastery, scope, cid in targets))
        gaps.sort(key=lambda g: (-g.priority, scope.index(g.concept.concept_id)))
        due = [c for c in mastered if c in model.due_for_review]
        gap_set = KnowledgeGapSet(gap_set_id="pending", learner_id=model.learner_id, goal_id=goal.goal_id,
                                  domain=model.domain, gaps=gaps, mastered=mastered, due_for_review=due, scope=scope,
                                  config_fingerprint=cfg.fingerprint())
        body = gap_set.model_dump_json(exclude={"gap_set_id"})
        return gap_set.model_copy(update={"gap_set_id": stable_id("gaps", body)})

    def _gap(self, model: LearnerModel, graph: ConceptGraph, cid: str, mastery: float, scope: list[str],
             is_target: bool) -> KnowledgeGap:
        cfg = self.config
        concept = graph.concept(cid)
        state = model.state(cid)
        assessed = model.has_evidence(cid)
        band = cfg.bands.band_for(mastery)
        prerequisites = graph.prerequisites(cid)
        unmet = [p for p in prerequisites if model.mastery_of(p) < cfg.prerequisite_threshold]
        streak = state.incorrect_streak if state else 0
        if assessed and state is not None and state.last_assessed_at is not None:
            idle_days = max(0.0, (model.as_of - state.last_assessed_at).total_seconds() / 86400)
            recency = min(1.0, idle_days / cfg.recency_horizon_days)
        else:
            recency = 1.0
        factors = PriorityFactors(
            deficit=round(max(0.0, cfg.mastery_target - mastery) / cfg.mastery_target, 4),
            prerequisite_importance=graph.importance(cid, scope),
            goal_relevance=1.0 if is_target else 0.5,
            recent_errors=round(min(1.0, model.errors_on(cid) / cfg.error_saturation), 4),
            recency=round(recency, 4),
            repeated_failure=1.0 if streak >= cfg.repeated_failure_streak else 0.0,
        )
        weights = cfg.weights.model_dump()
        priority = round(sum(weights[k] * v for k, v in factors.model_dump().items()) / sum(weights.values()), 4)
        action = self._action(assessed, band, unmet, factors.repeated_failure > 0)
        return KnowledgeGap(concept=concept, mastery=mastery, confidence=state.confidence if state else 0.0,
                            band=band, assessed=assessed, priority=priority, factors=factors,
                            prerequisites=prerequisites, unmet_prerequisites=unmet,
                            reason=self._reason(concept.name, mastery, assessed, band, unmet, factors, is_target),
                            recommended_action=action)

    @staticmethod
    def _action(assessed: bool, band: str, unmet: list[str], repeated_failure: bool) -> GapAction:
        if unmet:
            return "prerequisite_first"
        if repeated_failure:
            return "reteach"
        if not assessed or band == "foundational":
            return "introduce"
        if band == "guided":
            return "reteach"
        return "reinforce"

    def _reason(self, name: str, mastery: float, assessed: bool, band: str, unmet: list[str],
                factors: PriorityFactors, is_target: bool) -> str:
        parts = [f"{name}: mastery {mastery:.2f} is below the target {self.config.mastery_target:.2f} ({band})"
                 if assessed else f"{name}: no evidence yet"]
        parts.append("a goal concept" if is_target else "a prerequisite of the goal")
        if unmet:
            parts.append(f"prerequisites not yet secure: {', '.join(unmet)}")
        if factors.recent_errors:
            parts.append("recent errors")
        if factors.repeated_failure:
            parts.append("repeated failures")
        if factors.prerequisite_importance:
            parts.append("other goal concepts depend on it")
        return "; ".join(parts) + "."
