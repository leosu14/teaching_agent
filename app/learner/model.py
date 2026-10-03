"""Builds the LearnerModel: a deterministic view of mastery, evidence, history and goals in one domain."""

from __future__ import annotations

from datetime import datetime

from app.schemas.learner import LearnerProfile, LearningEvent, LearningEvidence, LearningGoal
from app.schemas.pedagogy import EvaluationSummary, LearnerError, LearnerModel, PedagogyConfig


def build_learner_model(
    profile: LearnerProfile,
    *,
    domain: str,
    framework_id: str,
    target_level: str | None,
    evidence: list[LearningEvidence],
    events: list[LearningEvent],
    goals: list[LearningGoal],
    universe: list[str],
    config: PedagogyConfig,
    now: datetime,
) -> LearnerModel:
    """`universe` is every concept the domain's knowledge base knows; a concept without evidence is unknown.

    Categories come from the configured bands: below `guided` weak, up to the mastery target developing, at or above
    it mastered. They are derived for presentation; the numeric `concepts` state is the canonical one."""
    state = profile.subjects.get(domain)
    concepts = sorted((c for c in profile.concepts.values() if c.subject == domain), key=lambda c: c.concept_id)
    assessed = {c.concept_id: c for c in concepts if c.evidence_count}
    mastered, developing, weak = [], [], []
    for cid, c in assessed.items():
        if c.mastery >= config.mastery_target:
            mastered.append(cid)
        elif c.mastery >= config.bands.guided:
            developing.append(cid)
        else:
            weak.append(cid)
    unknown = sorted((set(universe) | {c.concept_id for c in concepts}) - set(assessed))
    domain_evidence = [e for e in evidence if e.concept_id in {*universe, *(c.concept_id for c in concepts)}]
    errors = [e for e in domain_evidence if e.correctness == "incorrect"][-config.error_window:]
    evaluations = [a for a in profile.assessments if a.subject == domain][-5:]
    history = [e for e in events if e.subject in (None, domain)]
    history = sorted(history, key=lambda e: (e.at, e.event_id))[-config.history_limit:]
    return LearnerModel(
        learner_id=profile.learner_id,
        domain=domain,
        framework_id=framework_id,
        level=state.estimated_level if state else None,
        target_level=target_level or (state.target_level if state else None),
        goals=sorted((g for g in goals if g.domain == domain), key=lambda g: (g.priority, g.goal_id)),
        concepts=concepts,
        mastered_concepts=sorted(mastered),
        developing_concepts=sorted(developing),
        weak_concepts=sorted(weak),
        unknown_concepts=unknown,
        due_for_review=sorted(c.concept_id for c in concepts if c.next_review_at and c.next_review_at <= now),
        recent_errors=[LearnerError(concept_id=e.concept_id, evidence_id=e.evidence_id, source_type=e.source_type,
                                    at=e.timestamp) for e in errors],
        recent_evaluations=[EvaluationSummary(task_id=a.task_id, kind=a.kind, at=a.at, questions=len(a.evaluations),
                                              correct=sum(ev.correct for ev in a.evaluations),
                                              concept_ids=sorted({ev.concept_id for ev in a.evaluations}))
                            for a in evaluations],
        learning_history=history,
        preferences=profile.preferences,
        evidence_count=len(domain_evidence),
        as_of=now,
        updated_at=profile.updated_at,
    )
