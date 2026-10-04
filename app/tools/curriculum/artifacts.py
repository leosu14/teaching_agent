"""The artifacts of a curriculum version: LEARNING_GOAL -> CURRICULUM_VERSION -> LEARNING_OBJECTIVE (one per
objective) and CURRICULUM (the version's summary). A new version's CURRICULUM_VERSION also derives from the previous
version's, so the history is a chain. Artifact ids come back in `CurriculumVersion.artifact_ids`: "version" and one
per objective id; lessons derive from the objective artifact they serve."""

from __future__ import annotations

import json

from app.schemas.artifact import ArtifactDraft, ArtifactType, StoredArtifacts
from app.schemas.curriculum import CurriculumPlan
from app.schemas.learner import LearningGoal

VERSION_KEY = "version"


def curriculum_drafts(plan: CurriculumPlan, goal: LearningGoal, previous_id: str | None) -> list[ArtifactDraft]:
    """`previous_id`: the previous version's CURRICULUM_VERSION artifact (None for a first version)."""
    meta = {"goal_id": plan.goal_id, "curriculum_id": plan.curriculum_id, "version": plan.version,
            "version_id": plan.version_id}
    drafts = [
        ArtifactDraft(key="learning_goal", name="learning_goal", type=ArtifactType.LEARNING_GOAL,
                      media_type="application/json", content=goal.model_dump_json(indent=2, exclude={"updated_at"}),
                      metadata={"goal_id": goal.goal_id, "domain": goal.domain, "target_level": goal.target_level,
                                "status": goal.status.value}),
        ArtifactDraft(key=VERSION_KEY, name="curriculum_version", type=ArtifactType.CURRICULUM_VERSION,
                      media_type="application/json", content=plan.model_dump_json(indent=2),
                      parent_keys=["learning_goal"], parent_ids=[previous_id] if previous_id else [],
                      metadata=meta | {"base_version": plan.base_version, "reasons": [r.value for r in plan.reasons],
                                       "content_hash": plan.content_hash,
                                       "warnings": [w.code for w in plan.warnings]}),
    ]
    for o in plan.objectives:
        drafts.append(ArtifactDraft(
            key=o.objective_id, name=f"learning_objective_{o.concept_id}", type=ArtifactType.LEARNING_OBJECTIVE,
            media_type="application/json", content=o.model_dump_json(indent=2, exclude={"current_mastery", "status"}),
            parent_keys=[VERSION_KEY],
            metadata=meta | {"objective_id": o.objective_id, "concept_id": o.concept_id, "order": o.order,
                             "role": o.role, "mode": o.mode}))
    drafts.append(ArtifactDraft(
        key="curriculum", name="curriculum", type=ArtifactType.CURRICULUM, media_type="application/json",
        content=json.dumps({"curriculum_id": plan.curriculum_id, "goal_id": plan.goal_id, "title": plan.title,
                            "version": plan.version, "version_id": plan.version_id,
                            "objectives": [o.objective_id for o in plan.objectives], "rationale": plan.rationale},
                           indent=2),
        parent_keys=[VERSION_KEY], metadata=meta | {"objectives": len(plan.objectives)}))
    return drafts


def artifact_ids(plan: CurriculumPlan, stored: StoredArtifacts) -> dict[str, str]:
    """The ids a version records: its CURRICULUM_VERSION artifact and each objective's artifact."""
    return {VERSION_KEY: stored.by_key[VERSION_KEY],
            **{o.objective_id: stored.by_key[o.objective_id] for o in plan.objectives}}
