"""Deterministic curriculum validation. A curriculum that fails here never reaches lesson generation.

Checked: the goal belongs to the learner; every objective belongs to the goal and references a real knowledge-base
concept; every goal target is covered; prerequisites are exactly the knowledge base's (never model text), exist in
the curriculum and are acyclic; target mastery is valid; objectives are ordered prerequisites first; versions of a
curriculum are consistent (1..n, each the child of the previous one, deterministic ids and hashes).

A model's proposal is reviewed here too: unknown concepts are rejected as unresolved, known concepts outside the
goal's scope are rejected, and a suggested order that breaks a prerequisite is not used.
"""

from __future__ import annotations

from app.pedagogy.graph import ConceptGraph
from app.schemas.curriculum import (
    CurriculumConfig,
    CurriculumObjective,
    CurriculumProposal,
    CurriculumVersion,
    CurriculumWarning,
    ProposalReview,
    content_hash,
    curriculum_consistency,
    version_id_for,
)
from app.schemas.learner import LearningGoal


class CurriculumValidationError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid curriculum: " + "; ".join(problems))
        self.problems = problems


def find_cycle(edges: dict[str, list[str]]) -> list[str] | None:
    """A prerequisite cycle among `edges` (node -> prerequisites), as a path, or None."""
    state: dict[str, int] = {}

    def visit(node: str, path: tuple[str, ...]) -> list[str] | None:
        if state.get(node) == 2:
            return None
        if state.get(node) == 1:
            return [*path[path.index(node):], node]
        state[node] = 1
        for pre in sorted(edges.get(node, [])):
            found = visit(pre, (*path, node))
            if found:
                return found
        state[node] = 2
        return None

    for node in sorted(edges):
        found = visit(node, ())
        if found:
            return found
    return None


def curriculum_problems(*, learner_id: str, goal: LearningGoal, objectives: list[CurriculumObjective],
                        graph: ConceptGraph, config: CurriculumConfig) -> list[str]:
    problems: list[str] = []
    if goal.learner_id != learner_id:
        problems.append(f"goal {goal.goal_id} does not belong to the learner")
    if not objectives:
        return [*problems, "a curriculum needs at least one objective"]
    if len(objectives) > config.max_objectives:
        problems.append(f"{len(objectives)} objectives exceed the limit of {config.max_objectives}")
    unknown = [o.concept_id for o in objectives if o.concept_id not in graph]
    if unknown:
        problems.append(f"objectives reference concepts the knowledge base does not know: {unknown}")
    cycle = find_cycle({o.concept_id: o.prerequisites for o in objectives})
    if cycle:
        problems.append(f"circular prerequisites: {' -> '.join(cycle)}")
    for o in objectives:
        if not config.prerequisite_threshold <= o.target_mastery <= 1:
            problems.append(f"{o.concept_id}: target mastery {o.target_mastery} must be between the prerequisite "
                            f"threshold {config.prerequisite_threshold} and 1")
        if o.concept_id in graph and sorted(o.prerequisites) != sorted(graph.prerequisites(o.concept_id)):
            problems.append(f"{o.concept_id}: prerequisites {sorted(o.prerequisites)} are not the knowledge base's "
                            f"{sorted(graph.prerequisites(o.concept_id))}")
    missing_targets = sorted(set(goal.target_concepts) - {o.concept_id for o in objectives})
    if missing_targets:
        problems.append(f"goal targets without an objective: {missing_targets}")
    problems += curriculum_consistency(goal.goal_id, objectives)
    return problems


def validate_curriculum(*, learner_id: str, goal: LearningGoal, objectives: list[CurriculumObjective],
                        graph: ConceptGraph, config: CurriculumConfig) -> None:
    problems = curriculum_problems(learner_id=learner_id, goal=goal, objectives=objectives, graph=graph,
                                   config=config)
    if problems:
        raise CurriculumValidationError(problems)


def version_problems(versions: list[CurriculumVersion]) -> list[str]:
    """A curriculum's version history: 1..n, one goal and curriculum, each version the child of the previous one,
    deterministic ids and content hashes, and no version repeating its predecessor's content."""
    problems: list[str] = []
    for i, v in enumerate(versions, start=1):
        if v.version != i:
            problems.append(f"version {v.version} is out of sequence (expected {i})")
        if v.parent_version != (i - 1 if i > 1 else None):
            problems.append(f"version {v.version} has parent {v.parent_version}")
        if v.version_id != version_id_for(v.curriculum_id, v.version, v.content_hash):
            problems.append(f"version {v.version} does not have its deterministic id")
        if v.content_hash != content_hash(v.goal, v.objectives, v.config_fingerprint):
            problems.append(f"version {v.version}'s content hash does not match its content")
        if i > 1 and v.content_hash == versions[i - 2].content_hash:
            problems.append(f"version {v.version} repeats version {v.version - 1}")
    if len({(v.curriculum_id, v.goal_id, v.learner_id) for v in versions}) > 1:
        problems.append("versions belong to different curricula, goals or learners")
    return problems


def review_proposal(proposal: CurriculumProposal | None, objectives: list[CurriculumObjective],
                    known: set[str]) -> tuple[dict[str, str], ProposalReview, list[CurriculumWarning]]:
    """Which of the model's wording may be used. Returns (concept id -> description, review, warnings)."""
    scope = {o.concept_id for o in objectives}
    if proposal is None:
        return {}, ProposalReview(missing=[o.concept_id for o in objectives]), []
    descriptions: dict[str, str] = {}
    review = ProposalReview()
    warnings: list[CurriculumWarning] = []
    for p in proposal.objectives:
        if p.concept_id not in known:
            review.unresolved.append(p.concept_id)
            warnings.append(CurriculumWarning(code="unresolved_concept", details={"concept_id": p.concept_id},
                                              message=f"proposed concept {p.concept_id!r} is not in the knowledge "
                                                      "base; rejected"))
        elif p.concept_id not in scope:
            review.rejected[p.concept_id] = "outside the goal's scope (not a target nor a prerequisite of one)"
            warnings.append(CurriculumWarning(code="rejected_concept", details={"concept_id": p.concept_id},
                                              message=f"proposed concept {p.concept_id} is outside the goal's scope; "
                                                      "rejected"))
        elif p.concept_id in descriptions:
            review.rejected[p.concept_id] = "proposed twice; the first wording is used"
        else:
            descriptions[p.concept_id] = p.description.strip()
            review.accepted.append(p.concept_id)
    review.missing = [o.concept_id for o in objectives if o.concept_id not in descriptions]
    if review.missing:
        warnings.append(CurriculumWarning(code="description_missing", details={"concepts": review.missing},
                                          message="objectives without proposed wording use the template wording"))
    order = [c for c in proposal.suggested_order if c in scope]
    position = {c: i for i, c in enumerate(order)}
    for o in objectives:
        for pre in o.prerequisites:
            if o.concept_id in position and pre in position and position[pre] > position[o.concept_id]:
                review.ordering_issues.append(f"{o.concept_id} is suggested before its prerequisite {pre}")
    if review.ordering_issues:
        review.ordering_accepted = False
        warnings.append(CurriculumWarning(code="ordering_rejected", details={"issues": review.ordering_issues},
                                          message="the suggested order breaks prerequisites; the deterministic "
                                                  "prerequisite order is used"))
    return descriptions, review, warnings
