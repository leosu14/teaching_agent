"""PedagogicalStrategy: how a domain turns "teach concept X in mode M at band B" into objectives and activities.

Subject-specific teaching (language learning, mathematics, exam preparation, vocabulary, grammar, ...) is a
strategy; the engine only knows the interface. `GenericStrategy` is the default for every domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.concepts import Concept
from app.schemas.pedagogy import (
    ConceptTreatment,
    DifficultyBand,
    LearningActivity,
    LearningObjective,
    PedagogyConfig,
    Phase,
)


@dataclass(frozen=True)
class PlannedActivity:
    phase: Phase
    activity: LearningActivity


class PedagogicalStrategy(ABC):
    strategy_id: str
    activity_types: tuple[str, ...]  # the activity types this strategy uses; not every type for every subject

    @abstractmethod
    def objective(self, concept: Concept, treatment: ConceptTreatment, config: PedagogyConfig) -> LearningObjective:
        ...

    @abstractmethod
    def activities(self, concept: Concept, treatment: ConceptTreatment, objective: LearningObjective | None,
                   config: PedagogyConfig) -> list[PlannedActivity]:
        """Instruction, practice and review activities for one concept (assessment is separate)."""

    @abstractmethod
    def assessment(self, objective: LearningObjective, band: DifficultyBand,
                   config: PedagogyConfig) -> PlannedActivity: ...

    def minutes(self, treatment: ConceptTreatment, config: PedagogyConfig) -> int:
        """Total time the concept needs in a lesson, assessment included."""
        objective = self.objective(_placeholder(treatment), treatment, config)
        total = sum(p.activity.estimated_minutes for p in self.activities(_placeholder(treatment), treatment,
                                                                            objective, config))
        if treatment.mode != "review":
            total += self.assessment(objective, treatment.band, config).activity.estimated_minutes
        return total


def _placeholder(treatment: ConceptTreatment) -> Concept:
    return Concept(concept_id=treatment.concept_id, name=treatment.name, domain="-")


class GenericStrategy(PedagogicalStrategy):
    """Explain, then practise at the learner's band, then check against the objective. Practice difficulty follows
    the band: recognition (multiple choice) when foundational, completion (fill in the blank) when guided, open
    production (free response) when independent or consolidating."""

    strategy_id = "generic"
    activity_types = ("explanation", "multiple_choice", "fill_blank", "free_response")
    PRACTICE: dict[str, str] = {"foundational": "multiple_choice", "guided": "fill_blank",
                                "independent": "free_response", "consolidation": "free_response"}
    RESPONSE: dict[str, str] = {"multiple_choice": "the correct option", "fill_blank": "the missing form or term",
                                "free_response": "an original answer that applies the concept",
                                "explanation": "attention; no response required"}

    def practice_type(self, band: DifficultyBand) -> str:
        return self.PRACTICE[band]

    def objective(self, concept: Concept, treatment: ConceptTreatment, config: PedagogyConfig) -> LearningObjective:
        if treatment.mode == "introduce":
            description = f"Understand {concept.name} and apply it with support"
            target = config.bands.independent
        elif treatment.mode == "reinforce":
            description = f"Apply {concept.name} accurately with less support"
            target = config.mastery_target
        else:
            description = f"Recall and apply {concept.name} without support"
            target = config.mastery_target
        return LearningObjective(objective_id=f"obj_{concept.concept_id}", concept_id=concept.concept_id,
                                 description=description, target_mastery=max(target, round(treatment.mastery, 4)),
                                 assessment_method=self.practice_type(treatment.band))

    def activities(self, concept: Concept, treatment: ConceptTreatment, objective: LearningObjective | None,
                   config: PedagogyConfig) -> list[PlannedActivity]:
        m = config.planner.minutes
        cid, band = concept.concept_id, treatment.band
        practice = self.practice_type(band)
        target = objective.objective_id if objective else None
        if treatment.mode == "review":
            phase: Phase = "prerequisite_review" if treatment.role == "prerequisite" else "spaced_review"
            return [PlannedActivity(phase, self._activity(f"act_{cid}_review", practice, cid, band, m.review,
                                                          f"Retrieve {concept.name} from memory, then check.",
                                                          None))]
        explain = (f"Explain {concept.name} step by step with worked examples." if treatment.mode == "introduce"
                   else f"Recap {concept.name}, addressing earlier errors.")
        return [
            PlannedActivity("instruction", self._activity(f"act_{cid}_explain", "explanation", cid, band,
                                                          m.explanation if treatment.mode == "introduce" else m.review,
                                                          explain, None)),
            PlannedActivity("practice", self._activity(f"act_{cid}_practice", practice, cid, band, m.practice,
                                                       f"Practise {concept.name} ({practice.replace('_', ' ')}).",
                                                       target)),
        ]

    def assessment(self, objective: LearningObjective, band: DifficultyBand,
                   config: PedagogyConfig) -> PlannedActivity:
        return PlannedActivity("assessment", self._activity(
            f"act_{objective.concept_id}_check", objective.assessment_method, objective.concept_id, band,
            config.planner.minutes.assessment, f"Check: {objective.description}.", objective.objective_id))

    def _activity(self, activity_id: str, kind: str, concept_id: str, band: DifficultyBand, minutes: int,
                  instructions: str, target: str | None) -> LearningActivity:
        return LearningActivity(activity_id=activity_id, type=kind, concept_ids=[concept_id], difficulty=band,
                                estimated_minutes=minutes, instructions=instructions,
                                expected_response=self.RESPONSE.get(kind, "a response that applies the concept"),
                                assessment_target=target)


class UnknownStrategy(KeyError):
    pass


class StrategyRegistry:
    def __init__(self, default: PedagogicalStrategy | None = None) -> None:
        self._default = default or GenericStrategy()
        self._strategies: dict[str, PedagogicalStrategy] = {self._default.strategy_id: self._default}

    def register(self, strategy: PedagogicalStrategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> PedagogicalStrategy:
        try:
            return self._strategies[strategy_id]
        except KeyError:
            raise UnknownStrategy(strategy_id) from None

    def for_domain(self, domain: str, config: PedagogyConfig) -> PedagogicalStrategy:
        """The strategy configured for the domain, otherwise the default."""
        chosen = config.strategies.get(domain)
        return self.get(chosen) if chosen else self._default

    def ids(self) -> list[str]:
        return sorted(self._strategies)
