"""Research failure policy. Applied by the workflow, never inside the ResearchAgent.

- mandatory: a failed research stops the task with an explicit error.
- optional: the lesson continues with an empty bundle that carries an explicit warning.
Sources are never invented in either case.
"""

from __future__ import annotations

from typing import Literal

from app.runtime.workflow.nodes import NodeFatal
from app.schemas.research import ResearchBundle

ResearchRequirement = Literal["mandatory", "optional"]


class ResearchRequired(NodeFatal):
    """Research failed and the workflow requires it."""


def apply_research_policy(bundle: ResearchBundle, requirement: ResearchRequirement) -> ResearchBundle:
    if bundle.status != "failed":
        return bundle
    reasons = "; ".join(e.message for e in bundle.errors) or "; ".join(bundle.warnings)
    if requirement == "mandatory":
        raise ResearchRequired(f"research is mandatory and failed: {reasons}")
    warning = f"Research is optional and failed, so this lesson has no cited sources: {reasons}"
    return bundle.model_copy(update={"warnings": [*bundle.warnings, warning]})
