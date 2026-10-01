"""Visual failure policy. Applied by the workflow, never inside the VisualAgent.

- An optional visual that fails never stops the lesson: the failure is recorded and becomes a warning.
- A required visual that fails follows `VisualFailurePolicy`:
  - fail: the task fails with an explicit error naming the visual and why.
  - continue: the lesson completes without it and carries an explicit warning.
Nothing is ever substituted silently: no placeholder image and no invented attribution.
"""

from __future__ import annotations

from typing import Literal

from app.runtime.workflow.nodes import NodeFatal
from app.schemas.visual import VisualResult

VisualFailurePolicy = Literal["fail", "continue"]


class VisualRequired(NodeFatal):
    """A required visual could not be produced and the workflow requires it."""


def apply_visual_policy(result: VisualResult, policy: VisualFailurePolicy) -> VisualResult:
    required = [f for f in result.failures if f.required]
    if not required:
        return result
    reasons = "; ".join(f"{f.visual_id} ({f.lesson_section_id}): {f.reason}" for f in required)
    if policy == "fail":
        raise VisualRequired(f"required visuals failed: {reasons}")
    warning = f"Required visuals failed and the visual policy is 'continue', so the lesson has no image for: {reasons}"
    return result.model_copy(update={"warnings": [*result.warnings, warning]})
