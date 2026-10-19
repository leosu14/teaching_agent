"""Budget exhaustion as callers see it.

A task that reaches a budget limit fails with the BudgetExceededError category and can be resumed once the limit is
raised (unchanged). Interactive requests (a teacher turn, the grading of an answer) cannot wait in a FAILED state, so
they answer `BudgetExhausted`: HTTP 429 with fixed text, telling the client to retry later, distinct from 503 (the
teacher or a provider is unavailable). Which limit was reached stays in the internal `budget.exhausted` event.
"""

from __future__ import annotations

from app.observability.budget import BudgetExceededError


class BudgetExhausted(RuntimeError):
    MESSAGE = "The spending limit for this work has been reached. Try again later."

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)

    def __str__(self) -> str:
        return self.MESSAGE


def budget_cause(exc: BaseException | None) -> BudgetExceededError | None:
    """The BudgetExceededError in an exception's cause or context chain, if any."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, BudgetExceededError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None
