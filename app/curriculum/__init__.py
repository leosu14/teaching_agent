"""The curriculum engine: long-term learning goals -> versioned curricula of objectives -> objective progress ->
the next learning action. Deterministic, no model calls and no I/O beyond the repository it is given.

It consumes the existing learner model (app/learner, app/schemas/pedagogy.LearnerModel) and concept graph
(app/pedagogy/graph.py); it never writes mastery. A model may only word a curriculum (app/agents/curriculum)."""
