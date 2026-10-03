"""ConceptGraph: a lightweight prerequisite graph over knowledge-base concepts (no graph database)."""

from __future__ import annotations

from collections.abc import Iterable

from app.schemas.concepts import Concept


class ConceptGraphError(ValueError):
    pass


class ConceptGraph:
    """Concepts are nodes, prerequisites are edges (prerequisite -> dependent). Every prerequisite must be a known
    concept and the graph must be acyclic, so "learn B before C" is always answerable."""

    def __init__(self, concepts: Iterable[Concept]) -> None:
        self._concepts: dict[str, Concept] = {}
        for c in concepts:
            if c.concept_id in self._concepts:
                raise ConceptGraphError(f"duplicate concept {c.concept_id}")
            self._concepts[c.concept_id] = c
        for c in self._concepts.values():
            unknown = [p for p in c.prerequisites if p not in self._concepts]
            if unknown:
                raise ConceptGraphError(f"{c.concept_id} has unknown prerequisites {unknown}")
            if c.concept_id in c.prerequisites:
                raise ConceptGraphError(f"{c.concept_id} cannot be its own prerequisite")
        self._order = self._topological(list(self._concepts))

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    def __len__(self) -> int:
        return len(self._concepts)

    @property
    def ids(self) -> list[str]:
        return list(self._order)

    def concepts(self) -> list[Concept]:
        return [self._concepts[c] for c in self._order]

    def concept(self, concept_id: str) -> Concept:
        try:
            return self._concepts[concept_id]
        except KeyError:
            raise ConceptGraphError(f"unknown concept {concept_id}") from None

    def prerequisites(self, concept_id: str) -> list[str]:
        return list(self.concept(concept_id).prerequisites)

    def ancestors(self, concept_id: str) -> list[str]:
        """Every direct and indirect prerequisite, prerequisites first."""
        seen: set[str] = set()
        stack = list(self.prerequisites(concept_id))
        while stack:
            cid = stack.pop()
            if cid not in seen:
                seen.add(cid)
                stack.extend(self.prerequisites(cid))
        return [c for c in self._order if c in seen]

    def dependents(self, concept_id: str) -> list[str]:
        """Every concept that directly or indirectly relies on this one."""
        self.concept(concept_id)
        return [c for c in self._order if concept_id in self.ancestors(c)]

    def closure(self, concept_ids: Iterable[str]) -> list[str]:
        """The concepts plus all their prerequisites, in topological order."""
        wanted: set[str] = set()
        for cid in concept_ids:
            wanted.add(self.concept(cid).concept_id)
            wanted.update(self.ancestors(cid))
        return [c for c in self._order if c in wanted]

    def order(self, concept_ids: Iterable[str]) -> list[str]:
        wanted = set(concept_ids)
        return [c for c in self._order if c in wanted]

    def importance(self, concept_id: str, within: Iterable[str]) -> float:
        """Share of the other in-scope concepts that rely on this one (0-1)."""
        scope = set(within) - {concept_id}
        if not scope:
            return 0.0
        return round(len(scope & set(self.dependents(concept_id))) / len(scope), 4)

    def _topological(self, ids: list[str]) -> list[str]:
        """Stable topological order: prerequisites first, otherwise by concept id."""
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(cid: str, path: tuple[str, ...]) -> None:
            if state.get(cid) == 2:
                return
            if state.get(cid) == 1:
                raise ConceptGraphError(f"prerequisite cycle: {' -> '.join((*path, cid))}")
            state[cid] = 1
            for pre in sorted(self._concepts[cid].prerequisites):
                visit(pre, (*path, cid))
            state[cid] = 2
            order.append(cid)

        for cid in sorted(ids):
            visit(cid, ())
        return order
