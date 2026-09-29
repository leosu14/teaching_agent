"""Pluggable level frameworks. CEFR is one framework among others; nothing else assumes it."""

from __future__ import annotations


class UnknownFramework(KeyError):
    pass


class LevelFramework:
    def __init__(self, framework_id: str, name: str, levels: list[str]) -> None:
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ValueError("a level framework needs at least two distinct levels")
        self.framework_id = framework_id
        self.name = name
        self.levels = list(levels)

    def index(self, level: str) -> int:
        try:
            return self.levels.index(level)
        except ValueError:
            raise ValueError(f"'{level}' is not a level of framework {self.framework_id}") from None

    def score_for(self, level: str) -> float:
        """Midpoint of the level's band on the 0..1 scale."""
        return (self.index(level) + 0.5) / len(self.levels)

    def level_for(self, score: float) -> str:
        clamped = min(max(score, 0.0), 1.0)
        return self.levels[min(len(self.levels) - 1, int(clamped * len(self.levels)))]


class FrameworkRegistry:
    def __init__(self) -> None:
        self._frameworks: dict[str, LevelFramework] = {}

    def register(self, framework: LevelFramework) -> None:
        self._frameworks[framework.framework_id] = framework

    def get(self, framework_id: str) -> LevelFramework:
        try:
            return self._frameworks[framework_id]
        except KeyError:
            raise UnknownFramework(framework_id) from None

    def ids(self) -> list[str]:
        return sorted(self._frameworks)


CEFR = LevelFramework("cefr", "Common European Framework of Reference", ["A1", "A2", "B1", "B2", "C1", "C2"])
MASTERY_SCALE = LevelFramework("mastery", "Generic mastery scale",
                               ["novice", "beginner", "intermediate", "advanced", "expert"])


def default_frameworks() -> FrameworkRegistry:
    registry = FrameworkRegistry()
    registry.register(CEFR)
    registry.register(MASTERY_SCALE)
    return registry
