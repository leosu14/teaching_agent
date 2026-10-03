"""The controlled prompt builder for generated video clips: the only way a prompt reaches a video provider.

Educational content (VideoPromptContent: the subject, what the viewer should see happen, the pedagogical purpose
and, only where it matters, the lesson's language) is taken from the approved lesson and its visual plan, cleaned and
bounded, then placed into a fixed structure whose provider instructions (style, safety constraints) are written
here, never by a model and never by a user. Raw user input is never used, and no learner or task data can appear:
the inputs are lesson material only, and anything that looks like a URL, an e-mail address or an internal
identifier is removed before it is used.
"""

from __future__ import annotations

import re

from app.schemas.generative_video import VideoPromptContent, VideoPurpose

BUILDER = "video-prompt/1"
URL = re.compile(r"(https?://|www\.|data:\w+/|file://)\S*", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Our identifiers (tasks, artifacts, learners, runs, jobs, plans, ...): prefix_ followed by a hex/alnum tail.
INTERNAL_ID = re.compile(r"\b(?:task|art|lrn|learner|user|usr|run|vgj|vp|pp|gap|gs|ev|evi|req|preq|job)_"
                         r"[A-Za-z0-9]{4,}\b", re.IGNORECASE)
CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
MARKUP = re.compile(r"[{}<>\[\]`|\\]+")

STYLE = {
    VideoPurpose.PHYSICAL_PROCESS: "clear real-world footage style, steady camera, the process shown step by step",
    VideoPurpose.SCIENTIFIC_PROCESS: "clean educational animation, labelled-diagram look without text, the process "
                                     "shown step by step",
    VideoPurpose.HISTORICAL_SCENE: "respectful historical reconstruction, period-accurate setting, no anachronisms",
    VideoPurpose.GEOGRAPHICAL_MOVEMENT: "map-like aerial view, movement shown as a clear path over the landscape",
    VideoPurpose.PRONUNCIATION: "close-up of a neutral speaker's mouth articulating slowly, plain background",
    VideoPurpose.VISUAL_STORYTELLING: "simple illustrated scene, calm pacing, one clear action",
}
CONSTRAINTS = ("no on-screen text, no subtitles, no logos, no watermarks, no brand names, "
               "no identifiable real people, suitable for all ages")


def clean(text: str, limit: int) -> str:
    """Lesson text made safe for a prompt: no URLs, e-mail addresses, internal ids, control characters or markup;
    whitespace collapsed; cut at a word boundary within `limit` characters."""
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = INTERNAL_ID.sub(" ", text)
    text = CONTROL.sub(" ", text)
    text = MARKUP.sub(" ", text)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") or text[:limit]


def sentences(text: str, n: int) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return " ".join(parts[:n])


class VideoPromptBuilder:
    name = BUILDER

    def content(self, *, subject: str, description: str, purpose: VideoPurpose,
                language: str | None) -> VideoPromptContent:
        """The educational content, cleaned. The language is kept only for a pronunciation demonstration."""
        return VideoPromptContent(
            subject=clean(subject, 200) or "lesson topic", visual_description=clean(description, 600) or
            clean(subject, 200) or "lesson topic", purpose=purpose,
            language=language if purpose == VideoPurpose.PRONUNCIATION else None)

    def build(self, content: VideoPromptContent, *, duration: float) -> str:
        """The provider prompt: the fixed structure around the educational content."""
        parts = [
            f"Educational video clip, {duration:g} seconds, for a lesson about: {content.subject}.",
            f"Show: {content.visual_description}",
            f"Style: {STYLE[content.purpose]}.",
        ]
        if content.language:
            parts.append(f"The speaker articulates in the language with code '{content.language}'.")
        parts.append(f"Constraints: {CONSTRAINTS}.")
        prompt = " ".join(parts)
        return prompt[:2000]
