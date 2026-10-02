The video planner is deterministic: it does not call a model. It decides how an approved, narrated presentation
becomes a video, using fixed rules, and never touches files, storage, FFmpeg or provider SDKs.

- One video segment per slide, in timeline order, with exactly the slide's start and end time from the
  PresentationTimeline. The timeline is authoritative; nothing is re-timed.
- The visual is the slide's own IMAGE_ASSET when it has one the compositor can draw (PNG or JPEG), placed on a
  slide card with the slide's title and text. Without one, the slide card alone (neutral background, deck title,
  slide title, a few lines of content). No image is ever generated.
- Narration: every AUDIO_ASSET of the slide, at its timeline position. A slide without narration is silence for
  its whole duration.
- Subtitles: the narration text of each AUDIO_ASSET, split into cues of at most `max_lines` lines of
  `max_chars_per_line` characters, timed inside the segment's audio in proportion to their length. No
  speech-to-text.
- Transitions: a cut into the first slide; the configured transition (cut by default, or fade) everywhere else.
