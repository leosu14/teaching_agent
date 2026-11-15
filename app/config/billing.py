"""How each paid provider and model is billed, so the run-wide cap (TA_BUDGET_TOTAL_MAX_COST_USD) is enforced only
where a configured price can bound a request before it is sent. A combination this module cannot bound is refused
at startup while the cap is set (`billing_problems`), never reserved at a guess.

OpenAI (https://openai.com/api/pricing and https://platform.openai.com/docs/pricing, "Audio" and "Image
generation"; the rates themselves are configured by the operator, never written here):

- Speech. `tts-1` and `tts-1-hd` are billed per input character: TTS_PRICE_PER_MCHAR is USD per million
  characters, and a request's characters are known before it is sent, so its reservation is exact. Speed and voice
  do not change that bill. The `gpt-4o*-tts` models are billed by text input tokens, their `instructions` text, and
  audio output tokens; the audio length depends on the voice and `speed` (down to 0.25x), so no price per character
  bounds it: refused under the cap.
- Images. One image per request (`n=1`) at a quality and size pinned by IMAGE_QUALITY and IMAGE_SIZE (never the
  vendor's `auto` default). `dall-e-3` and `dall-e-2` are billed per image by quality and size:
  IMAGE_PRICE_PER_IMAGE is that per-image price. GPT image models (`gpt-image-*`) are billed by tokens: the image
  output tokens are fixed by quality and size (OpenAI publishes the per-image price of each), which
  IMAGE_PRICE_PER_IMAGE prices, and the prompt's text input tokens, which IMAGE_INPUT_PRICE_PER_MTOK (USD per
  million) prices at an upper bound of its UTF-8 bytes.

Mock providers are free, and billed per character and per image in the rehearsal's priced runs."""

from __future__ import annotations

MOCK = "mock"

# OpenAI speech models billed per input character (dated snapshots of them included).
OPENAI_TTS_PER_CHARACTER = ("tts-1", "tts-1-hd", "tts-1-1106", "tts-1-hd-1106")
OPENAI_TTS_DEFAULT = "gpt-4o-mini-tts"  # app/providers/tts/openai.py, DEFAULT_MODEL

OPENAI_IMAGE_DEFAULT = "gpt-image-1"  # app/providers/image/openai.py, DEFAULT_MODEL
# Model family -> (qualities it accepts that fix the bill, sizes it generates).
OPENAI_IMAGE_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "gpt-image": (("low", "medium", "high"), ("1024x1024", "1536x1024", "1024x1536")),
    "dall-e-3": (("standard", "hd"), ("1024x1024", "1792x1024", "1024x1792")),
    "dall-e-2": (("standard",), ("256x256", "512x512", "1024x1024")),
}


def image_family(model: str) -> str | None:
    return next((family for family in OPENAI_IMAGE_FAMILIES if model.startswith(family)), None)


def token_billed_image(provider: str, model: str | None) -> bool:
    """The image's prompt is billed too (GPT image models): IMAGE_INPUT_PRICE_PER_MTOK prices it."""
    return provider == "openai" and image_family(model or OPENAI_IMAGE_DEFAULT) == "gpt-image"


def image_option_problems(model: str | None, quality: str | None, size: str | None) -> list[str]:
    """IMAGE_QUALITY / IMAGE_SIZE that the OpenAI image model does not accept (checked whenever they are set)."""
    name = model or OPENAI_IMAGE_DEFAULT
    family = image_family(name)
    if family is None:
        return [f"IMAGE_MODEL='{name}' is not a known OpenAI image model (known: "
                f"{', '.join(f + '*' for f in OPENAI_IMAGE_FAMILIES)})"] if quality or size else []
    qualities, sizes = OPENAI_IMAGE_FAMILIES[family]
    problems = []
    if quality is not None and quality not in qualities:
        problems.append(f"IMAGE_QUALITY='{quality}' is not a quality of {name} (one of: {', '.join(qualities)})")
    if size is not None and size not in sizes:
        problems.append(f"IMAGE_SIZE='{size}' is not a size {name} generates (one of: {', '.join(sizes)})")
    return problems


def billing_problems(*, tts_chain: list[str], tts_model: str | None, image_chain: list[str],
                     image_model: str | None, image_quality: str | None, image_size: str | None,
                     image_input_price_per_mtok: float | None) -> list[str]:
    """Why a paid TTS or image request could cost more than its reservation under the cap: a model whose billing the
    configured price cannot bound, or a billing parameter left to the vendor's default. Empty: every request is
    bounded before it is sent."""
    problems = []
    for provider in tts_chain:
        if provider == MOCK:
            continue
        if provider != "openai":
            problems.append(f"TTS provider '{provider}': its billing is not known here, so no price can bound it")
            continue
        model = tts_model or OPENAI_TTS_DEFAULT
        if model not in OPENAI_TTS_PER_CHARACTER:
            problems.append(f"TTS_MODEL='{model}' is billed by audio output tokens, whose number depends on the "
                            "voice and speed and is not known before the request; set TTS_MODEL to a model billed "
                            f"per character ({', '.join(OPENAI_TTS_PER_CHARACTER[:2])}), priced by TTS_PRICE_PER_MCHAR")
    for provider in image_chain:
        if provider == MOCK:
            continue
        if provider != "openai":
            problems.append(f"image provider '{provider}': its billing is not known here, so no price can bound it")
            continue
        model = image_model or OPENAI_IMAGE_DEFAULT
        if image_family(model) is None:
            problems.append(f"IMAGE_MODEL='{model}' is not a known OpenAI image model, so no price can bound it")
            continue
        if image_quality is None:
            problems.append(f"IMAGE_QUALITY is not set: {model}'s price depends on the quality, and the vendor's "
                            "default ('auto') is not a fixed one")
        if image_size is None:
            problems.append(f"IMAGE_SIZE is not set: {model}'s price depends on the size, which would otherwise "
                            "follow each image's aspect ratio")
        problems += image_option_problems(model, image_quality, image_size)
        if token_billed_image(provider, model) and image_input_price_per_mtok is None:
            problems.append(f"IMAGE_INPUT_PRICE_PER_MTOK is not set: {model} also bills the prompt's text input "
                            "tokens")
    return problems
