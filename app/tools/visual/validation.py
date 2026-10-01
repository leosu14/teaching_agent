"""Image validation: every image is checked against its own bytes before it can become an asset.

Format and dimensions are read from the bytes (never trusted from metadata), the checksum is recomputed,
the aspect ratio is compared with the requirement, and the provenance is checked: a searched image needs
a source (and a licence and credit when attribution is required); a generated image needs provider, model,
timestamp and a prompt hash that matches its prompt. Failures are structured errors, never silent.
"""

from __future__ import annotations

import hashlib

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.visual import (
    GenerationAttribution,
    ImageValidationError,
    ImageValidationReport,
    ImageValidationRequest,
    MeasuredImage,
    SearchAttribution,
    aspect_matches,
)
from app.tools.base import Tool, ToolError
from app.utils.images import ImageProbeError, probe_image

ALLOWED_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"})
MIN_SIDE = 64
MAX_SIDE = 8192


class ImageValidator:
    name = "image-validator-v1"

    def validate(self, req: ImageValidationRequest, content: bytes) -> ImageValidationReport:
        errors: list[ImageValidationError] = []
        checks: list[str] = []

        def fail(code, field, message, expected=None, actual=None) -> None:
            errors.append(ImageValidationError(code=code, field=field, message=message,
                                               expected=None if expected is None else str(expected),
                                               actual=None if actual is None else str(actual)))

        def report(measured: MeasuredImage | None = None) -> ImageValidationReport:
            return ImageValidationReport(valid=not errors, errors=errors, checks=checks, measured=measured,
                                         validator=self.name)

        checks.append("content")
        if not content:
            fail("empty_content", "object", "the image has no bytes")
            return report()
        checks.append("checksum")
        digest = hashlib.sha256(content).hexdigest()
        if digest != req.object.checksum or len(content) != req.object.size_bytes:
            fail("checksum_mismatch", "object.checksum", "stored bytes do not match the recorded checksum",
                 req.object.checksum, digest)
        checks.append("format")
        try:
            probe = probe_image(content)
        except ImageProbeError as exc:
            fail("unreadable_image", "object", f"the bytes are not a readable image: {exc}")
            return report()
        measured = MeasuredImage(format=probe.format, media_type=probe.media_type, width=probe.width,
                                 height=probe.height, checksum=digest, size_bytes=len(content))
        if probe.media_type not in ALLOWED_MEDIA_TYPES:
            fail("unsupported_format", "format", "unsupported image format", sorted(ALLOWED_MEDIA_TYPES),
                 probe.media_type)
        if probe.media_type != req.declared_format:
            fail("format_mismatch", "format", "the image format differs from the declared one",
                 req.declared_format, probe.media_type)
        if probe.media_type != req.object.media_type:
            fail("format_mismatch", "object.media_type", "the stored MIME type differs from the image format",
                 probe.media_type, req.object.media_type)

        checks.append("dimensions")
        if (probe.width, probe.height) != (req.declared_width, req.declared_height):
            fail("dimension_mismatch", "dimensions", "the image size differs from the declared size",
                 f"{req.declared_width}x{req.declared_height}", f"{probe.width}x{probe.height}")
        if min(probe.width, probe.height) < MIN_SIDE:
            fail("too_small", "dimensions", f"both sides must be at least {MIN_SIDE}px", MIN_SIDE,
                 f"{probe.width}x{probe.height}")
        if max(probe.width, probe.height) > MAX_SIDE:
            fail("too_large", "dimensions", f"no side may exceed {MAX_SIDE}px", MAX_SIDE,
                 f"{probe.width}x{probe.height}")
        if req.expected_aspect_ratio:
            checks.append("aspect_ratio")
            if not aspect_matches(probe.width, probe.height, req.expected_aspect_ratio):
                fail("aspect_ratio_mismatch", "aspect_ratio", "the image does not have the expected aspect ratio",
                     req.expected_aspect_ratio, f"{probe.width}:{probe.height}")

        checks.append("provenance")
        attribution = req.attribution
        expected_kind = "search" if req.origin == "search" else "generated"
        if attribution.kind != expected_kind:
            fail("origin_mismatch", "attribution.kind", "the attribution does not match the image origin",
                 expected_kind, attribution.kind)
        elif isinstance(attribution, SearchAttribution):
            self._search(attribution, req.attribution_required, checks, fail)
        else:
            self._generated(attribution, probe.width, probe.height, checks, fail)
        return report(measured)

    def _search(self, a: SearchAttribution, required: bool, checks: list[str], fail) -> None:
        for field in ("image_url", "source_url"):
            if not getattr(a, field).startswith(("https://", "http://")):
                fail("missing_source", f"attribution.{field}", "a searched image needs its source URL")
        if not a.provider.strip():
            fail("missing_source", "attribution.provider", "a searched image needs its provider")
        if required:
            checks.append("attribution")
            if a.license is None:
                fail("missing_license", "attribution.license", "attribution is required but no licence is known")
            if not (a.creator or a.attribution_text or a.publisher):
                fail("missing_attribution", "attribution", "attribution is required but no creator, credit line "
                     "or publisher is known")

    def _generated(self, a: GenerationAttribution, width: int, height: int, checks: list[str], fail) -> None:
        checks.append("generation_metadata")
        if not a.provider.strip() or not a.model.strip():
            fail("missing_source", "attribution", "a generated image needs its provider and model")
        if a.prompt_hash != hashlib.sha256(a.prompt.encode("utf-8")).hexdigest():
            fail("invalid_generation_metadata", "attribution.prompt_hash", "the prompt hash does not match the prompt")
        if a.generated_at.tzinfo is None:
            fail("invalid_generation_metadata", "attribution.generated_at", "the generation time needs a timezone")
        if (a.width, a.height) != (width, height):
            fail("invalid_generation_metadata", "attribution.size", "the recorded size differs from the image",
                 f"{a.width}x{a.height}", f"{width}x{height}")


class ImageValidationTool(Tool[ImageValidationRequest, ImageValidationReport]):
    name = "image.validate"
    description = "Validate a stored image against its bytes: checksum, format, dimensions, aspect ratio, source, " \
                  "attribution and generation metadata. Returns structured errors."
    input_model = ImageValidationRequest
    output_model = ImageValidationReport
    permissions = frozenset({"artifact:read"})

    def __init__(self, artifacts: ArtifactService, validator: ImageValidator | None = None) -> None:
        self._artifacts = artifacts
        self._validator = validator or ImageValidator()

    async def run(self, data: ImageValidationRequest, scope: ExecutionScope) -> ImageValidationReport:
        try:
            content = self._artifacts.read_object(data.object.uri)
        except (OSError, ValueError) as exc:
            raise ToolError(f"cannot read image object {data.object.uri}: {exc}") from exc
        return self._validator.validate(data, content)
