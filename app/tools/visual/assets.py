"""IMAGE_ASSET creation. The tool re-validates the image itself, so no invalid image can become an asset."""

from __future__ import annotations

import json

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.visual import ImageAssetRequest, ImageValidationReport, ImageValidationRequest
from app.tools.base import Tool, ToolError
from app.tools.visual.validation import ImageValidator


class ImageAssetRejected(ToolError):
    """The image failed validation. `report` holds the structured errors."""

    def __init__(self, report: ImageValidationReport) -> None:
        super().__init__("image failed validation: " + json.dumps(
            [e.model_dump(exclude_none=True) for e in report.errors]))
        self.report = report


class ImageAssetTool(Tool[ImageAssetRequest, Artifact]):
    name = "image.create_asset"
    description = "Create an IMAGE_ASSET artifact pointing at a validated, stored image; its metadata keeps the " \
                  "visual requirement, source or generation attribution, selection and validation."
    input_model = ImageAssetRequest
    output_model = Artifact
    permissions = frozenset({"artifact:write", "artifact:read"})

    def __init__(self, artifacts: ArtifactService, validator: ImageValidator | None = None) -> None:
        self._artifacts = artifacts
        self._validator = validator or ImageValidator()

    async def run(self, data: ImageAssetRequest, scope: ExecutionScope) -> Artifact:
        if scope.task_id is None:
            raise ToolError("image assets can only be created inside a task")
        asset = data.asset
        try:
            content = self._artifacts.read_object(asset.object.uri)
        except (OSError, ValueError) as exc:
            raise ToolError(f"cannot read image object {asset.object.uri}: {exc}") from exc
        report = self._validator.validate(ImageValidationRequest(
            object=asset.object, declared_width=asset.width, declared_height=asset.height,
            declared_format=asset.format, expected_aspect_ratio=data.expected_aspect_ratio,
            attribution=asset.attribution, attribution_required=data.attribution_required, origin=asset.origin,
        ), content)
        if not report.valid:
            raise ImageAssetRejected(report)
        asset = asset.model_copy(update={"validation": report})
        return self._artifacts.store_object(
            task_id=scope.task_id, name=data.name, type=ArtifactType.IMAGE_ASSET, obj=asset.object,
            provider=asset.attribution.provider, parent_ids=data.parent_ids,
            metadata=asset.model_dump(mode="json"), scope=scope,
        )
