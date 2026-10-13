"""Dataset-independent data model.

Every dataset adapter converts its native format into these types, so that everything downstream
(reading, detectors, evaluation) depends on one contract only.
"""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Split(StrEnum):
    """Dataset partition a document belongs to."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Label(StrEnum):
    """Ground truth at document level. Forged is the positive class."""

    AUTHENTIC = "authentic"
    FORGED = "forged"


class Technique(StrEnum):
    """How a region was forged, following the taxonomy of Tornés et al. (2023)."""

    CPI = "cpi"  # copy and paste inside the same document
    CPO = "cpo"  # copy and paste from another document
    IMI = "imi"  # text box imitating the original font
    CUT = "cut"  # deletion of characters without replacement
    PIX = "pix"  # free-hand pixel edits
    OTHER = "other"


class Entity(StrEnum):
    """Which kind of information a forged region changed."""

    TOTAL_PAYMENT = "total_payment"
    PRODUCT = "product"
    METADATA = "metadata"  # date or time
    COMPANY = "company"
    OTHER = "other"


class RegionRole(StrEnum):
    """Whether a region was modified or served as the source of a copy."""

    MODIFIED = "modified"
    SOURCE = "source"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BBox(_Frozen):
    """Axis-aligned box in pixel coordinates; ``x``, ``y`` is the top-left corner."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def x2(self) -> int:
        """Exclusive right edge."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Exclusive bottom edge."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height

    def fits_within(self, width: int, height: int) -> bool:
        """Return whether the box lies entirely inside an image of the given size."""
        return self.x2 <= width and self.y2 <= height

    def overlaps(self, other: BBox) -> bool:
        """Return whether the two boxes share at least one pixel."""
        return self.x < other.x2 and other.x < self.x2 and self.y < other.y2 and other.y < self.y2


class Region(_Frozen):
    """An annotated region of a forged document."""

    box: BBox
    role: RegionRole
    techniques: frozenset[Technique] = frozenset()
    entity: Entity | None = None


class Document(_Frozen):
    """One document instance with its ground truth, if known.

    Attributes:
        dataset: Name of the dataset the document comes from.
        doc_id: Identifier unique within the dataset.
        source_id: Identity of the underlying physical document across datasets. Two datasets
            built on the same source (for example SROIE) share it, which is what cross-dataset
            leakage checks compare.
        split: Partition, or ``None`` for documents outside any dataset (e.g. uploads).
        label: Ground truth, or ``None`` when unknown.
        image_path: Path to the document image as distributed.
        transcript_path: Path to a transcript, if the dataset provides one.
        regions: Annotated forgery regions (modified areas and copy sources).
        has_region_annotations: Whether region annotations exist. ``False`` for a forged
            document means its forgery cannot be localised, not that nothing was modified.
        digital_annotation: Benign digitally added text (e.g. case numbers) is present.
        handwritten_annotation: Benign handwriting, highlighting or stamps are present.
        editing_software: Software used to forge the document, when recorded.
    """

    dataset: str
    doc_id: str
    source_id: str
    split: Split | None = None
    label: Label | None = None
    image_path: Path
    transcript_path: Path | None = None
    regions: tuple[Region, ...] = ()
    has_region_annotations: bool = False
    digital_annotation: bool = False
    handwritten_annotation: bool = False
    editing_software: str | None = None

    @property
    def uid(self) -> str:
        """Identifier unique across datasets."""
        return f"{self.dataset}/{self.doc_id}"

    @property
    def modified_regions(self) -> tuple[Region, ...]:
        """Regions that were actually changed, excluding copy sources."""
        return tuple(r for r in self.regions if r.role is RegionRole.MODIFIED)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.regions and not self.has_region_annotations:
            msg = f"{self.uid}: regions given but has_region_annotations is False"
            raise ValueError(msg)
        if self.label is Label.AUTHENTIC and self.modified_regions:
            msg = f"{self.uid}: authentic documents cannot have modified regions"
            raise ValueError(msg)
        return self
