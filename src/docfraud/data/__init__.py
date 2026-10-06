"""Datasets: acquisition, the document model, normalisation, splits and audits."""

from docfraud.data.models import (
    BBox,
    Document,
    Entity,
    Label,
    Region,
    RegionRole,
    Split,
    Technique,
)

__all__ = ["BBox", "Document", "Entity", "Label", "Region", "RegionRole", "Split", "Technique"]
