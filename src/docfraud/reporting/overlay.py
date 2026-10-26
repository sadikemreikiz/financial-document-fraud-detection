"""The receipt with the boxes of the evidence towards forgery drawn on it, for the reviewer."""

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from docfraud.detectors.finding import COPY_MOVE, LINKING, RULES, SYSTEM_1, SYSTEM_2
from docfraud.reporting.assessment import Assessment

COLOURS = {
    RULES: (0, 114, 178),
    SYSTEM_2: (213, 94, 0),
    SYSTEM_1: (230, 159, 0),
    COPY_MOVE: (204, 121, 167),
    LINKING: (0, 158, 115),
}
"""One colour per detector, from a palette that colour-blind readers can tell apart."""

LINE = 3


def overlay(page: npt.NDArray[np.uint8], assessment: Assessment) -> Image.Image:
    """The grayscale page in colour, with the boxes of each piece of evidence towards forgery."""
    image = Image.fromarray(page).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in assessment.evidence:
        if item.contribution <= 0:
            continue
        for finding in item.findings:
            colour = COLOURS.get(finding.detector)
            if colour is None:
                continue
            for box in finding.boxes:
                draw.rectangle((box.x, box.y, box.x2 - 1, box.y2 - 1), outline=colour, width=LINE)
    return image
