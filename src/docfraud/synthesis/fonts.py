"""The fonts with which imitations (IMI) are typed (ADR-0021, decision 3).

Fonts under the SIL Open Font Licence 1.1, from the repository of Google Fonts at a fixed commit,
pinned by size and SHA-256 like every other download (``docfraud synthesis download-fonts``).
They are not stored in this repository. A forger typing over a receipt uses whatever font the
editing program offers, so the set mixes a typewriter face, a modern fixed-width face and a
proportional one, each regular and bold.
"""

from pathlib import Path

from docfraud.data.download import PinnedFile

FONT_COMMIT = "01d5d9f05246fb7090a8403b81257535cb1a088b"
"""The commit of github.com/google/fonts the fonts are taken from (2026-09-28)."""

_LICENCE = "SIL Open Font License 1.1"


def _font(directory: str, filename: str, size: int, sha256: str, copyright_: str) -> PinnedFile:
    return PinnedFile(
        name=filename.removesuffix(".ttf"),
        url=f"https://raw.githubusercontent.com/google/fonts/{FONT_COMMIT}/ofl/{directory}/{filename}",
        filename=filename,
        size_bytes=size,
        sha256=sha256,
        homepage=f"https://github.com/google/fonts/tree/{FONT_COMMIT}/ofl/{directory}",
        citation=f"{copyright_}; {_LICENCE}",
    )


_COURIER = "Copyright 2015 The Courier Prime Project Authors"
_PLEX = "Copyright 2017 IBM Corp., Reserved Font Name Plex"
_LATO = "Copyright 2010-2014 tyPoland Lukasz Dziedzic, Reserved Font Name Lato"

FONTS: dict[str, tuple[PinnedFile, PinnedFile]] = {
    "Courier Prime": (
        _font(
            "courierprime",
            "CourierPrime-Regular.ttf",
            71_188,
            "72f793376f8e2841656bf21d77a5de010f2929bd6956a22ee848ad0c7eb978af",
            _COURIER,
        ),
        _font(
            "courierprime",
            "CourierPrime-Bold.ttf",
            72_856,
            "ff1f38786c849d1c41fa8e447960abdb2bd75fdfb0cfcdeb524fad65a5af3638",
            _COURIER,
        ),
    ),
    "IBM Plex Mono": (
        _font(
            "ibmplexmono",
            "IBMPlexMono-Regular.ttf",
            135_580,
            "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46",
            _PLEX,
        ),
        _font(
            "ibmplexmono",
            "IBMPlexMono-Bold.ttf",
            137_784,
            "ac27abd6450a64dd94467580a02fe6235156d5b92f2926ebbc8e7489df64e0be",
            _PLEX,
        ),
    ),
    "Lato": (
        _font(
            "lato",
            "Lato-Regular.ttf",
            656_568,
            "d636e4683231f931eda222d588e944d082bfd3bdba02f928bee461c0f185b251",
            _LATO,
        ),
        _font(
            "lato",
            "Lato-Bold.ttf",
            656_544,
            "8a0aace75d33794eece4b28187bfc1df0bbd2888b5d8a56e01788c8d65d16be1",
            _LATO,
        ),
    ),
}
"""Each family's regular and bold file."""


def font_files() -> list[PinnedFile]:
    """Every font file, in a fixed order."""
    return [file for pair in FONTS.values() for file in pair]


def font_path(directory: Path, family: str, *, bold: bool) -> Path:
    """Where a downloaded font lies.

    Raises:
        FileNotFoundError: If the font has not been downloaded.
    """
    path = directory / FONTS[family][int(bold)].filename
    if not path.is_file():
        msg = f"{path} is missing: run docfraud synthesis download-fonts"
        raise FileNotFoundError(msg)
    return path
