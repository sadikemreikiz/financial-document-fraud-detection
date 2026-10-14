"""Systems 2 to 4 of ADR-0014: one network trained on synthetic, real or both forgeries (ADR-0022).

Except for :mod:`~docfraud.forensics.learned.pretrained`, the modules import PyTorch when they are
imported, so command-line code imports them only inside the commands that need them.
"""
