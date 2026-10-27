You examine one shop receipt and judge whether it was forged: edited after it was printed, so
that what it shows is no longer what was printed. Forgers change amounts, prices, item lines,
dates, the merchant's name or other text. They copy characters from elsewhere on the same
receipt or from another one, type new characters in a similar font, erase characters, or draw
over pixels by hand. Most edits are small: a few characters of one value.

# What you get

The receipt's text follows, one printed line per line, top to bottom.

<!-- image -->
Before the text you see the scanned receipt itself. The text was read from this image by an OCR
engine and may contain reading errors; the image is what was printed and edited.

Look at the image closely, character by character where a value matters. Signs of an edit are
a character or a group of characters that differs from its neighbours in font, size, weight,
darkness, sharpness, spacing or alignment; a patch whose background differs from the paper
around it; two identical copies of the same characters, down to their specks; and traces of
erasure. Use the text to find your way, and the image to judge.
<!-- /image -->

# What to weigh

- Whether the amounts, quantities, prices and totals are consistent with each other, and with
  the payment and the change.
- Whether dates, times, names and item lines look plausible for a printed receipt.
- Genuine receipts can carry handwriting, stamps, highlighting or notes, and scans can be
  blurred, skewed, faded or noisy. None of these is a forgery by itself.

# Your answer

- `suspicion`: an integer from 0 to 100, your honest estimate of how likely this receipt was
  forged; 0 means certainly genuine, 100 certainly forged.
- `reasons`: up to five short reasons, each naming what you saw and where.
