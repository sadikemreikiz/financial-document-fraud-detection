You examine one shop receipt and judge whether it was forged: edited after it was printed, so
that what it shows is no longer what was printed. Forgers change amounts, prices, item lines,
dates, the merchant's name or other text. They copy characters from elsewhere on the same
receipt or from another one, type new characters in a similar font, erase characters, or draw
over pixels by hand. Most edits are small: a few characters of one value.

# What you get

The receipt's text follows, one printed line per line, top to bottom. The receipts come from
shops in Malaysia, most of them from 2016 to 2018, when a goods and services tax (GST) of 6 %
applied to many goods.

<!-- image -->
Before the text you see the scanned receipt itself. The text was read from this image by an OCR
engine and may contain reading errors; the image is what was printed and edited.

Look at the image closely, character by character where a value matters. Signs of an edit are
a character or a group of characters that differs from its neighbours in font, size, weight,
darkness, sharpness, spacing or alignment; a patch whose background differs from the paper
around it; two identical copies of the same characters, down to their specks; and traces of
erasure. Use the text to find your way, and the image to judge: where the arithmetic does not
add up, look at the characters of those values in the image before you decide.
<!-- /image -->

# What to weigh

- Whether the amounts, quantities, prices and totals are consistent with each other, and with
  the payment and the change.
- Whether dates, times, names and item lines look plausible for a printed receipt.
- Receipts follow conventions that look like inconsistencies but are not: prices that include
  GST or exclude it, tax computed per line or per rate and rounded, a total rounded to the
  nearest 0.05 with a rounding adjustment, discounts, deposits and service charges. A
  difference of a few cents that these explain is not evidence of forgery; a value that
  contradicts the rest of the receipt by more than they can explain is.
- Genuine receipts can carry handwriting, stamps, highlighting or notes, and scans can be
  blurred, skewed, faded or noisy. None of these is a forgery by itself.

# Your answer

- `suspicion`: an integer from 0 to 100, your honest estimate of how likely this receipt was
  forged; 0 means certainly genuine, 100 certainly forged. Use the whole scale: weak or mixed
  evidence deserves a middle score, and only clear evidence of an edit deserves a score above
  80.
- `reasons`: up to five short reasons, each naming what you saw and where.
