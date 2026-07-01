# NOTE — Spot the Fake Photo

## What I built

An RBF-SVM classifier model that looks at 19 features to decide if a photo was taken
of a real object or of a screen showing a picture. No deep learning — just careful
image analysis plus a small classifier on top.

The core idea: when you photograph a screen, it leaves behind subtle physical traces
that a direct photo never has. I built features to catch each of them:

- **Screen pixel grid:** a screen's RGB subpixel pattern creates faint periodic ripples
  in the image's frequency spectrum. Real-world textures don't have this regularity.
- **Color temperature shift:** screens glow bluer and cooler than natural lighting.
  A photo of a screen picks this up in the channel balance.
- **Uniform backlight:** a screen lights itself evenly from behind, making local contrast
  across the image unusually uniform. Real scenes have varied, patchy lighting.
- **Double JPEG compression:** a screen image has already been JPEG-compressed once
  (to display it). Photographing it compresses it again. This double-pass leaves faint
  8-pixel-wide block seams that a single-shot photo doesn't have.
- **Colour fringing at edges:** the tiny red, green, blue subpixels on a screen are
  physically offset from each other. This creates very slight colour fringing at fine
  edges — detectable by checking how correlated the R, G, B channels are in the
  fine-detail layer of the image.
- **Sensor noise:** a real camera photo has characteristic sensor noise. Photographing
  a screen mixes in screen-refresh artifacts and moiré, which looks different from
  pure sensor noise under a fine-detail filter.

All 19 features feed into the RBF-SVM classifier. The whole model fits in an 11 KB file.

## How accurate is it

**94.2% accuracy** — measured using 100-fold repeated cross-validation (20 rounds of
5-fold CV with a random seed not used during any hyperparameter tuning), so this is a
stable, honest estimate rather than one lucky split. AUC is 0.957, meaning the model
ranks a randomly chosen screen photo above a randomly chosen real photo 95.7% of the time.

Real photos are almost never falsely flagged (98% pass correctly). Screens are caught
90% of the time — the remaining 10% are mostly shallow-angle shots where the moiré
pattern is faint.

This is just under the 95% target. The main reason is that all my training images
came from one phone + one laptop screen. More variety of screens (phones, TVs, different
pixel densities) would close the gap.

## Required numbers

**Latency:** ~340 ms per image on a laptop CPU (warm, model already loaded).
The bottleneck is reading and decoding a 4–7 MB full-resolution camera file.
For typical compressed phone uploads under 1 MB, expect ~40–80 ms.
Cold start (Python launching, model loading) is ~1.2 s once — not per image.

**Cost per image:**
- On-device: effectively free. The model is 11 KB and uses only standard
  maths operations — no GPU, no network call, runs fine on a phone CPU.
- Cloud server: at 340 ms per image on one CPU core, a cheap \$0.05/hr cloud
  instance handles ~3 images/second. That works out to roughly **$0.005 per
  1,000 images** — half a cent per thousand, just for compute.

## What I'd improve with more time

- Shoot images across multiple screen types (OLED phones, monitors, TVs) and more
  angles. That alone would likely push accuracy past 95%.
- Add a proper multi-scale frequency analysis instead of a single crop.
- Calibrate the decision threshold rather than leaving it at the default.

## For the curious: adapting & threshold selection

**As cheaters adapt:** The model can be retrained periodically. Any prediction in the
borderline 0.3–0.7 range can go to a human review queue, and confirmed cheats get added
to the training set. Tracking accuracy by device type over time flags when a new screen
model is slipping through.

**Choosing the cutoff:** The default 0.5 threshold isn't always right. Falsely blocking
a real user is usually worse than missing a cheat, so you'd
pick a point on the ROC curve with very low false-positive rate — say, flag automatically
above 0.8 and send 0.4–0.8 to human review rather than auto-blocking.
