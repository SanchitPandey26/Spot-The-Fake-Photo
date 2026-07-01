# Spot the Fake Photo — Submission

**Task:** Given a photo, decide if it is a real photo (0) or a photo of a screen (1).

---

## Files in this submission

| File                 | What it is                                                      |
|----------------------|-----------------------------------------------------------------|
| `predict.py`         | The main script — run this on any image                         |
| `features.py`        | Feature extraction logic (used internally by predict.py)        |
| `model_final.joblib` | Trained classifier — loaded automatically by predict.py         |
| `TRAINING.ipynb`     | Full training notebook: feature analysis, CV results, ROC curve |
| `NOTE.md`            | Half-page write-up: approach, accuracy, latency, cost           |

---

## How to run

### Step 1 — Install dependencies

```bash
pip install numpy scipy pillow scikit-learn joblib
```

That's it. No GPU, no deep learning framework needed.

### Step 2 — Run on an image

```bash
python predict.py some_image.jpg
```

It prints one number:
- **Close to 0** → real photo
- **Close to 1** → photo of a screen

**Example:**

```
$ python predict.py test.jpg
0.8357
```

A score of 0.8357 means the model is 84% confident this is a screen recapture.
The default decision threshold is 0.5 — anything above that is flagged as a screen.
---

## Folder structure (keep these together)

```
your-folder/
├── predict.py           ← run this
├── features.py          ← used by predict.py
├── model_final.joblib   ← used by predict.py
├── NOTE.md
└── TRAINING.ipynb
```

If you move any file out of the folder, `predict.py` will not find the model and will fail.

---

## Requirements

- Python 3.8 or higher
- Libraries: `numpy`, `scipy`, `Pillow`, `scikit-learn`, `joblib`

Tested with:
- numpy 2.4.4
- scipy 1.17.1
- Pillow 12.1.1
- scikit-learn 1.8.0
- joblib 1.5.3

Any reasonably recent version of these libraries should work fine.

---

## What happens inside (brief)

When you run `predict.py image.jpg`:

1. The image is opened, center-cropped, and resized to 768×768
2. `features.py` extracts 19 numbers from it (frequency patterns, colour balance,
   texture stats, etc.) — this takes ~300 ms on a laptop
3. `model_final.joblib` (an 11 KB SVM classifier) turns those 19 numbers into
   a probability score between 0 and 1
4. That score is printed

---

## Accuracy & latency

- **94.2% accuracy** on my own dataset (100-fold cross-validation, estimate)
- **~340 ms per image** on a laptop CPU once Python is running (warm)
- Cold start is ~1.2 s the first time (Python + model loading), not per image

Full details in `NOTE.md` and `TRAINING.ipynb`.
