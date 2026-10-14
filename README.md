# Tomato Ripeness

An Android app that points your camera at a tomato and tells you whether it is
ready to eat — on screen and out loud.

> "This tomato is safe to eat."
> "Not ready to eat yet. Please wait a few more days."

No login, no account, no internet. Every frame is analysed on the device and
thrown away immediately.

## Running it

1. Open the **`android/`** folder in Android Studio (not the repository root).
2. Let Gradle sync, then Run on a physical device.
3. Allow the camera permission, point the box at one tomato.

A real device is strongly recommended — the emulator's fake camera has no tomato
in it. Build setup is AGP 8.13.2 / Gradle 8.14.3 / Kotlin 2.2.20, `minSdk 24`,
JDK 17+.

## What you see

The verdict updates live in the bottom panel: the stage, the sentence, advice,
and a confidence bar. It is **spoken aloud** whenever the verdict settles on
something new, and **"Check this tomato"** pops it up as a dialog and repeats it
on demand. The speaker button mutes the voice; the info button opens a screen
explaining the method, the stages, the limits, and the dataset credit.

## How the verdict is reached

A tomato ripens through a predictable sequence of colours — green, yellow,
orange, red — so the hue of the skin is a direct proxy for ripeness.

Each frame, about 12,000 pixels inside the focus box are converted to HSV.
Pixels that cannot be tomato skin are discarded: greys (walls, tables), deep
shadow, blown-out glare, and anything blue through violet. The survivors are
sorted into hue bands, each band carries a ripeness weight, and the weighted
mean becomes a **ripeness score** from 0.00 to 1.00.

| Score | Stage | Verdict |
|---|---|---|
| below 0.35 | Unripe | Not ready — about a week to go |
| 0.35 – 0.60 | Starting to turn | Not ready — roughly 3 to 5 days |
| 0.60 – 0.80 | Almost ripe | One or two more days |
| 0.80 and up | Ripe | **Safe to eat** |
| deep red, dark *and* dull | Overripe | Better cooked than raw |

Two guards keep this honest. If less than 18% of the box looks like tomato skin
at all, the app reports "no tomato found" instead of guessing. And verdicts are
majority-voted across the last seven frames, so the reading does not flicker
while you move the phone.

Confidence combines how much of the box is filled with how far the score sits
from a stage boundary — a score of 0.81 is only just "ripe" and says so, while
0.95 is unambiguous.

## Optional: train the model from the Kaggle dataset

The app is fully functional on colour analysis alone. Adding the trained model
brings the dataset's own ripe/unripe judgement into the decision.

The dataset is annotated for YOLO object detection, so
`tools/train_ripeness_model.py` crops every annotated tomato out of its image,
builds a classification dataset from those crops, fine-tunes MobileNetV2, and
exports TFLite:

```bash
pip install -r tools/requirements.txt

# Unzip the Kaggle dataset first, then point at its root:
python tools/train_ripeness_model.py --dataset ~/Downloads/tomato-dataset

# Inspect the crops before committing to a training run:
python tools/train_ripeness_model.py --dataset ~/Downloads/tomato-dataset --crops-only

cp tools/out/model.tflite tools/out/labels.txt android/app/src/main/assets/
```

Rebuild, and the info screen will report "Trained model plus colour analysis".

The script finds images and labels across the layouts these datasets ship in
(labels beside images, or parallel `images/` and `labels/` directories) and
reads class names from `data.yaml`. If there is no `data.yaml` it stops and asks
for `--class-names` rather than guessing an order that could silently swap ripe
and unripe.

With the model loaded, it decides ripe vs unripe — that is what the dataset
labels — and colour analysis refines that into *how many days are left*.
Confidence is reduced whenever the two methods disagree.

## Project layout

```
android/                                  Android Studio project
  app/src/main/java/com/example/tomatoripeness/
    MainActivity.kt                       camera, UI, speech triggers
    InfoActivity.kt                       explanation screen
    ripeness/
      ColourRipenessAnalyzer.kt           HSV sampling -> ripeness score
      RipenessEngine.kt                   score/model -> stage + confidence
      TomatoModelClassifier.kt            optional TFLite classifier
      ResultSmoother.kt                   majority vote across frames
      RgbaFrame.kt                        camera frame access, crops
      RipenessStage.kt, RipenessResult.kt
    speech/RipenessSpeaker.kt             text-to-speech with repeat guard
  app/src/main/assets/                    drop model.tflite + labels.txt here
tools/train_ripeness_model.py             dataset -> model.tflite
```

## Limits worth knowing

Colour is a good ripeness signal, but it only describes the outside of the fruit.

- Coloured lighting shifts hues; daylight or neutral white is most reliable.
- Yellow, orange, green and striped varieties never turn red, so they read as
  unripe for their whole life.
- **Ripeness is not food safety.** The app cannot see mould, bruising, splits,
  or spoilage inside the fruit. Always look at and smell a tomato before eating
  it. Treat the verdict as a guide, not a guarantee.
- Only the focus box is inspected, and only one tomato at a time.

## Dataset credit

Ripeness classes follow the **Riped and Unriped Tomato Dataset** by
[sumn2u](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset)
on Kaggle, along with its companion
[YOLOv8 detection notebook](https://www.kaggle.com/code/sumn2u/yolo-v8-ripe-and-unripe-tomatoes-detection).
