#!/usr/bin/env python3
"""
Train a tomato ripeness classifier from the Kaggle "Riped and Unriped Tomato
Dataset" and export it as model.tflite for the Android app.

The dataset is annotated for YOLO object detection: each image has a .txt file
of bounding boxes. This script crops every annotated tomato out of its image to
build a classification dataset, fine-tunes MobileNetV2 on it, and writes a
TFLite model plus the matching labels.txt.

Usage
-----
    pip install -r tools/requirements.txt

    # 1. Download and unzip the dataset from Kaggle, then:
    python tools/train_ripeness_model.py --dataset /path/to/unzipped/dataset

    # 2. Copy the two generated files into the app:
    cp tools/out/model.tflite tools/out/labels.txt \
       android/app/src/main/assets/

The exported model takes RGB input normalised to 0..1, which is exactly what
TomatoModelClassifier.classify() feeds it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fraction of the box size added around each crop, so the model sees a little
# context and is not hurt by slightly tight annotations.
BOX_PADDING = 0.08

# Crops smaller than this (before resizing) are too blurry to learn from.
MIN_BOX_PIXELS = 32


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #

def find_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def label_path_for(image: Path) -> Path | None:
    """
    Locate the YOLO .txt for an image.

    Handles both common layouts: the label sitting next to the image, and the
    `images/` -> `labels/` parallel directory convention.
    """
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling

    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            candidate_parts = parts[:i] + ["labels"] + parts[i + 1:]
            candidate = Path(*candidate_parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
    return None


def parse_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parses `class cx cy w h` lines (normalised 0..1). Bad lines are skipped."""
    boxes = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def read_class_names(root: Path) -> list[str] | None:
    """Reads class names from a YOLO data.yaml, if the dataset ships one."""
    try:
        import yaml
    except ImportError:
        return None

    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        for candidate in root.rglob(name):
            try:
                content = yaml.safe_load(candidate.read_text(errors="ignore"))
            except Exception:
                continue
            if not isinstance(content, dict) or "names" not in content:
                continue
            names = content["names"]
            if isinstance(names, list) and names:
                print(f"  class names from {candidate.name}: {names}")
                return [str(n) for n in names]
            if isinstance(names, dict) and names:
                ordered = [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
                print(f"  class names from {candidate.name}: {ordered}")
                return ordered
    return None


# --------------------------------------------------------------------------- #
# Crop extraction
# --------------------------------------------------------------------------- #

def build_crops(dataset: Path, out_dir: Path, class_names: list[str] | None,
                image_size: int) -> tuple[Path, list[str]]:
    from PIL import Image

    images = find_images(dataset)
    if not images:
        sys.exit(f"No images found under {dataset}")
    print(f"Found {len(images)} images under {dataset}")

    labelled = [(img, lbl) for img in images if (lbl := label_path_for(img))]
    print(f"  {len(labelled)} of them have YOLO label files")

    crops_dir = out_dir / "crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    if labelled:
        names = class_names or read_class_names(dataset)
        ids_present = sorted({
            box[0] for _, lbl in labelled for box in parse_yolo_labels(lbl)
        })
        if not ids_present:
            sys.exit("Label files exist but contain no usable boxes.")

        if not names:
            sys.exit(
                "Could not determine class names.\n"
                f"  Class ids present in the labels: {ids_present}\n"
                "  No data.yaml was found, so the order of the classes is unknown.\n"
                "  Re-run with the names in class-id order, for example:\n"
                "    --class-names 'riped tomato,unriped tomato'\n"
                "  Guessing here would silently invert ripe and unripe."
            )
        if max(ids_present) >= len(names):
            sys.exit(
                f"Labels use class id {max(ids_present)} but only "
                f"{len(names)} class names were given: {names}"
            )

        counts = Counter()
        skipped = 0
        for image_path, label_path in labelled:
            boxes = parse_yolo_labels(label_path)
            if not boxes:
                continue
            try:
                with Image.open(image_path) as handle:
                    image = handle.convert("RGB")
                    width, height = image.size

                    for index, (class_id, cx, cy, bw, bh) in enumerate(boxes):
                        pad_w = bw * BOX_PADDING
                        pad_h = bh * BOX_PADDING
                        left = int(max(0.0, (cx - bw / 2 - pad_w)) * width)
                        top = int(max(0.0, (cy - bh / 2 - pad_h)) * height)
                        right = int(min(1.0, (cx + bw / 2 + pad_w)) * width)
                        bottom = int(min(1.0, (cy + bh / 2 + pad_h)) * height)

                        if right - left < MIN_BOX_PIXELS or bottom - top < MIN_BOX_PIXELS:
                            skipped += 1
                            continue

                        label = names[class_id]
                        target = crops_dir / sanitise(label)
                        target.mkdir(parents=True, exist_ok=True)
                        crop = image.crop((left, top, right, bottom)).resize(
                            (image_size, image_size), Image.BILINEAR
                        )
                        crop.save(target / crop_name(image_path, index), quality=92)
                        counts[label] += 1
            except Exception as error:  # unreadable image, truncated file, ...
                print(f"  skipping {image_path.name}: {error}")

        print(f"\nCropped {sum(counts.values())} tomatoes ({skipped} boxes too small):")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")

    else:
        # No detection labels: fall back to a directory-per-class layout.
        print("  no label files found - treating parent directory names as classes")
        counts = Counter()
        for image_path in images:
            label = image_path.parent.name
            target = crops_dir / sanitise(label)
            target.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(image_path) as handle:
                    handle.convert("RGB").resize(
                        (image_size, image_size), Image.BILINEAR
                    ).save(target / crop_name(image_path, 0), quality=92)
                counts[label] += 1
            except Exception as error:
                print(f"  skipping {image_path.name}: {error}")
        print(f"\nPrepared {sum(counts.values())} images:")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")

    present = sorted(d.name for d in crops_dir.iterdir() if d.is_dir())
    if len(present) < 2:
        sys.exit(
            f"Need at least 2 classes to train, found {present}. "
            "Check that the dataset path points at the unzipped dataset root."
        )
    return crops_dir, present


def sanitise(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()).strip("_")


def crop_name(image_path: Path, box_index: int) -> str:
    """
    Unique filename for one crop.

    The source path is hashed in because datasets split across train/ and valid/
    routinely reuse image names, and a plain stem would let those crops silently
    overwrite each other and shrink the training set.
    """
    tag = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    return f"{image_path.stem}_{tag}_{box_index}.jpg"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(crops_dir: Path, out_dir: Path, image_size: int, batch_size: int,
          head_epochs: int, finetune_epochs: int, seed: int) -> None:
    import tensorflow as tf

    print(f"\nTensorFlow {tf.__version__}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir, validation_split=0.15, subset="training", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir, validation_split=0.15, subset="validation", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size,
        label_mode="categorical",
    )

    # This order is the model's output order, so labels.txt must match it exactly.
    class_names = list(train_ds.class_names)
    print(f"\nClasses (output order): {class_names}")

    # Class weights stop a dominant class from swamping the rarer one.
    counts = Counter()
    for class_index, name in enumerate(class_names):
        counts[class_index] = len(list((crops_dir / name).glob("*.jpg")))
    total = sum(counts.values())
    class_weight = {
        i: total / (len(class_names) * max(1, counts[i])) for i in range(len(class_names))
    }
    print(f"Class weights: { {k: round(v, 3) for k, v in class_weight.items()} }")

    # Augmentation lives in the data pipeline, not the model, so the exported
    # graph stays clean and inference-only.
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.20),
    ], name="augment")

    autotune = tf.data.AUTOTUNE
    # Scale to 0..1 here, matching what the Android app feeds the model.
    train_ds = (train_ds
                .map(lambda x, y: (augment(x, training=True) / 255.0, y),
                     num_parallel_calls=autotune)
                .prefetch(autotune))
    val_ds = (val_ds
              .map(lambda x, y: (x / 255.0, y), num_parallel_calls=autotune)
              .prefetch(autotune))

    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    # Input arrives as 0..1; MobileNetV2's pretrained weights expect -1..1.
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="tomato_ripeness")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n--- Stage 1: training the head ({head_epochs} epochs) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
              class_weight=class_weight)

    if finetune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"\n--- Stage 2: fine-tuning the top of MobileNetV2 "
              f"({finetune_epochs} epochs) ---")
        model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                  class_weight=class_weight)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {accuracy:.4f} (loss {loss:.4f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_bytes = to_tflite(model, out_dir)

    model_path = out_dir / "model.tflite"
    model_path.write_bytes(tflite_bytes)

    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(class_names) + "\n")

    (out_dir / "training_report.json").write_text(json.dumps({
        "classes": class_names,
        "validation_accuracy": round(float(accuracy), 4),
        "validation_loss": round(float(loss), 4),
        "image_size": image_size,
        "input_range": "0..1 float RGB",
        "crops_per_class": {name: counts[i] for i, name in enumerate(class_names)},
    }, indent=2) + "\n")

    print(f"\nWrote {model_path} ({model_path.stat().st_size / 1_000_000:.2f} MB)")
    print(f"Wrote {labels_path}: {class_names}")
    print("\nCopy both into android/app/src/main/assets/ and rebuild the app.")


def to_tflite(model, out_dir: Path) -> bytes:
    """
    Converts to TFLite with float16 weights (float32 in/out, so the Android
    side keeps feeding plain floats).

    Tries the direct Keras path first and falls back to exporting a SavedModel,
    because which one works depends on the installed Keras major version.
    """
    import tensorflow as tf

    def configure(converter):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter

    try:
        return configure(tf.lite.TFLiteConverter.from_keras_model(model)).convert()
    except Exception as error:
        print(f"\nDirect Keras conversion failed ({error});"
              " retrying via SavedModel export.")
        saved = out_dir / "saved_model"
        if saved.exists():
            shutil.rmtree(saved)
        if hasattr(model, "export"):
            model.export(str(saved))          # Keras 3
        else:
            model.save(str(saved))            # Keras 2
        return configure(tf.lite.TFLiteConverter.from_saved_model(str(saved))).convert()


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tomato ripeness classifier and export model.tflite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Root of the unzipped Kaggle dataset")
    parser.add_argument("--out", type=Path, default=Path("tools/out"),
                        help="Where to write model.tflite and labels.txt")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Comma-separated class names in class-id order, "
                             "used when the dataset has no data.yaml")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--crops-only", action="store_true",
                        help="Build and inspect the crops without training")
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        sys.exit(f"Not a directory: {dataset}")

    class_names = None
    if args.class_names:
        class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_dir, present = build_crops(dataset, out_dir, class_names, args.image_size)
    print(f"\nCrops written to {crops_dir} ({len(present)} classes)")

    if args.crops_only:
        print("--crops-only set: stopping before training.")
        return

    train(crops_dir, out_dir, args.image_size, args.batch_size,
          args.head_epochs, args.finetune_epochs, args.seed)


if __name__ == "__main__":
    main()
