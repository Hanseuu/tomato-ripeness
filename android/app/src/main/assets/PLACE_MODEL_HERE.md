# Optional trained model goes here

The app runs without a model: `RipenessEngine` falls back to colour analysis and
stays fully functional.

To use the Kaggle dataset model instead, put these two files in this folder:

- `model.tflite` — image classifier, input `[1, N, N, 3]`, RGB normalised to 0..1
- `labels.txt` — one class name per line, in the model's output order

Both are produced by `tools/train_ripeness_model.py`. Label text is matched
case-insensitively by `RipenessEngine.stageFromLabel`, so names like
`riped tomato` / `unriped tomato` are understood as-is.

If either file is missing or fails to load, the app logs the reason under the
`TomatoModel` tag and continues on colour analysis alone.
