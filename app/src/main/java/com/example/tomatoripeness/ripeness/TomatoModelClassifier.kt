package com.example.tomatoripeness.ripeness

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One prediction from the trained model. */
data class ModelPrediction(val label: String, val confidence: Float)

/**
 * Optional TensorFlow Lite classifier driven by `assets/model.tflite`.
 *
 * Entirely optional: [createOrNull] returns null when the model or its
 * labels file is absent, and the app then runs on colour analysis alone.
 * Train the model with `tools/train_ripeness_model.py`, which writes a model
 * expecting 0..1 normalised RGB input - matching [classify] below.
 */
class TomatoModelClassifier private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    val inputSize: Int,
    private val quantisedInput: Boolean,
    private val quantisedOutput: Boolean,
    private val classCount: Int
) : Closeable {

    private val pixels = IntArray(inputSize * inputSize)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(inputSize * inputSize * 3 * if (quantisedInput) 1 else 4)
        .order(ByteOrder.nativeOrder())

    /** Runs the model over a square [inputSize] crop. Returns null on failure. */
    fun classify(bitmap: Bitmap): ModelPrediction? {
        return try {
            bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

            inputBuffer.rewind()
            for (pixel in pixels) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                if (quantisedInput) {
                    inputBuffer.put(r.toByte())
                    inputBuffer.put(g.toByte())
                    inputBuffer.put(b.toByte())
                } else {
                    inputBuffer.putFloat(r / 255f)
                    inputBuffer.putFloat(g / 255f)
                    inputBuffer.putFloat(b / 255f)
                }
            }
            inputBuffer.rewind()

            val scores = FloatArray(classCount)
            if (quantisedOutput) {
                val raw = Array(1) { ByteArray(classCount) }
                interpreter.run(inputBuffer, raw)
                for (i in 0 until classCount) {
                    scores[i] = (raw[0][i].toInt() and 0xFF) / 255f
                }
            } else {
                val raw = Array(1) { FloatArray(classCount) }
                interpreter.run(inputBuffer, raw)
                raw[0].copyInto(scores)
            }

            var best = 0
            for (i in 1 until classCount) if (scores[i] > scores[best]) best = i

            ModelPrediction(
                label = labels.getOrElse(best) { "class_$best" },
                confidence = scores[best].coerceIn(0f, 1f)
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Model inference failed", t)
            null
        }
    }

    override fun close() {
        runCatching { interpreter.close() }
    }

    companion object {
        private const val TAG = "TomatoModel"
        private const val MODEL_ASSET = "model.tflite"
        private const val LABELS_ASSET = "labels.txt"

        fun createOrNull(context: Context): TomatoModelClassifier? {
            return try {
                val assets = context.assets.list("")?.toSet() ?: emptySet()
                if (MODEL_ASSET !in assets) {
                    Log.i(TAG, "No $MODEL_ASSET in assets - using colour analysis only.")
                    return null
                }
                if (LABELS_ASSET !in assets) {
                    Log.w(TAG, "$MODEL_ASSET found but $LABELS_ASSET is missing - skipping model.")
                    return null
                }

                val labels = context.assets.open(LABELS_ASSET).bufferedReader().use { reader ->
                    reader.readLines().map { it.trim() }.filter { it.isNotEmpty() }
                }
                if (labels.isEmpty()) {
                    Log.w(TAG, "$LABELS_ASSET is empty - skipping model.")
                    return null
                }

                // Read into a direct buffer rather than memory-mapping, so the model
                // loads whether or not the APK stored the asset compressed.
                val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
                val modelBuffer = ByteBuffer.allocateDirect(bytes.size)
                    .order(ByteOrder.nativeOrder())
                    .put(bytes)
                modelBuffer.rewind()

                val interpreter = Interpreter(modelBuffer, Interpreter.Options())

                val inputTensor = interpreter.getInputTensor(0)
                val shape = inputTensor.shape() // expected [1, size, size, 3]
                val inputSize = if (shape.size >= 3) shape[1] else 0
                if (inputSize <= 0) {
                    Log.w(TAG, "Unexpected input shape ${shape.joinToString()} - skipping model.")
                    interpreter.close()
                    return null
                }

                val outputShape = interpreter.getOutputTensor(0).shape()
                val classCount = outputShape.last()
                if (classCount <= 0) {
                    Log.w(TAG, "Unexpected output shape ${outputShape.joinToString()} - skipping.")
                    interpreter.close()
                    return null
                }
                if (classCount != labels.size) {
                    Log.w(
                        TAG,
                        "Model outputs $classCount classes but $LABELS_ASSET has " +
                            "${labels.size} lines. Labels may be mismatched."
                    )
                }

                Log.i(TAG, "Loaded model: input ${inputSize}x$inputSize, $classCount classes $labels")
                TomatoModelClassifier(
                    interpreter = interpreter,
                    labels = labels,
                    inputSize = inputSize,
                    quantisedInput = inputTensor.dataType() == DataType.UINT8,
                    quantisedOutput = interpreter.getOutputTensor(0).dataType() == DataType.UINT8,
                    classCount = classCount
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Could not load $MODEL_ASSET - falling back to colour analysis.", t)
                null
            }
        }
    }
}
