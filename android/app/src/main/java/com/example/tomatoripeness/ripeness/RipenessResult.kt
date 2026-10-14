package com.example.tomatoripeness.ripeness

/** Where the verdict came from, shown on the info screen so the result is explainable. */
enum class DetectionSource {
    /** No model.tflite in assets: skin-colour analysis only. */
    COLOUR_ONLY,

    /** model.tflite is loaded; its ripe/unripe call is refined by colour. */
    MODEL_AND_COLOUR
}

/**
 * One ripeness verdict for one frame (or for a smoothed run of frames).
 *
 * @param stage        the reported ripeness class
 * @param confidence   0f..1f, how much to trust this verdict
 * @param ripenessScore 0f..1f, how far the skin colour has moved from green to red
 * @param coverage     0f..1f, how much of the focus box looks like tomato skin
 * @param reading      the raw colour breakdown, used for the detail readout
 * @param source       colour-only, or model + colour
 * @param modelLabel   raw winning label from the .tflite model, when one is loaded
 */
data class RipenessResult(
    val stage: RipenessStage,
    val confidence: Float,
    val ripenessScore: Float,
    val coverage: Float,
    val reading: ColourReading,
    val source: DetectionSource,
    val modelLabel: String? = null
) {
    val confidencePercent: Int get() = (confidence * 100f).toInt().coerceIn(0, 100)
    val redPercent: Int get() = (reading.redFamilyFraction * 100f).toInt().coerceIn(0, 100)
    val greenPercent: Int get() = (reading.greenFamilyFraction * 100f).toInt().coerceIn(0, 100)

    companion object {
        fun noTomato(source: DetectionSource, reading: ColourReading) = RipenessResult(
            stage = RipenessStage.NO_TOMATO,
            confidence = 0f,
            ripenessScore = 0f,
            coverage = reading.coverage,
            reading = reading,
            source = source
        )
    }
}
