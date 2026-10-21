package com.example.tomatoripeness.ripeness

import android.content.Context
import java.io.Closeable
import kotlin.math.abs
import kotlin.math.min

/**
 * Turns a camera frame into a ripeness verdict.
 *
 * Two routes, picked automatically at startup:
 *
 *  - No `assets/model.tflite`: colour analysis decides the stage outright.
 *  - Model present: the model makes the ripe/unripe call (that is what the
 *    Kaggle dataset labels), and colour analysis refines it into "wait a week"
 *    vs "wait a couple of days" and lowers confidence when the two disagree.
 */
class RipenessEngine private constructor(
    private val classifier: TomatoModelClassifier?
) : Closeable {

    val source: DetectionSource =
        if (classifier != null) DetectionSource.MODEL_AND_COLOUR else DetectionSource.COLOUR_ONLY

    val modelLoaded: Boolean get() = classifier != null

    fun analyse(frame: RgbaFrame): RipenessResult {
        val roi = RoiRect.centredSquare(frame.width, frame.height, ColourRipenessAnalyzer.ROI_FRACTION)
        val reading = ColourRipenessAnalyzer.analyse(frame, roi)

        // Not enough tomato-looking skin in the box: say so instead of guessing.
        if (reading.coverage < ColourRipenessAnalyzer.MIN_COVERAGE || reading.skinPixels == 0) {
            return RipenessResult.noTomato(source, reading)
        }

        val colourStage = stageFromColour(reading)
        val colourConfidence = colourConfidence(reading)

        val prediction = classifier?.let { model ->
            model.classify(frame.cropToBitmap(roi, model.inputSize))
        }

        if (prediction == null) {
            return RipenessResult(
                stage = colourStage,
                confidence = colourConfidence,
                ripenessScore = reading.ripenessScore,
                coverage = reading.coverage,
                reading = reading,
                source = source
            )
        }

        val modelStage = stageFromLabel(prediction.label)
        val fusedStage = fuse(modelStage, reading)

        // Disagreement about edibility is the case worth flagging, so damp confidence.
        val conflict = modelStage != null && modelStage.safeToEat != colourStage.safeToEat
        val blended = 0.6f * prediction.confidence + 0.4f * colourConfidence
        val confidence = (if (conflict) blended * 0.8f else blended).coerceIn(0f, 1f)

        return RipenessResult(
            stage = fusedStage,
            confidence = confidence,
            ripenessScore = reading.ripenessScore,
            coverage = reading.coverage,
            reading = reading,
            source = source,
            modelLabel = prediction.label
        )
    }

    /** Model decides ripe/unripe; colour decides "how much longer". */
    private fun fuse(modelStage: RipenessStage?, reading: ColourReading): RipenessStage {
        val score = reading.ripenessScore
        return when (modelStage) {
            null -> stageFromColour(reading)
            RipenessStage.OVERRIPE -> RipenessStage.OVERRIPE
            RipenessStage.RIPE ->
                // Model says ripe but the skin is not red enough to agree: stay cautious.
                if (score >= TURNING_MIN) RipenessStage.RIPE else RipenessStage.TURNING
            RipenessStage.UNRIPE -> when {
                score >= TURNING_MIN -> RipenessStage.TURNING
                score >= BREAKER_MIN -> RipenessStage.BREAKER
                else -> RipenessStage.UNRIPE
            }
            else -> modelStage
        }
    }

    override fun close() {
        classifier?.close()
    }

    companion object {
        // Ripeness-score band edges. A tomato ripens green -> yellow -> orange -> red.
        private const val BREAKER_MIN = 0.35f
        private const val TURNING_MIN = 0.60f
        private const val RIPE_MIN = 0.80f

        // Overripe skin is deep red but dull and dark. Deliberately strict: a ripe
        // tomato photographed in dim light must not be called overripe.
        private const val OVERRIPE_RED_FRACTION = 0.80f
        private const val OVERRIPE_MAX_VALUE = 0.38f
        private const val OVERRIPE_MAX_SATURATION = 0.62f

        /**
         * Only the boundaries *between* stages are ambiguous. A score of 0.00 or
         * 1.00 is the most certain reading there is, so the outer ends of the
         * range are deliberately not treated as edges.
         */
        private val STAGE_BOUNDARIES = floatArrayOf(BREAKER_MIN, TURNING_MIN, RIPE_MIN)

        /** How close to a boundary counts as a coin toss. */
        private const val AMBIGUITY_WIDTH = 0.10f

        fun create(context: Context): RipenessEngine =
            RipenessEngine(TomatoModelClassifier.createOrNull(context))

        /** Visible for the colour-only path and for threshold testing. */
        fun stageFromColour(reading: ColourReading): RipenessStage {
            if (reading.redFamilyFraction >= OVERRIPE_RED_FRACTION &&
                reading.redMeanValue <= OVERRIPE_MAX_VALUE &&
                reading.redMeanSaturation <= OVERRIPE_MAX_SATURATION
            ) {
                return RipenessStage.OVERRIPE
            }
            return when {
                reading.ripenessScore >= RIPE_MIN -> RipenessStage.RIPE
                reading.ripenessScore >= TURNING_MIN -> RipenessStage.TURNING
                reading.ripenessScore >= BREAKER_MIN -> RipenessStage.BREAKER
                else -> RipenessStage.UNRIPE
            }
        }

        /**
         * Confidence from two signals: how much of the focus box is tomato, and how
         * far the score sits from the nearest stage boundary. A score sitting right
         * on a boundary could land either side, so it is reported as less certain.
         */
        fun colourConfidence(reading: ColourReading): Float {
            val coverageFactor = ((reading.coverage - ColourRipenessAnalyzer.MIN_COVERAGE) /
                (0.55f - ColourRipenessAnalyzer.MIN_COVERAGE)).coerceIn(0f, 1f)

            val score = reading.ripenessScore.coerceIn(0f, 1f)
            var nearestBoundary = Float.MAX_VALUE
            for (boundary in STAGE_BOUNDARIES) {
                nearestBoundary = min(nearestBoundary, abs(score - boundary))
            }
            val marginFactor = (nearestBoundary / AMBIGUITY_WIDTH).coerceIn(0f, 1f)

            return (0.40f + 0.35f * marginFactor + 0.25f * coverageFactor).coerceIn(0f, 1f)
        }

        /**
         * Maps a model label onto a stage.
         *
         * Order matters: "unriped" contains "riped", so negatives are tested first.
         */
        fun stageFromLabel(rawLabel: String): RipenessStage? {
            val label = rawLabel.lowercase().trim()
            return when {
                label.contains("over") || label.contains("rotten") ||
                    label.contains("spoil") || label.contains("damaged") -> RipenessStage.OVERRIPE

                label.contains("unripe") || label.contains("unriped") ||
                    label.contains("green") -> RipenessStage.UNRIPE

                label.contains("half") || label.contains("semi") ||
                    label.contains("turning") || label.contains("breaker") -> RipenessStage.TURNING

                label.contains("ripe") || label.contains("riped") ||
                    label.contains("red") -> RipenessStage.RIPE

                else -> null
            }
        }
    }
}
