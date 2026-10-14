package com.example.tomatoripeness.ripeness

import kotlin.math.max

/** Colour breakdown of the tomato skin found inside the focus box. */
data class ColourReading(
    /** Fraction of sampled pixels that look like tomato skin at all (0f..1f). */
    val coverage: Float,
    /** Green-to-red progress of that skin, 0f = fully green, 1f = fully red. */
    val ripenessScore: Float,
    val redFamilyFraction: Float,
    val orangeFraction: Float,
    val yellowFraction: Float,
    val greenFamilyFraction: Float,
    /** Mean brightness/saturation of the red-ish pixels, used to spot dull overripe skin. */
    val redMeanValue: Float,
    val redMeanSaturation: Float,
    val sampledPixels: Int,
    val skinPixels: Int
) {
    companion object {
        val EMPTY = ColourReading(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0, 0)
    }
}

/**
 * Decides how ripe a tomato is from the hue distribution of its skin.
 *
 * A tomato ripens green -> yellow -> orange -> red, so the average hue of the
 * skin is a direct proxy for ripeness. Each sampled pixel is converted to HSV,
 * clearly non-skin pixels (grey walls, shadows, blown-out highlights, blue/violet
 * background) are discarded, and the survivors are bucketed by hue. Each bucket
 * carries a ripeness weight and the weighted mean becomes [ColourReading.ripenessScore].
 *
 * Pure arithmetic on purpose: no Android types, so the thresholds can be
 * reasoned about and tested in isolation.
 */
object ColourRipenessAnalyzer {

    /** Side of the inspected square as a fraction of the frame's short edge. */
    const val ROI_FRACTION = 0.62f

    /** Target samples per axis inside the focus box (~12k pixels inspected per frame). */
    private const val SAMPLES_PER_AXIS = 110

    // --- Pixel rejection thresholds -------------------------------------------------
    private const val MIN_SATURATION = 0.30f   // greys: walls, tables, paper
    private const val MIN_VALUE = 0.12f        // near-black: shadow
    private const val HIGHLIGHT_VALUE = 0.97f  // blown-out specular highlight
    private const val HIGHLIGHT_SATURATION = 0.45f

    /** Below this skin coverage we refuse to guess and report NO_TOMATO. */
    const val MIN_COVERAGE = 0.18f

    // --- Hue buckets (degrees) and their ripeness weights ---------------------------
    private const val W_RED = 1.00f
    private const val W_DEEP_RED = 0.95f
    private const val W_ORANGE_RED = 0.85f
    private const val W_ORANGE = 0.62f
    private const val W_YELLOW = 0.42f
    private const val W_YELLOW_GREEN = 0.22f
    private const val W_GREEN = 0.00f

    fun analyse(frame: RgbaFrame, roi: RoiRect): ColourReading {
        val step = max(1, roi.side / SAMPLES_PER_AXIS)

        var sampled = 0
        var skin = 0

        var red = 0
        var deepRed = 0
        var orangeRed = 0
        var orange = 0
        var yellow = 0
        var yellowGreen = 0
        var green = 0

        var weightSum = 0f
        var redValueSum = 0f
        var redSaturationSum = 0f
        var redCount = 0

        val endY = (roi.top + roi.side).coerceAtMost(frame.height)
        val endX = (roi.left + roi.side).coerceAtMost(frame.width)

        var y = roi.top
        while (y < endY) {
            var x = roi.left
            while (x < endX) {
                sampled++

                val r = frame.red(x, y)
                val g = frame.green(x, y)
                val b = frame.blue(x, y)

                val maxC = max(r, max(g, b))
                val minC = minOf(r, g, b)
                val delta = maxC - minC

                val value = maxC / 255f
                val saturation = if (maxC == 0) 0f else delta / maxC.toFloat()

                if (saturation < MIN_SATURATION || value < MIN_VALUE) { x += step; continue }
                if (value >= HIGHLIGHT_VALUE && saturation < HIGHLIGHT_SATURATION) { x += step; continue }
                if (delta == 0) { x += step; continue }

                val hue = hueDegrees(r, g, b, maxC, delta)

                // Blue through violet cannot be tomato skin: treat as background.
                if (hue > 170f && hue < 320f) { x += step; continue }

                skin++
                when {
                    hue >= 345f || hue <= 12f -> { red++; weightSum += W_RED }
                    hue >= 320f -> { deepRed++; weightSum += W_DEEP_RED }
                    hue <= 28f -> { orangeRed++; weightSum += W_ORANGE_RED }
                    hue <= 45f -> { orange++; weightSum += W_ORANGE }
                    hue <= 68f -> { yellow++; weightSum += W_YELLOW }
                    hue <= 90f -> { yellowGreen++; weightSum += W_YELLOW_GREEN }
                    else -> { green++; weightSum += W_GREEN }
                }

                // Track brightness of the red family only - dull dark red means overripe.
                if (hue >= 320f || hue <= 28f) {
                    redValueSum += value
                    redSaturationSum += saturation
                    redCount++
                }

                x += step
            }
            y += step
        }

        if (sampled == 0 || skin == 0) return ColourReading.EMPTY

        val skinF = skin.toFloat()
        return ColourReading(
            coverage = skin / sampled.toFloat(),
            ripenessScore = (weightSum / skinF).coerceIn(0f, 1f),
            redFamilyFraction = (red + deepRed + orangeRed) / skinF,
            orangeFraction = orange / skinF,
            yellowFraction = yellow / skinF,
            greenFamilyFraction = (green + yellowGreen) / skinF,
            redMeanValue = if (redCount == 0) 0f else redValueSum / redCount,
            redMeanSaturation = if (redCount == 0) 0f else redSaturationSum / redCount,
            sampledPixels = sampled,
            skinPixels = skin
        )
    }

    /** Standard RGB -> HSV hue, in degrees 0f..360f. */
    private fun hueDegrees(r: Int, g: Int, b: Int, maxC: Int, delta: Int): Float {
        val d = delta.toFloat()
        val hue = when (maxC) {
            r -> 60f * (((g - b) / d) % 6f)
            g -> 60f * (((b - r) / d) + 2f)
            else -> 60f * (((r - g) / d) + 4f)
        }
        return if (hue < 0f) hue + 360f else hue
    }
}
