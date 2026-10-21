package com.example.tomatoripeness.ripeness

import androidx.annotation.ColorRes
import androidx.annotation.StringRes
import com.example.tomatoripeness.R

/**
 * The ripeness classes the app can report.
 *
 * The Kaggle dataset this project is built around only labels two classes
 * ("riped tomato" / "unriped tomato"), so the intermediate stages below are
 * derived from how much red has developed on the skin. That is what lets the
 * app answer "how many more days?" instead of only "ripe / not ripe".
 */
enum class RipenessStage(
    @StringRes val titleRes: Int,
    @StringRes val verdictRes: Int,
    @StringRes val adviceRes: Int,
    @ColorRes val colorRes: Int,
    val safeToEat: Boolean
) {
    NO_TOMATO(
        titleRes = R.string.stage_none_title,
        verdictRes = R.string.stage_none_verdict,
        adviceRes = R.string.stage_none_advice,
        colorRes = R.color.stage_none,
        safeToEat = false
    ),

    UNRIPE(
        titleRes = R.string.stage_unripe_title,
        verdictRes = R.string.stage_unripe_verdict,
        adviceRes = R.string.stage_unripe_advice,
        colorRes = R.color.stage_unripe,
        safeToEat = false
    ),

    BREAKER(
        titleRes = R.string.stage_breaker_title,
        verdictRes = R.string.stage_breaker_verdict,
        adviceRes = R.string.stage_breaker_advice,
        colorRes = R.color.stage_breaker,
        safeToEat = false
    ),

    TURNING(
        titleRes = R.string.stage_turning_title,
        verdictRes = R.string.stage_turning_verdict,
        adviceRes = R.string.stage_turning_advice,
        colorRes = R.color.stage_turning,
        safeToEat = false
    ),

    RIPE(
        titleRes = R.string.stage_ripe_title,
        verdictRes = R.string.stage_ripe_verdict,
        adviceRes = R.string.stage_ripe_advice,
        colorRes = R.color.stage_ripe,
        safeToEat = true
    ),

    OVERRIPE(
        titleRes = R.string.stage_overripe_title,
        verdictRes = R.string.stage_overripe_verdict,
        adviceRes = R.string.stage_overripe_advice,
        colorRes = R.color.stage_overripe,
        safeToEat = true
    );

    val isTomato: Boolean get() = this != NO_TOMATO
}
