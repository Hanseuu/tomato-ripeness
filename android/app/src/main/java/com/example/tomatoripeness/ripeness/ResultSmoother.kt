package com.example.tomatoripeness.ripeness

/** A verdict averaged over recent frames, plus whether it has settled. */
data class SmoothedResult(val result: RipenessResult, val isStable: Boolean)

/**
 * Majority-votes the stage across a sliding window of frames.
 *
 * Per-frame verdicts flicker as the camera moves and auto-exposure hunts, which
 * looks broken and would make the app speak over itself. Voting on the stage
 * (rather than averaging the score) keeps this correct for the model path too,
 * where the stage is not a pure function of the colour score.
 */
class ResultSmoother(private val windowSize: Int = 7) {

    private val window = ArrayDeque<RipenessResult>()

    fun add(result: RipenessResult): SmoothedResult {
        window.addLast(result)
        while (window.size > windowSize) window.removeFirst()

        val counts = HashMap<RipenessStage, Int>()
        for (entry in window) counts[entry.stage] = (counts[entry.stage] ?: 0) + 1

        val winner = counts.maxByOrNull { it.value }?.key ?: result.stage
        val winnerCount = counts[winner] ?: 0
        val matching = window.filter { it.stage == winner }

        // Average the numbers over agreeing frames only, so a couple of stray
        // frames cannot drag the reported confidence around.
        val averaged = matching.last().copy(
            stage = winner,
            confidence = matching.map { it.confidence }.average().toFloat(),
            ripenessScore = matching.map { it.ripenessScore }.average().toFloat(),
            coverage = matching.map { it.coverage }.average().toFloat()
        )

        // Settled once the window is full and the winner holds a clear majority.
        val isStable = window.size >= windowSize && winnerCount * 10 >= windowSize * 6

        return SmoothedResult(averaged, isStable)
    }

    fun reset() = window.clear()
}
