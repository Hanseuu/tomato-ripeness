package com.example.tomatoripeness.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Speaks the verdict out loud.
 *
 * Guards against the two ways this gets annoying: it refuses to repeat the same
 * sentence inside [REPEAT_COOLDOWN_MS], and it silently no-ops when the device
 * has no usable TTS engine instead of crashing.
 */
class RipenessSpeaker(context: Context) {

    var enabled: Boolean = true

    private var ready = false
    private var lastSpoken: String? = null
    private var lastSpokenAt = 0L

    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
        ready = status == TextToSpeech.SUCCESS
        if (!ready) {
            Log.w(TAG, "Text-to-speech unavailable (status=$status); verdicts stay on screen only.")
        }
    }.also { engine ->
        val preferred = Locale.getDefault()
        val available = engine.isLanguageAvailable(preferred)
        engine.language = if (
            available == TextToSpeech.LANG_AVAILABLE ||
            available == TextToSpeech.LANG_COUNTRY_AVAILABLE ||
            available == TextToSpeech.LANG_COUNTRY_VAR_AVAILABLE
        ) preferred else Locale.US
    }

    /** @param force speak even if it repeats the previous sentence (manual check button). */
    fun speak(text: String, force: Boolean = false) {
        if (!enabled || !ready || text.isBlank()) return

        val now = System.currentTimeMillis()
        if (!force && text == lastSpoken && now - lastSpokenAt < REPEAT_COOLDOWN_MS) return

        lastSpoken = text
        lastSpokenAt = now
        runCatching {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
        }.onFailure { Log.e(TAG, "speak() failed", it) }
    }

    fun stop() {
        runCatching { tts.stop() }
    }

    /** Clears the repeat guard so the next verdict is always spoken. */
    fun forgetLastSpoken() {
        lastSpoken = null
    }

    fun shutdown() {
        runCatching {
            tts.stop()
            tts.shutdown()
        }
    }

    private companion object {
        const val TAG = "RipenessSpeaker"
        const val UTTERANCE_ID = "ripeness-verdict"
        const val REPEAT_COOLDOWN_MS = 6_000L
    }
}
