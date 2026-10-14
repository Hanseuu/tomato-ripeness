package com.example.tomatoripeness

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.tomatoripeness.databinding.ActivityMainBinding
import com.example.tomatoripeness.ripeness.RgbaFrame
import com.example.tomatoripeness.ripeness.RipenessEngine
import com.example.tomatoripeness.ripeness.RipenessResult
import com.example.tomatoripeness.ripeness.RipenessStage
import com.example.tomatoripeness.ripeness.ResultSmoother
import com.example.tomatoripeness.ripeness.SmoothedResult
import com.example.tomatoripeness.speech.RipenessSpeaker
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Live camera screen: shows the ripeness verdict, speaks it, and pops it up on demand.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var speaker: RipenessSpeaker
    private lateinit var engine: RipenessEngine
    private lateinit var cameraExecutor: ExecutorService

    private val smoother = ResultSmoother()

    /** Most recent smoothed verdict, read by the "check" button. */
    @Volatile
    private var latest: SmoothedResult? = null

    private var announcedStage: RipenessStage? = null
    private var lastAnalysisAt = 0L
    private var cameraStarted = false

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showPermissionPanel(denied = true)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        speaker = RipenessSpeaker(this)
        engine = RipenessEngine.create(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.infoButton.setOnClickListener {
            startActivity(
                Intent(this, InfoActivity::class.java)
                    .putExtra(InfoActivity.EXTRA_MODEL_LOADED, engine.modelLoaded)
            )
        }
        binding.voiceButton.setOnClickListener { toggleVoice() }
        binding.checkButton.setOnClickListener { announceNow() }
        binding.permissionButton.setOnClickListener {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }

        updateVoiceButton()
        renderIdle()

        if (!hasCameraPermission()) {
            showPermissionPanel(denied = false)
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        // Also covers returning from Settings after granting permission by hand.
        if (hasCameraPermission()) startCamera()
    }

    override fun onPause() {
        super.onPause()
        speaker.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        speaker.shutdown()
        engine.close()
    }

    // --- Camera ---------------------------------------------------------------------

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        binding.permissionPanel.visibility = View.GONE
        if (cameraStarted) return
        cameraStarted = true

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                bindUseCases(providerFuture.get())
            } catch (t: Throwable) {
                cameraStarted = false
                Log.e(TAG, "Could not start the camera", t)
                Toast.makeText(this, R.string.camera_error, Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases(provider: ProcessCameraProvider) {
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(binding.previewView.surfaceProvider)
        }

        // A small frame is plenty: ripeness comes from colour statistics, not detail,
        // and a 640x480 frame keeps the per-frame analysis cheap.
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { useCase ->
                useCase.setAnalyzer(cameraExecutor) { image -> analyseFrame(image) }
            }

        provider.unbindAll()
        provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }

    /** Runs on [cameraExecutor], never the main thread. */
    private fun analyseFrame(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()
            if (now - lastAnalysisAt < ANALYSIS_INTERVAL_MS) return
            lastAnalysisAt = now

            val frame = RgbaFrame.from(image) ?: return
            val smoothed = smoother.add(engine.analyse(frame))
            latest = smoothed
            runOnUiThread { render(smoothed) }
        } catch (t: Throwable) {
            Log.e(TAG, "Frame analysis failed", t)
        } finally {
            // Must always close, or the camera stops delivering frames.
            image.close()
        }
    }

    // --- UI -------------------------------------------------------------------------

    private fun renderIdle() {
        binding.stageTitle.text = getString(R.string.scanning)
        binding.stageTitle.setTextColor(ContextCompat.getColor(this, R.color.stage_none))
        binding.verdictText.text = ""
        binding.adviceText.text = getString(R.string.hint_aim)
        binding.detailText.text = ""
        binding.confidenceBar.progress = 0
        binding.checkButton.isEnabled = false
    }

    private fun render(smoothed: SmoothedResult) {
        val result = smoothed.result
        val stage = result.stage
        val colour = ContextCompat.getColor(this, stage.colorRes)

        binding.stageTitle.text = getString(stage.titleRes)
        binding.stageTitle.setTextColor(colour)
        binding.verdictText.text = getString(stage.verdictRes)
        binding.adviceText.text = getString(stage.adviceRes)
        binding.checkButton.isEnabled = stage.isTomato

        if (stage.isTomato) {
            binding.confidenceBar.setIndicatorColor(colour)
            binding.confidenceBar.progress = result.confidencePercent
            binding.detailText.text = detailFor(result)
        } else {
            binding.confidenceBar.progress = 0
            binding.detailText.text = ""
        }

        // Speak once per settled verdict, not once per frame.
        if (!stage.isTomato) {
            announcedStage = null
        } else if (smoothed.isStable && stage != announcedStage) {
            announcedStage = stage
            speaker.speak(getString(stage.verdictRes))
        }
    }

    private fun detailFor(result: RipenessResult): String {
        val confidence = getString(R.string.confidence_label, result.confidencePercent)
        val detail = result.modelLabel?.let { label ->
            getString(
                R.string.detail_with_model,
                result.redPercent, result.greenPercent, result.ripenessScore, label
            )
        } ?: getString(
            R.string.detail_readout,
            result.redPercent, result.greenPercent, result.ripenessScore
        )
        return "$confidence · $detail"
    }

    /** "Check this tomato": force-speak the verdict and show it as a popup. */
    private fun announceNow() {
        val result = latest?.result ?: return
        val stage = result.stage

        speaker.forgetLastSpoken()
        speaker.speak(getString(stage.verdictRes), force = true)

        MaterialAlertDialogBuilder(this)
            .setTitle(getString(stage.titleRes))
            .setMessage(
                buildString {
                    append(getString(stage.verdictRes))
                    append("\n\n")
                    append(getString(stage.adviceRes))
                    if (stage.isTomato) {
                        append("\n\n")
                        append(detailFor(result))
                    }
                }
            )
            .setPositiveButton(R.string.dialog_close, null)
            .show()
    }

    private fun toggleVoice() {
        speaker.enabled = !speaker.enabled
        if (!speaker.enabled) speaker.stop()
        updateVoiceButton()
        Toast.makeText(
            this,
            if (speaker.enabled) R.string.voice_enabled else R.string.voice_disabled,
            Toast.LENGTH_SHORT
        ).show()
    }

    private fun updateVoiceButton() {
        binding.voiceButton.setImageResource(
            if (speaker.enabled) R.drawable.ic_volume_on else R.drawable.ic_volume_off
        )
        binding.voiceButton.contentDescription =
            getString(if (speaker.enabled) R.string.action_voice_on else R.string.action_voice_off)
    }

    private fun showPermissionPanel(denied: Boolean) {
        binding.permissionPanel.visibility = View.VISIBLE
        binding.permissionMessage.setText(
            if (denied) R.string.permission_denied else R.string.permission_rationale
        )
    }

    private companion object {
        const val TAG = "MainActivity"

        /** ~6 analyses per second: responsive without draining the battery. */
        const val ANALYSIS_INTERVAL_MS = 150L
    }
}
