package com.example.tomatoripeness

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.tomatoripeness.databinding.ActivityInfoBinding

/** Explains how the verdict is reached, the known limits, and the dataset credit. */
class InfoActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding = ActivityInfoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val modelLoaded = intent.getBooleanExtra(EXTRA_MODEL_LOADED, false)
        binding.sourceBody.setText(
            if (modelLoaded) R.string.info_source_model else R.string.info_source_colour
        )
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    companion object {
        const val EXTRA_MODEL_LOADED = "model_loaded"
    }
}
