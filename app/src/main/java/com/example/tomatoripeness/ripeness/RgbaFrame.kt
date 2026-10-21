package com.example.tomatoripeness.ripeness

import android.graphics.Bitmap
import android.graphics.PixelFormat
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

/** The centred square region of the frame that the app actually inspects. */
data class RoiRect(val left: Int, val top: Int, val side: Int) {
    companion object {
        fun centredSquare(width: Int, height: Int, fraction: Float): RoiRect {
            val side = max(1, (min(width, height) * fraction).toInt())
            return RoiRect(
                left = ((width - side) / 2).coerceAtLeast(0),
                top = ((height - side) / 2).coerceAtLeast(0),
                side = min(side, min(width, height))
            )
        }
    }
}

/**
 * Random-access reader over a CameraX RGBA_8888 frame.
 *
 * Reads the plane with absolute [ByteBuffer.get] indexes so the buffer's own
 * position is never mutated, and honours `rowStride` so padded camera buffers
 * are not read skewed. Only valid while the backing [ImageProxy] is open.
 */
class RgbaFrame(
    private val buffer: ByteBuffer,
    val width: Int,
    val height: Int,
    private val rowStride: Int,
    private val pixelStride: Int
) {
    private fun offset(x: Int, y: Int) = y * rowStride + x * pixelStride

    fun red(x: Int, y: Int): Int = buffer.get(offset(x, y)).toInt() and 0xFF
    fun green(x: Int, y: Int): Int = buffer.get(offset(x, y) + 1).toInt() and 0xFF
    fun blue(x: Int, y: Int): Int = buffer.get(offset(x, y) + 2).toInt() and 0xFF

    /** Opaque ARGB int for one pixel. */
    fun argb(x: Int, y: Int): Int {
        val i = offset(x, y)
        val r = buffer.get(i).toInt() and 0xFF
        val g = buffer.get(i + 1).toInt() and 0xFF
        val b = buffer.get(i + 2).toInt() and 0xFF
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    /**
     * Nearest-neighbour crop of [roi] into a square [size] x [size] bitmap.
     *
     * Built pixel by pixel rather than via `copyPixelsFromBuffer` because the
     * camera plane may be row-padded, which that shortcut does not account for.
     * Only called when a .tflite model is loaded.
     */
    fun cropToBitmap(roi: RoiRect, size: Int): Bitmap {
        val pixels = IntArray(size * size)
        for (dy in 0 until size) {
            val srcY = (roi.top + dy * roi.side / size).coerceIn(0, height - 1)
            val row = dy * size
            for (dx in 0 until size) {
                val srcX = (roi.left + dx * roi.side / size).coerceIn(0, width - 1)
                pixels[row + dx] = argb(srcX, srcY)
            }
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    companion object {
        /** Returns null if the frame is not RGBA_8888, so callers can skip it safely. */
        fun from(image: ImageProxy): RgbaFrame? {
            if (image.format != PixelFormat.RGBA_8888) return null
            val plane = image.planes.firstOrNull() ?: return null
            if (image.width <= 0 || image.height <= 0) return null
            return RgbaFrame(
                buffer = plane.buffer,
                width = image.width,
                height = image.height,
                rowStride = plane.rowStride,
                pixelStride = plane.pixelStride
            )
        }
    }
}
