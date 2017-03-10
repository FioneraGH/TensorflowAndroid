package com.fionera.tensorflow

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * Classifier
 * Created by fionera on 17-3-10 in android_project.
 */
interface Classifier {
    data class Recognition(var id: String, var title: String, var confidence: Float, var location: RectF?)

    fun recognizeImage(bitmap: Bitmap): List<Recognition>
    fun enableStatLogging(debug: Boolean)
    val statString: String
    fun close()
}
