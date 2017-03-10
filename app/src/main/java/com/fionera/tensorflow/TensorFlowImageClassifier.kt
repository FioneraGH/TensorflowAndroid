package com.fionera.tensorflow

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Trace
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.*

/**
 * TensorFlowImageClassifier
 * Created by fionera on 17-3-10 in android_project.
 */
class TensorFlowImageClassifier : Classifier {

    companion object {
        val MAX_RESULTS = 3
        val THRESHOLD = 0.1f

        @Throws(IOException::class)
        fun create(assetManager: AssetManager,
                   modelFilename: String,
                   labelFileName: String,
                   inputSize: Int,
                   imageMean: Int,
                   imageStd: Float,
                   inputName: String,
                   outputName: String): TensorFlowImageClassifier {
            val c = TensorFlowImageClassifier()
            c.inputName = inputName
            c.outputName = outputName

            val actualFilename = labelFileName.split("file:///android_asset/")[1]
            println("Reading label from:$actualFilename")

            val br = BufferedReader(InputStreamReader(assetManager.open(actualFilename)))
            var line = br.readLine()
            while (line != null) {
                c.labels.addElement(line)
                line = br.readLine()
            }
            br.close()

            c.inferenceInterface = TensorFlowInferenceInterface()
            if (0 != c.inferenceInterface.initializeTensorFlow(assetManager, modelFilename)) {
                throw RuntimeException("TensorFlow init failed")
            }

            val numClasses = c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1).toInt()
            println("Read ${c.labels.size} labels, output layer size is $numClasses")

            c.inputSize = inputSize
            c.imageMean = imageMean
            c.imageStd = imageStd

            c.outputNames = arrayOf(outputName)
            c.intValues = IntArray(inputSize * inputSize)
            c.floatValues = FloatArray(inputSize * inputSize * 3)
            c.outputs = FloatArray(numClasses)

            return c
        }
    }

    private var inputName = ""
    private var outputName = ""
    private var inputSize = 0
    private var imageMean = 0
    private var imageStd = 0f

    private var labels = Vector<String>()
    private var intValues: IntArray? = null
    private var floatValues: FloatArray? = null
    private var outputs: FloatArray? = null
    private var outputNames = arrayOf<String>()

    private lateinit var inferenceInterface: TensorFlowInferenceInterface

    override fun recognizeImage(bitmap: Bitmap): List<Classifier.Recognition> {
        Trace.beginSection("recognizeImage")

        Trace.beginSection("preProcessBitmap")
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        intValues?.forEachIndexed { index, i ->
            floatValues!![index * 3 + 0] = (i.shr(16).and(0xFF) - imageMean) / imageStd
            floatValues!![index * 3 + 1] = (i.shr(8).and(0xFF) - imageMean) / imageStd
            floatValues!![index * 3 + 2] = (i.shr(0).and(0xFF) - imageMean) / imageStd
        }
        Trace.endSection()

        Trace.beginSection("fillNodeFloat")
        inferenceInterface.fillNodeFloat(inputName, intArrayOf(1, inputSize, inputSize, 3), floatValues)
        Trace.endSection()

        Trace.beginSection("runInference")
        inferenceInterface.runInference(outputNames)
        Trace.endSection()

        Trace.beginSection("readNodeFloat")
        inferenceInterface.readNodeFloat(outputName,outputs)
        Trace.endSection()

        val pq = PriorityQueue<Classifier.Recognition>(3,
                Comparator { lhs, rhs -> (rhs.confidence - lhs.confidence).toInt() })
        outputs?.forEachIndexed { index, fl ->
            if (fl > THRESHOLD) {
                pq.add(Classifier.Recognition("$index",
                        if (labels.size > index) labels[index] else "unknown",
                        fl,
                        null))
            }
        }

        val recognitions = arrayListOf<Classifier.Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0..recognitionsSize - 1) {
            recognitions.add(pq.poll())
        }

        Trace.endSection()

        return recognitions
    }

    override fun enableStatLogging(debug: Boolean) {
        inferenceInterface.enableStatLogging(debug)
    }

    override val statString: String
        get() = inferenceInterface.statString

    override fun close() {
        inferenceInterface.close()
    }
}