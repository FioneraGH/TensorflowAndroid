package com.fionera.tensorflow

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import com.flurgle.camerakit.CameraListener
import kotlinx.android.synthetic.main.activity_main.*
import java.util.concurrent.Executors

class MainActivity : BaseActivity() {

    companion object {
        val INPUT_SIZE = 224
        val IMAGE_MEAN = 117
        val IMAGE_STD = 1.0f
        val INPUT_NAME = "input"
        val OUTPUT_NAME = "output"

        val MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb"
        val LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt"
    }

    private var classifier: Classifier? = null
    private var executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cv_preview.setCameraListener(object : CameraListener() {
            override fun onPictureTaken(jpeg: ByteArray) {
                val bitmap = Bitmap.createScaledBitmap(
                        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size),
                        INPUT_SIZE,
                        INPUT_SIZE,
                        false)
                iv_preview.setImageBitmap(bitmap);

                val results = classifier?.recognizeImage(bitmap)
                tv_result.text = results?.toString()
            }
        })

        btn_swap.setOnClickListener {
            cv_preview.toggleFacing()
        }

        btn_detect.setOnClickListener {
            cv_preview.captureImage()
        }

        initTensorFlow()
    }

    private fun initTensorFlow() {
        executor.execute {
            try {
                classifier = TensorFlowImageClassifier.create(
                        assets,
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        cv_preview.start()
    }

    override fun onPause() {
        super.onPause()
        cv_preview.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute {
            classifier?.close()
        }
    }
}

