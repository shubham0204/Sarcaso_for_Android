package com.ml.quaterion.sarcaso

import android.os.Bundle
import android.text.TextUtils
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.content_main.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var embeddingBuilder: EmbeddingBuilder
    private var isVocabLoaded = false


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(toolbar)

        startLoadVocab()

        classify_button.setOnClickListener( View.OnClickListener {
            val message = message_text.text.toString().toLowerCase().trim()
            if ( !TextUtils.isEmpty( message ) ){
                if ( isVocabLoaded ) {
                    val tokenizedMessage = embeddingBuilder.tokenize( message )
                    println( Arrays.deepToString( tokenizedMessage ) )
                    val paddedMessage = embeddingBuilder.padSequence( tokenizedMessage )
                    println( Arrays.deepToString( paddedMessage ) )
                    val results = classifySequence( paddedMessage )
                    val class1 = results[0]
                    val class2 = results[1]
                    result_text.text = "SARCASTIC  : $class2\nNOT SARCASTIC : $class1 "
                }
                else {
                    Toast.makeText(this@MainActivity, "Vocab not loaded", Toast.LENGTH_SHORT).show()
                }
            }
            else{
                Toast.makeText( this@MainActivity, "Please enter a message.", Toast.LENGTH_LONG).show();
            }
        })

    }

    private fun startLoadVocab(){
        embeddingBuilder = EmbeddingBuilder( this , "embedding.json" , 50 )
        embeddingBuilder.setMaxLength( 16 )
        embeddingBuilder.setCallback( object : EmbeddingBuilder.VocabCallback {
            override fun onDataProcessed( result : HashMap<String, DoubleArray>? ) {
                embeddingBuilder.setVocab( result )
                isVocabLoaded = true
            }
        })
        embeddingBuilder.loadVocab()
    }


    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val MODEL_ASSETS_PATH = "model.tflite"
        val assetFileDescriptor = assets.openFd(MODEL_ASSETS_PATH)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startoffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    }

    fun classifySequence ( sequence : Array<DoubleArray> ): FloatArray {
        val interpreter = Interpreter( loadModelFile() )
        val inputs : Array<Array<FloatArray>> =  arrayOf(
            sequence.map{
                it.map {
                    it.toFloat()
                }.toFloatArray()
            }.toTypedArray()
        )
        val outputs : Array<FloatArray> = arrayOf( floatArrayOf( 0.0f , 0.0f ) )
        interpreter.run( inputs , outputs )
        return outputs[0]
    }

}
