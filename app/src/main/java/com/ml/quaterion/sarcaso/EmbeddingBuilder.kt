package com.ml.quaterion.sarcaso

import android.content.Context
import android.os.AsyncTask
import org.json.JSONObject
import java.io.IOException
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap

class EmbeddingBuilder  {

    private var context : Context? = null
    private var filename : String? = null
    private var callback : VocabCallback? = null
    private var maxlen : Int? = null
    private var embeddingData : HashMap< String , DoubleArray >? = null
    private var embeddingDim : Int? = null


    constructor( context: Context , jsonFilename : String , embeddingDim : Int ){
        this.context = context
        this.filename = jsonFilename
        this.embeddingDim = embeddingDim
    }

    fun loadVocab () {
        val loadVocabularyTask = LoadVocabularyTask( callback )
        loadVocabularyTask.execute( loadJSONFromAsset( filename ))
    }

    private fun loadJSONFromAsset(filename : String? ): String? {
        var json: String? = null
        try {
            val inputStream = context!!.assets.open(filename )
            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()
            json = String(buffer)
        }
        catch (ex: IOException) {
            ex.printStackTrace()
            return null
        }
        return json
    }

    fun setCallback( callback: VocabCallback ) {
        this.callback = callback
    }

    fun tokenize ( message : String ): Array<DoubleArray> {
        val tokens : List<String> = Tokenizer.getTokens( message ).toList()
        val tokenizedMessage = ArrayList<DoubleArray>()
        for ( part in tokens ) {
            var vector : DoubleArray? = null
            if ( embeddingData!![part] == null ) {
                vector = DoubleArray( embeddingDim!! ){ 0.0 }
            }
            else{
                vector = embeddingData!![part]
            }
            tokenizedMessage.add( vector!! )

        }
        return tokenizedMessage.toTypedArray()
    }

    fun padSequence ( sequence : Array<DoubleArray> ) : Array<DoubleArray> {
        val maxlen = this.maxlen
        if ( sequence.size > maxlen!!) {
            return sequence.sliceArray( 0 until maxlen )
        }
        else if ( sequence.size < maxlen ) {
            val array = ArrayList<DoubleArray>()
            array.addAll( sequence.asList() )
            for ( i in array.size until maxlen ){
                array.add( DoubleArray( embeddingDim!! ){ 0.0 })
            }
            return array.toTypedArray()
        }
        else{
            return sequence
        }
    }

    fun setVocab( data : HashMap<String, DoubleArray>? ) {
        this.embeddingData = data
    }

    fun setMaxLength( maxlen : Int ) {
        this.maxlen = maxlen
    }

    interface VocabCallback {
        fun onDataProcessed( result : HashMap<String, DoubleArray>?)
    }

    private inner class LoadVocabularyTask(callback: VocabCallback?) : AsyncTask<String, Void, HashMap<String, DoubleArray>?>() {

        private var callback : VocabCallback? = callback

        override fun doInBackground(vararg params: String?): HashMap<String, DoubleArray>? {
            val jsonObject = JSONObject( params[0] )
            val iterator : Iterator<String> = jsonObject.keys()
            val data = HashMap< String , DoubleArray >()
            while ( iterator.hasNext() ) {
                val key = iterator.next()
                val array = jsonObject.getJSONArray( key )
                val embeddingArray = DoubleArray( array.length() )
                for ( x in 0 until array.length() ) {
                    embeddingArray[x] = array.get( x ) as Double
                }
                data[key] = embeddingArray
            }
            return data
        }

        override fun onPostExecute(vocab: HashMap<String, DoubleArray>?) {
            super.onPostExecute(vocab)
            callback?.onDataProcessed( vocab )
        }

    }

}
