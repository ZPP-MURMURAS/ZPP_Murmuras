package com.example.litertdeployment

import android.content.Context
import android.util.Log
import androidx.annotation.WorkerThread
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class ModelHandler(
    private val contextFetch: ()->Context,
    private val onLoadSuccessCb: ()->Unit,
    private val onLoadFailCb: ()->Unit,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    ): AutoCloseable, ViewModel() {
    private lateinit var interpreter: Interpreter
    private lateinit var embedding: HashMap<String, Long>
    private lateinit var embeddingRev: HashMap<Long, String>

    private var status: LoadStatus = LoadStatus.LOADING

    enum class LoadStatus {
        LOADED,
        FAILED,
        LOADING;
    }

    init {
        viewModelScope.launch {
            loadModel()
        }
    }

    private fun loadJSONFromAsset(fileName: String): String? {
        try {
            val file = contextFetch().assets.open(fileName)

            return file.bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            return null
        }
    }

    private fun loadEmbedding(): Boolean {
        val embeddingJSON = loadJSONFromAsset("tokenizer.json")?.let { JSONObject(it) } ?: return false
        embedding = HashMap()
        embeddingRev = HashMap()
        // unsafe, but who cares
        val dict: JSONObject = embeddingJSON.getJSONObject("model").getJSONObject("vocab")
        for (token: String in dict.keys()) {
            val value: Long = dict.getLong(token)
            embedding[token] = value
            embeddingRev[value] = token
        }
        return true
    }

    private fun tokenize(input: String): LongArray {
        val split = input.split(' ')
        val res = LongArray(SENTENCE_LENGTH)
        for (i in 0..<SENTENCE_LENGTH) {
            embedding[split.elementAtOrNull(i - SENTENCE_LENGTH + split.size) ?: "[PAD]"]?.let { res[i] = it }
        }
        return res
    }

    private fun deTokenize(noWords: Int): String {
        val res = emptyList<String>().toMutableList()
        var aMax: Long
        var max: Float
        for (i in 0..<SENTENCE_LENGTH) {
            aMax = -1
            max = Float.MIN_VALUE
            for (j in 0..<VOCAB_SIZE) {
                val f: Float = outputBuffer.get()
                if (f > max) {
                    max = f
                    aMax = j.toLong()
                }
            }
            val word = embeddingRev.getOrDefault(aMax, "")
            if (i + noWords > SENTENCE_LENGTH) {
                res += word
            }
        }
        return res.joinToString(" ")
    }

    private suspend fun loadModel() {
        return withContext(dispatcher) {
            // Load model file
            val loadResult = loadModelFile(contextFetch())

            // Determine if load was successful
            if (loadResult.isFailure) {
                status = LoadStatus.FAILED
                Log.e("Model loading", "failed to open model file")
                onLoadFailCb()
                return@withContext
            }

            if (!loadEmbedding()) {
                status = LoadStatus.FAILED
                Log.e("Model loading", "failed to open embeddings")
                onLoadFailCb()
                return@withContext
            }

            // Instantiate interpreter with loaded model
            val model = loadResult.getOrNull()
            status = model?.let {
                interpreter = Interpreter(it)
                LoadStatus.LOADED
            } ?: LoadStatus.FAILED
            if (status == LoadStatus.LOADED) {
                onLoadSuccessCb()
                Log.i("Model loading", "loaded model")
            }
            else {
                onLoadFailCb()
            }
        }
    }

    private fun loadModelFile(context: Context): Result<MappedByteBuffer?> {
        try {
            val descriptor = context.assets.openFd("model.tflite")

            FileInputStream(descriptor.fileDescriptor).use { stream ->
                return Result.success(
                    stream.channel.map(
                        /* mode = */ FileChannel.MapMode.READ_ONLY,
                        /* position = */ descriptor.startOffset,
                        /* size = */ descriptor.declaredLength
                    )
                )
            }
        } catch (e: Exception) {
            return Result.failure(e)
        }
    }

    override fun close() {
        if (status == LoadStatus.LOADED) {
            interpreter.close()
        }
    }

    private val outputBuffer = FloatBuffer.allocate(SENTENCE_LENGTH * VOCAB_SIZE)

    @WorkerThread
    fun runInterpreterOn(input: String): String {
        outputBuffer.clear()

        val inp = arrayOf(
            LongArray(128) { 1 }, // attention
            tokenize(input),            // tokens
            LongArray(128) { 0 }  // idk what exactly (it is described as token types or sth)
        )

        val out = mapOf(0 to outputBuffer)

        interpreter.runForMultipleInputsOutputs(inp, out)

        // Set output buffer limit to current position & position to 0
        outputBuffer.flip()
        val res = deTokenize(input.split(" ").size)

        outputBuffer.clear()

        // Return bytes converted to String
        return res
    }

    companion object {
        private const val VOCAB_SIZE = 30522
        private const val SENTENCE_LENGTH = 128
    }
}
