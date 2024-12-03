// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// After countless hours of debugging, blood, sweat, and tears, I was able to get the code to run on my Android device.
// Admittedly, it doesn't really work as intended however I don't have the knowledge to fix it at the moment.
// This code does SOMETHING, I don't think it does anything worthwhile though.

package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

import java.nio.ByteOrder
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.LongBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var vocab: Map<String, Long>

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var enableQuantizedModel: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        ortEnv = OrtEnvironment.getEnvironment()

        vocab = readRawVocabulary(R.raw.vocab)

        binding.enableQuantizedmodelToggle.setOnCheckedChangeListener { _, isChecked ->
            enableQuantizedModel = isChecked
            setORTAnalyzer()
        }

        binding.analyzeButton.setOnClickListener {
            val inputText = binding.textInput.text.toString()
            if (inputText.isNotEmpty()) {
                scope.launch {
                    try {
                        val result = analyzeText(inputText)
                        updateUI(result)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error analyzing text", e)
                        Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun readRawVocabulary(rawResId: Int): Map<String, Long> {
        val vocab = mutableMapOf<String, Long>()
        resources.openRawResource(rawResId).bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, line ->
                vocab[line.trim()] = index.toLong()
            }
        }
        return vocab
    }

    private suspend fun analyzeText(inputText: String): Result {
        return withContext(Dispatchers.IO) {
            val session = createOrtSession()
            if (session != null) {
                try {
                    val result = runInference(session, inputText)
                    result
                } catch (e: Exception) {
                    Log.e(TAG, "Error during inference", e)
                    Result(emptyList())
                }
            } else {
                Log.e(TAG, "Error creating ORT session")
                Result(emptyList())
            }
        }
    }

    fun createTokenTypeTensor(tokenTypeIds: Array<LongArray>): OnnxTensor {
        val myLongBuffer = LongBuffer.allocate(tokenTypeIds[0].size)
        for (tokenId in tokenTypeIds[0]) {
            myLongBuffer.put(tokenId.toLong())
        }
        myLongBuffer.rewind()
        val shape = longArrayOf(1, myLongBuffer.capacity().toLong())
        return OnnxTensor.createTensor(ortEnv, myLongBuffer, shape)
    }

    private suspend fun runInference(session: OrtSession?, inputText: String): Result {
        return withContext(Dispatchers.IO) {
            val inputTensor = createTextTensor(inputText, ortEnv!!, vocab)

            val attentionMaskTensor = createAttentionMaskTensor(inputText)

//            DEBUGGING
//            println("Input Tensor Shape: ${inputTensor::class.simpleName}")
//            println("Attention Mask Tensor Shape: ${attentionMaskTensor::class.simpleName}")
//
//            println("Input Tensor ${inputTensor}")
//            println("Attention Mask Tensor ${attentionMaskTensor}")

            val tokenTypeIds = LongArray(inputText.length) { 0 }

            val paddedTokenTypeIds = tokenTypeIds.toMutableList()

            while (paddedTokenTypeIds.size < 512) {
                paddedTokenTypeIds.add(0)
            }

            val paddedTokenTypeIdsArray = paddedTokenTypeIds.toLongArray()
            val tokenTypeIds2D = arrayOf(paddedTokenTypeIdsArray)
            val tokenTypeTensor = createTokenTypeTensor(tokenTypeIds2D)

            val inputs = mapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attentionMaskTensor,
                "token_type_ids" to tokenTypeTensor
            )

//            DEBUGGING
//            println("Input Tensor Shape: ${inputTensor.info.shape.contentToString()}")
//            println("Att mask Tensor Shape: ${attentionMaskTensor.info.shape.contentToString()}")
//            println("Token Type Tensor Shape: ${tokenTypeTensor.info.shape.contentToString()}")

            val output = session?.run(inputs)
            val resultTensor = output?.get(0)
            val resultScores = resultTensor?.value as? Array<Array<FloatArray>>

            if (resultScores.isNullOrEmpty()) {
                Log.e(TAG, "Empty output from model")
                return@withContext Result(emptyList())
            }

            val totalSize = resultScores?.sumBy { it.sumBy { it.size } }

            val flattenedScores = FloatArray(totalSize ?: 0)

            var index = 0
            for (batch in resultScores!!) {
                for (sequence in batch) {
                    for (score in sequence) {
                        flattenedScores[index] = score
                        index++
                    }
                }
            }

            val logits = flattenedScores
//            println("Logits before softmax: ${logits.joinToString(", ")}")
            val probabilities = softmax(logits)

            // Get the top 5 predictions
            val topIndices = probabilities.indices.sortedByDescending { probabilities[it] }.take(5)

//            DEBUGGING
//            println("Top indices: ${topIndices.joinToString(", ")}")
//            println("Top probabilities: ${topIndices.map { probabilities[it] }.joinToString(", ")}")

            val topResults = topIndices.map { index ->
                ResultItem(
                    token = index,
                    tokenStr = vocab.entries.firstOrNull { it.value == index.toLong() }?.key ?: "Unknown",
                    score = probabilities[index]
                )
            }

            // Print results for debugging
//            for (result in topResults) {
//                println("Token: ${result.tokenStr}, Score: ${result.score}")
//            }

            Result(topResults)
        }
    }

    fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f // Avoid overflow
        val expLogits = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sumExpLogits = expLogits.sum()
        return expLogits.map { it / sumExpLogits }.toFloatArray()
    }


    data class ResultItem(
        val token: Int,
        val tokenStr: String,
        val score: Float
    )

    fun createAttentionMaskTensor(inputText: String): OnnxTensor {
        val tokens = tokenize(inputText, vocab, maxLength = 512).tokens
        val attentionMask = FloatArray(tokens.size) { 1f }

        val tokenIds = tokens.map {
            (vocab[it] ?: vocab["[PAD]"] ?: 0).toLong()
        }

        val paddingStart = tokenIds.indexOfFirst { it == vocab["[PAD]"]?.toLong() }
        if (paddingStart != -1) {
            attentionMask.fill(0f, paddingStart, attentionMask.size)
        }

        val byteBuffer = floatArrayToByteBuffer(attentionMask)
        byteBuffer.rewind()

        val myLongBuffer = LongBuffer.allocate(attentionMask.size)
        for (tokenId in attentionMask) {
            myLongBuffer.put(tokenId.toLong())
        }
        myLongBuffer.rewind()
        val inputShape = longArrayOf(1.toLong(), myLongBuffer.capacity().toLong())

//        DEBUGGING
//        println("Attention mask long buffeer length ${myLongBuffer.capacity()}")
//        println("Attention Mask Shape: ${inputShape[0]} x ${inputShape[1]}")
//        println("byteBuffersize: ${byteBuffer.capacity()}")
//        println("Token IDs Data Type: ${tokenIds::class.simpleName}")
//        println("Attention Mask Data Type: ${attentionMask::class.simpleName}")

        return OnnxTensor.createTensor(ortEnv, myLongBuffer, inputShape)
    }

    fun createTextTensor(
        inputText: String,
        ortEnv: OrtEnvironment,
        vocab: Map<String, Long>,
        maxLength: Int = 512
    ): OnnxTensor {
        val tokenizedOutput = tokenize(inputText, vocab, maxLength)

        val paddedTokenIds = tokenizedOutput.tokens.take(maxLength).toMutableList()
        while (paddedTokenIds.size < maxLength) {
            paddedTokenIds.add("[PAD]")
        }

        val tokenIds = paddedTokenIds.map {
            (vocab[it] ?: vocab["[PAD]"] ?: 0).toLong()
        }.toLongArray()

        val bufferSize = tokenIds.size * Long.SIZE_BYTES
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())

        for (tokenId in tokenIds) {
            byteBuffer.putLong(tokenId)
        }
        byteBuffer.flip()

//        DEBUGGING
//        println("Token IDs: ${tokenIds.size}")
//        println("ByteBuffer size: ${byteBuffer.capacity()}")
//        println("Token IDs Data Type: ${tokenIds::class.simpleName}")
//        println("Attention Mask Data Type: ${byteBuffer::class.simpleName}")

        val myLongBuffer = LongBuffer.allocate(tokenIds.size)
        for (tokenId in tokenIds) {
            myLongBuffer.put(tokenId.toLong())
        }
        myLongBuffer.rewind()

        val inputShape = longArrayOf(1.toLong(), myLongBuffer.capacity().toLong())
//        DEBUGGING
//        println("Token IDs Long Buffer Length: ${myLongBuffer.capacity()}")
        return OnnxTensor.createTensor(ortEnv, myLongBuffer, inputShape)
    }

    fun floatArrayToByteBuffer(floatArray: FloatArray): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(floatArray.size * Float.SIZE_BYTES)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (value in floatArray) {
            byteBuffer.putFloat(value)
        }

        byteBuffer.flip()

        return byteBuffer
    }

    fun tokenize(text: String, vocab: Map<String, Long>, maxLength: Int): TokenizedOutput {
        var processedText = text.toLowerCase(Locale.ROOT)
//        processedText = processedText.replace("[^a-z0-9 ]".toRegex(), "")
        val words = processedText.split(" ")

        val tokens = mutableListOf<String>()
        tokens.add("[CLS]")

        for (word in words) {
            if (vocab.containsKey(word)) {
                tokens.add(word)
            } else {
                val subwords = word.chunked(2)
                subwords.forEach { subword ->
                    tokens.add(subword)
                }
            }
        }

        tokens.add("[SEP]")

        while (tokens.size < maxLength) {
            tokens.add("[PAD]")
        }
        if (tokens.size > maxLength) {
            tokens.subList(maxLength, tokens.size).clear()
        }

//        DEBUGGING
//        println("Tokens: ${tokens.size} (should be 512)")
        return TokenizedOutput(tokens, tokens.size)
    }

    data class TokenizedOutput(val tokens: MutableList<String>, val length: Int)

    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        ortEnv?.createSession(readModel())
    }

    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        val modelID =
            if (enableQuantizedModel) R.raw.mobilebert else R.raw.bert_model_quantized
        resources.openRawResource(modelID).readBytes()
    }

    private fun updateUI(result: Result) {
        runOnUiThread {
            var text = "Top results:\n"
            for (item in result.topResults) {
                text += "${item.tokenStr}: ${item.score}\n"
            }

            binding.resultText.text = text
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv?.close()
    }

    companion object {
        public const val TAG = "ORTTextClassifier"
    }

    private fun setORTAnalyzer() {
        val inputText = binding.textInput.text.toString()
        if (inputText.isNotEmpty()) {
            scope.launch {
                try {
                    val ortSession = createOrtSession()
                    if (ortSession != null) {
                        val result = analyzeText(inputText)
                        updateUI(result)
                    } else {
                        Log.e(TAG, "Error creating ORT session")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error analyzing text", e)
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show()
        }
    }

    data class Result(val topResults: List<ResultItem>)
}
