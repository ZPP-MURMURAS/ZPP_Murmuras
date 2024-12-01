package com.example.testapplication1

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import androidx.work.WorkRequest
import com.example.testapplication1.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var toastRunnable: Runnable
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        resultTextView = binding.resultTextView

        // Display Toast every 5 seconds
        startRepeatingToast()
        startSimpleWorker()

//        val intent = Intent(this, FibonacciService::class.java)
//        startService(intent)
    }

    private fun startSimpleWorker() {
        val workRequest: WorkRequest = OneTimeWorkRequest.Builder(SimpleWorker::class.java).build()
        WorkManager.getInstance(this).enqueue(workRequest)
        WorkManager.getInstance(this).getWorkInfoByIdLiveData(workRequest.id).observe(this) { workInfo ->
            if (workInfo != null && workInfo.state.isFinished) {
                // Retrieve the computed result from the output data
                val result = workInfo.outputData.getInt("result_key", -1)
                resultTextView.text = "Computed Result: $result"
            }
        }
    }

    /// Async with Runnable ///
    private fun startRepeatingToast() {
        toastRunnable = Runnable {
            Toast.makeText(this, "Hello", Toast.LENGTH_SHORT).show()
            handler.postDelayed(toastRunnable, 5000) // Schedule again after 5 seconds
        }
        handler.post(toastRunnable)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(toastRunnable)
        val intent = Intent(this, FibonacciService::class.java)
        stopService(intent)
    }
}
