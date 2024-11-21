package com.example.testapplication1

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*

class FibonacciService : Service() {

    private var isRunning = true // To control the service lifecycle

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Start the background coroutine
        // This global scope is sensitive so we should use it carefully
        GlobalScope.launch(Dispatchers.IO) {
            runFibonacci()
        }
        return START_STICKY
    }

    private suspend fun runFibonacci() {
        var n = 0
        while (isRunning) {
            val fibNumber = fibonacci(n)
            Log.d("FibonacciService", "Fibonacci($n): $fibNumber")
            n++
            delay(1000)
        }
    }

    // Recursive Fibonacci function
    private fun fibonacci(n: Int): Long {
        if (n <= 1) return n.toLong()
        return fibonacci(n - 1) + fibonacci(n - 2)
    }

    override fun onDestroy() {
        super.onDestroy()
        isRunning = false
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}
