package com.example.testapplication1

import android.content.Context
import androidx.work.Data
import androidx.work.Worker
import androidx.work.WorkerParameters
import kotlin.random.Random

class SimpleWorker(appContext: Context, workerParams: WorkerParameters) : Worker(appContext, workerParams) {

    override fun doWork(): Result {
        // Perform some background computation
        val result = performComputation()

        // Pass the result back to MainActivity
        val outputData = Data.Builder()
            .putInt("result_key", result)
            .build()

        return Result.success(outputData)
    }

    // Example of a worker doing a job in the "background"
    private fun performComputation(): Int {
        val lowerBound = 10000
        val upperBound = 909090

        var sum = 0
        for (i in 1..Random.nextInt(lowerBound, upperBound) * 1000) {
            sum += i
        }
        return sum
    }
}
