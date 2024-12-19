package com.example.microbenchmark

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.benchmark.junit4.BenchmarkRule
import androidx.benchmark.junit4.measureRepeated
import androidx.core.app.ActivityCompat.startActivityForResult
import androidx.core.content.ContextCompat.startActivity
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.benchmarkable.MemoryWaste
import junit.framework.TestCase.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Locale


/**
 * Benchmark, which will execute on an Android device.
 *
 * The body of [BenchmarkRule.measureRepeated] is measured in a loop, and Studio will
 * output the result. Modify your code to see how it affects performance.
 */
@RunWith(AndroidJUnit4::class)
class OverlayPermissionTest {
    @Test
    fun testOverlayPermission() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        assertTrue(
            "Overlay permission is not granted",
            Settings.canDrawOverlays(context)
        )
    }
}

@RunWith(AndroidJUnit4::class)
class ExampleBenchmark {
    private lateinit var memWaste : MemoryWaste

    @get:Rule
    val benchmarkRule = BenchmarkRule()

    @Before
    fun init() {
        memWaste = MemoryWaste()
    }

    @Test
    fun log() {
        benchmarkRule.measureRepeated {
            memWaste.wasteMemory(1000000)
        }
    }
}