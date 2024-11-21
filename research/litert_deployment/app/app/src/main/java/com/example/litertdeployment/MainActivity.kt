package com.example.litertdeployment

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.annotation.WorkerThread
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.litertdeployment.ui.theme.LiteRTDeploymentTheme

class MainActivity : ComponentActivity() {
    private lateinit var modelHandler: ModelHandler // should be as dependency injection but this is just poc
    private var isModelAvailable: Boolean = false;
    private val EXAMPLE_PROMPT = "[SEP] my favourite tv series is [MASK] ."

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        modelHandler = ModelHandler({applicationContext}, { isModelAvailable = true; updateGreetingLabel(generate(EXAMPLE_PROMPT)) }, { isModelAvailable = false });
        setContent {
            LiteRTDeploymentTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    private var greetingText = mutableStateOf(EXAMPLE_PROMPT);

    fun updateGreetingLabel(newGreeting: String) {
        greetingText.value += "\n$newGreeting"
    }

    fun generate(input: String): String {
        if (isModelAvailable) {
            return modelHandler.runInterpreterOn(input);
        }
        else {
            return "fail"
        }
    }

    @Composable
    fun Greeting(modifier: Modifier = Modifier) {
        Text(
            text = greetingText.value,
            modifier = modifier
        )
    }

    @Preview(showBackground = true)
    @Composable
    fun GreetingPreview() {
        LiteRTDeploymentTheme {
            Greeting()
        }
    }
}