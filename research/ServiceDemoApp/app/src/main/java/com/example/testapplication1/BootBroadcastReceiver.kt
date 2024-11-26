package com.example.testapplication1

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

class BootBroadcastReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (Intent.ACTION_BOOT_COMPLETED == intent.action) {
            // Start the service after boot
            Log.d("BootBroadcastReceiver", "Boot completed")
            val serviceIntent = Intent(context, TouchAS::class.java)
            context.startService(serviceIntent)
        }
    }
}
