package com.example.testapplication1

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.util.Log

class VolumeButtonReceiver : BroadcastReceiver() {
    private var lastVolume = -1

    @SuppressLint("UnsafeProtectedBroadcastReceiver")
    override fun onReceive(context: Context, intent: Intent) {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)

        if (lastVolume != currentVolume) {
            if (currentVolume > lastVolume) {
                Log.d("VolumeButtonReceiver", "Volume Up Pressed")
            } else {
                Log.d("VolumeButtonReceiver", "Volume Down Pressed")
            }
            lastVolume = currentVolume
        }
    }
}
