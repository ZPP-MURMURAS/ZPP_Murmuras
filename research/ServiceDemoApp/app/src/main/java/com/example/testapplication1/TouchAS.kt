package com.example.testapplication1

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.view.accessibility.AccessibilityEvent

class TouchAS : AccessibilityService() {
    override fun onInterrupt() {
        // Handle interruption if the service is stopped.
    }

    override fun onServiceConnected() {
        val info = AccessibilityServiceInfo().apply {
            eventTypes = AccessibilityEvent.TYPE_VIEW_CLICKED or
                    AccessibilityEvent.TYPE_TOUCH_EXPLORATION_GESTURE_START or
                    AccessibilityEvent.TYPE_TOUCH_EXPLORATION_GESTURE_END or
                    AccessibilityEvent.TYPE_GESTURE_DETECTION_START or
                    AccessibilityEvent.TYPE_GESTURE_DETECTION_END or
                    AccessibilityEvent.TYPE_VIEW_LONG_CLICKED or
                    AccessibilityEvent.TYPE_VIEW_SELECTED or
                    AccessibilityEvent.TYPE_VIEW_SCROLLED or
                    AccessibilityEvent.TYPE_VIEW_FOCUSED or
                    AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED or
                    AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                    AccessibilityServiceInfo.FLAG_REQUEST_TOUCH_EXPLORATION_MODE
        }
        serviceInfo = info
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.let {
            when (it.eventType) {
                AccessibilityEvent.TYPE_GESTURE_DETECTION_END -> {
                    logGesture("Gesture detection ended")
                }

                AccessibilityEvent.TYPE_GESTURE_DETECTION_START -> {
                    logGesture("Gesture detection started")
                }

                AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED -> {
                    logNotificationStateChange(it)
                }

                AccessibilityEvent.TYPE_SPEECH_STATE_CHANGE -> {
                    logSpeechStateChange(it)
                }

                AccessibilityEvent.TYPE_TOUCH_INTERACTION_START -> {
                    val coordinates = getTouchCoordinates(it)
                    println("Touch start coordinates: $coordinates")
                }

                AccessibilityEvent.TYPE_TOUCH_INTERACTION_END -> {
                    val coordinates = getTouchCoordinates(it)
                    println("Touch end coordinates: $coordinates")
                }
                AccessibilityEvent.TYPE_VIEW_ACCESSIBILITY_FOCUSED -> {
                    logViewFocused(it)
                }

                AccessibilityEvent.TYPE_VIEW_ACCESSIBILITY_FOCUS_CLEARED -> {
                    logViewFocusCleared(it)
                }

                AccessibilityEvent.TYPE_VIEW_CONTEXT_CLICKED -> {
                    logContextClick(it)
                }

                AccessibilityEvent.TYPE_VIEW_FOCUSED -> {
                    logViewFocused(it)
                }

                AccessibilityEvent.TYPE_VIEW_HOVER_ENTER -> {
                    logHoverEnter(it)
                }

                AccessibilityEvent.TYPE_VIEW_HOVER_EXIT -> {
                    logHoverExit(it)
                }

                AccessibilityEvent.TYPE_VIEW_LONG_CLICKED -> {
                    logLongClick(it)
                }

                AccessibilityEvent.TYPE_VIEW_SCROLLED -> {
                    logViewScrolled(it)
                }

                AccessibilityEvent.TYPE_VIEW_SELECTED -> {
                    logViewSelected(it)
                }

                AccessibilityEvent.TYPE_VIEW_TARGETED_BY_SCROLL -> {
                    logViewTargetedByScroll(it)
                }

                AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> {
                    logTextChanged(it)
                }

                AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED -> {
                    logTextSelectionChanged(it)
                }

                AccessibilityEvent.TYPE_VIEW_TEXT_TRAVERSED_AT_MOVEMENT_GRANULARITY -> {
                    logTextTraversedGranularity(it)
                }

                AccessibilityEvent.TYPE_WINDOWS_CHANGED -> {
                    logWindowChanged(it)
                }

                AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED -> {
                    logWindowContentChanged(it)
                }

                AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                    logWindowStateChanged(it)
                }
            }
        }
    }

    // Helper methods to log various events:
    private fun logGesture(message: String) {
        println(message)
    }

    private fun getTouchCoordinates(event: AccessibilityEvent): String {
        val coordinates = event.text
        return coordinates.toString()
    }

    private fun logNotificationStateChange(event: AccessibilityEvent) {
        println("Notification state changed: ${event.text}")
    }

    private fun logSpeechStateChange(event: AccessibilityEvent) {
        println("Speech state changed: ${event.text}")
    }

    private fun logTouchInteraction(message: String) {
        println(message)
    }

    private fun logViewFocused(event: AccessibilityEvent) {
        println("View focused: ${event.text}")
    }

    private fun logViewFocusCleared(event: AccessibilityEvent) {
        println("View focus cleared: ${event.text}")
    }

    private fun logContextClick(event: AccessibilityEvent) {
        println("Context click detected on: ${event.text}")
    }

    private fun logHoverEnter(event: AccessibilityEvent) {
        println("Hover enter detected on: ${event.text}")
    }

    private fun logHoverExit(event: AccessibilityEvent) {
        println("Hover exit detected from: ${event.text}")
    }

    private fun logLongClick(event: AccessibilityEvent) {
        println("Long click detected on: ${event.text}")
    }

    private fun logViewScrolled(event: AccessibilityEvent) {
        println("View scrolled: ${event.text}")
    }

    private fun logViewSelected(event: AccessibilityEvent) {
        println("View selected: ${event.text}")
    }

    private fun logViewTargetedByScroll(event: AccessibilityEvent) {
        println("View targeted by scroll: ${event.text}")
    }

    private fun logTextChanged(event: AccessibilityEvent) {
        println("Text changed in view: ${event.text}")
    }

    private fun logTextSelectionChanged(event: AccessibilityEvent) {
        println("Text selection changed: ${event.text}")
    }

    private fun logTextTraversedGranularity(event: AccessibilityEvent) {
        println("Text traversed at movement granularity: ${event.text}")
    }

    private fun logWindowChanged(event: AccessibilityEvent) {
        println("Window changed: ${event.text}")
    }

    private fun logWindowContentChanged(event: AccessibilityEvent) {
        println("Window content changed: ${event.text}")
    }

    private fun logWindowStateChanged(event: AccessibilityEvent) {
        println("Window state changed: ${event.text}")
    }
}
