# Android Background Services and Accessibility Features

## Features Implemented
1. **Fibonacci Logger Service**  
   - A background service that logs Fibonacci numbers one by one.
   - Currently commented out but can be useful for debugging purposes.

2. **Background Worker**  
   - Computes a large number in the background using asynchronous processing.

3. **Toast Notification**  
   - A simple toast message that appears every 5 seconds to demonstrate the use of asynchronous tasks.

4. **Android Accessibility Services**  
   - Registers the apps visited by the user and detects scrolling behavior.
   - Attempts to capture the **Volume Up/Down** button actions and the **Boot Signal** (work in progress).

---

## Permissions Required
To utilize the app's features, **Accessibility Services** must be enabled:

1. Go to **Settings** > **Accessibility** > **Downloaded Apps**.  
2. Select **[App Name]** and toggle the switch for "Use [App Name]".  
3. Grant all required permissions when prompted.

---

## Useful Resources
- **Introduction to Android Services**  
  [Android Services Overview](https://developer.android.com/develop/background-work/services)

- **Loading and Scheduling Computations**  
  - [JobScheduler API](https://developer.android.com/reference/android/app/job/JobScheduler)  
  - [JobInfo.Builder](https://developer.android.com/reference/android/app/job/JobInfo.Builder#Builder)

- **Threads and Processes**  
  [Android Threads and Processes](https://developer.android.com/guide/components/processes-and-threads)

- **Restrictions on Background Services (Oreo and Later)**  
  [Background Service Restrictions](https://developer.android.com/about/versions/oreo/background#services)

- **WorkManager**  
  [WorkManager Overview](https://developer.android.com/topic/libraries/architecture/workmanager)

- **Android Accessibility**  
  [Accessibility Service Guide](https://developer.android.com/guide/topics/ui/accessibility/service)

---

## Notes
- Additional work is needed to finalize the handling of **Volume Button Presses** and the **Boot Signal**.

