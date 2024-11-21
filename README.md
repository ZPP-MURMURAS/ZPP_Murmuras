Created a simple service that prints to logs the fibonacci numbers one by one; it runs in the background; this can be useful when debugging (this is commented out at the moment)
Created a simple worker to compute a large number in the background
Created a simple toast that shows up every 5 seconds (demonstrates the use of async)
Utilized android accessibility services to register the apps the user visits and whether they scroll. 
Attempted to capture the volume up and down buttons as well as the boot signal, but it is a work in progress. 

Please note that you must give permission to the app to use android accessibility 
Settings > Accessibility > Downloaded Apps > [App Name] > Turn on switch “Use [App Name] > Allow everything 

Useful links:
introduction -> https://developer.android.com/develop/background-work/services
łloading and scheduling computations -> https://developer.android.com/reference/android/app/job/JobScheduler 
https://developer.android.com/reference/android/app/job/JobInfo.Builder#Builder> 
threads and processes -> https://developer.android.com/guide/components/processes-and-threads 
restrictions on background services -> https://developer.android.com/about/versions/oreo/background#services 
workmanager -> https://developer.android.com/topic/libraries/architecture/workmanager 
android accessibility -> https://developer.android.com/guide/topics/ui/accessibility/service 


