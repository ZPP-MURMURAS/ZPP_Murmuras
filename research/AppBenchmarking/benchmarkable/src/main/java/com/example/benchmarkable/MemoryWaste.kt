package com.example.benchmarkable

class MemoryWaste {
    private lateinit var myMap : HashMap<String, Int>
    fun wasteMemory(n: Int) {
        myMap = HashMap<String, Int>()
        for (i: Int in 1 .. n) {
            myMap[i.toString()] = i
        }
    }
}