//
//  Uncategorised.swift
//  Coding Prep
//
//  Created by Pranshu Goyal on 29/03/21.
//

import Foundation

//q56. Merge Intervals
func mergeIntervals(_ intervals: [[Int]]) -> [[Int]] {
    let sorted = intervals.sorted { (l, h) -> Bool in
        return l[0] <= h[0]
    }

    var result: [[Int]] = [sorted[0]]
    for interval in sorted[1...] {
        let last = result.last!
        if interval[0] > last[1] {
            result.append(interval)
        } else if interval[1] > last[1] {
            result[result.count-1][1] = interval[1]
        }
    }

    return result
}

//q253.Meeting Rooms II
func minHallsRequired(_ lectures: [[Int]]) -> Int {
    var prefixSum: [Int] = Array(repeating: 0, count: 50)
    for interval in lectures {
        prefixSum[interval[0]] += 1
        prefixSum[interval[1] + 1] -= 1
    }

    var ans = prefixSum[0]
    for i in 1..<prefixSum.count {
        prefixSum[i] += prefixSum[i - 1];
        ans = max(ans, prefixSum[i]);
    }

    return ans
}
