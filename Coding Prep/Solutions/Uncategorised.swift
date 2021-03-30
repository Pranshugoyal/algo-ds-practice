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
