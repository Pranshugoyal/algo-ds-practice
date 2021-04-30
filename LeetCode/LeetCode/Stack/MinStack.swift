//
//  MinStack.swift
//  LeetCode
//
//  Created by Pranshu Goyal on 10/04/20.
//  Copyright © 2020 Pranshu Goyal. All rights reserved.
//

class MinStack {

	private var array: [Int] = []
	private var min = Int.max

    func push(_ x: Int) {
		guard !array.isEmpty else {
			min = x
			array.append(x)
			return
		}

		if x < min {
			array.append(2*x - min)
			min = x
		} else {
			array.append(x)
		}
    }

    func pop() -> Int {
		let y = array.popLast()!
		if y < min {
			let t = min
			min = 2*min - y
			return t
		} else {
			return y
		}
    }

    func top() -> Int {
		let y = array.last!
		if y < min {
			return min
		} else {
			return y
		}
    }

    func getMin() -> Int {
		return min
    }
}
