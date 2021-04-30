//
//  Independent Problems.swift
//  LeetCode
//
//  Created by Pranshu Goyal on 02/04/20.
//  Copyright © 2020 Pranshu Goyal. All rights reserved.
//

import Foundation

func lengthOfLongestSubstring(_ s: String) -> Int {

	var position: [Character: Int] = [:]

	var currentLen = 0
	var currentSubstringStart = 0

	var maxLen = 0
//	var maxSubstringStart = 0

	for (index, character) in s.enumerated() {
		guard let cPos = position[character] else {
			position[character] = index
			continue
		}

		guard cPos >= currentSubstringStart else {
			position[character] = index
			continue
		}

		currentLen = index - currentSubstringStart
		if maxLen < currentLen {
			maxLen = currentLen
//			maxSubstringStart = currentSubstringStart
		}

		currentSubstringStart = cPos + 1
		position[character] = index
	}

	if maxLen < s.count - currentSubstringStart {
		maxLen = s.count - currentSubstringStart
//		maxSubstringStart = currentSubstringStart
	}

	return maxLen
}

func countElements(_ arr: [Int]) -> Int {
	let set = Set(arr)
	return arr.reduce(into: 0) { (count, num) in
		count += set.contains(num+1) ? 1 : 0
	}
}

func backspaceCompare(_ S: String, _ T: String) -> Bool {
	func getOutputString(_ string: String) -> String {
		var output: String = ""

		for character in string {
			if character == "#" {
				_ = output.popLast()
			} else {
				output.append(character)
			}
		}

		return output
	}
	return getOutputString(S) == getOutputString(T)
}

//Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
//Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
func removeDuplicates(_ nums: inout [Int]) -> Int {
	var j = 0
	for i in 0..<nums.count {
		if i > 0, nums[i] == nums[i-1] {
			continue
		}

		nums[j] = nums[i]
		j += 1
	}
	return j
}

/*
We have a collection of stones, each stone has a positive integer weight.
Each turn, we choose the two heaviest stones and smash them together.  Suppose the stones have weights x and y with x <= y.  The result of this smash is:

If x == y, both stones are totally destroyed;
If x != y, the stone of weight x is totally destroyed, and the stone of weight y has new weight y-x.
At the end, there is at most 1 stone left.  Return the weight of this stone (or 0 if there are no stones left.)
*/
func lastStoneWeight(_ stones: [Int]) -> Int {
	var sortedWeights = stones.sorted()

	func smash(_ x: Int, and y: Int) -> Int? {
		if x == y {
			return nil
		} else {
			return y - x
		}
	}

	while sortedWeights.count > 1 {
		let y = sortedWeights.popLast()!
		let x = sortedWeights.popLast()!
		if let newWeight = smash(x, and: y) {
			let i = sortedWeights.insertionIndexOf(newWeight, isOrderedBefore: <)
			sortedWeights.insert(newWeight, at: i)
		}

	}

	return sortedWeights.first ?? 0
}

///Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
///Input: [0,1,0]
///Output: 2
///Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.
func findMaxLength(_ nums: [Int]) -> Int {
	var maxLength = 0
	var sumsHash: [Int: Int] = [0: -1]
	var sum = 0
	for (index, num) in nums.enumerated() {
		sum = sum + (num == 1 ? 1 : -1)
		if let lastIndex = sumsHash[sum] {
			maxLength = max(maxLength, index - lastIndex)
		} else {
			sumsHash[sum] = index
		}
	}

	return maxLength
}

/*
You are given a string s containing lowercase English letters, and a matrix shift, where shift[i] = [direction, amount]:

direction can be 0 (for left shift) or 1 (for right shift).
amount is the amount by which string s is to be shifted.
A left shift by 1 means remove the first character of s and append it to the end.
Similarly, a right shift by 1 means remove the last character of s and add it to the beginning.
Return the final string after all operations.
*/
func stringShift(_ s: String, _ shift: [[Int]]) -> String {

	var finalShift = 0
	for step in shift {
		finalShift += step[0] == 0 ? -step[1] : step[1]
	}

	guard finalShift != 0 else {
		return s
	}

	let partitionOffset: Int
	if finalShift > 0 {
		finalShift %= s.count
		partitionOffset = s.count - finalShift
	} else {
		finalShift = -finalShift%s.count
		partitionOffset = finalShift
	}

	let partitionIndex = s.index(s.startIndex, offsetBy: partitionOffset)
	return String(s[partitionIndex...]) + String(s[..<partitionIndex])
}

/*
Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
*/
func productExceptSelf(_ nums: [Int]) -> [Int] {
	let zeroCount = nums.filter({$0 == 0}).count

	if zeroCount > 1 {
		return Array<Int>(repeating: 0, count: nums.count)
	}

	let product = nums.reduce(1) { (partialResult, num) in
		num == 0 ? partialResult : partialResult*num
	}

	if zeroCount == 1, let index = nums.firstIndex(of: 0) {
		var result = Array<Int>(repeating: 0, count: nums.count)
		result[index] = product
		return result
	}

	return nums.map({product/$0})
}

/*
Given a string containing only three types of characters: '(', ')' and '*', write a function to check whether this string is valid. We define the validity of a string by these rules:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
An empty string is also valid.
*/
func checkValidString(_ s: String) -> Bool {
	guard !s.isEmpty else { return true }

	var (low, high) = (0, 0)

	for char in s {
		if char == "(" {
			low += 1
			high += 1
		} else if char == ")" {
			low = max(0, low-1)
			high -= 1
		} else {
			low = max(0, low-1)
			high += 1
		}

		if (high < 0) {
			return false
		}
	}

	return low == 0
}

///Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
///Note: You can only move either down or right at any point in time.
func numIslands(_ grid: [[Character]]) -> Int {
	guard !grid.isEmpty else {
		return 0
	}

	var visited: [[Bool]] = []

	func dfs(row: Int, column: Int) {
		let rowNeighbours = 	[-1, -0, 0, 1]
		let columnNeighbours =	[-0, -1, 1, 0]

		visited[row][column] = true

		func isConnected(row: Int, column: Int) -> Bool {
			return 	row >= 0 && row < grid.count &&
					column >= 0 && column < grid[0].count &&
					!visited[row][column] && grid[row][column] == "1"
		}

		for i in 0..<rowNeighbours.count {
			let neighbourRow = row + rowNeighbours[i]
			let neighbourColumn = column + columnNeighbours[i]
			if isConnected(row: neighbourRow, column: neighbourColumn) {
				dfs(row: neighbourRow, column: neighbourColumn)
			}
		}
	}

	let row = Array<Bool>(repeating: false, count: grid[0].count)
	visited = Array<[Bool]>(repeating: row, count: grid.count)

	var islands = 0
	for row in 0..<grid.count {
		for column in 0..<grid[row].count where !visited[row][column] {
			if grid[row][column] == "1" {
				dfs(row: row, column: column)
				islands += 1
			}
		}
	}

	return islands
}

struct BinaryMatrix {
	let data: [[Int]]

	func get(_ x: Int, _ y: Int) -> Int {
		return data[x][y]
	}

	func dimensions() -> [Int] {
		guard !data.isEmpty else {
			return [0, 0]
		}
		return [data.count, data[0].count]
	}
}

///A binary matrix means that all elements are 0 or 1. For each individual row of the matrix, this row is sorted in non-decreasing order.
///Given a row-sorted binary matrix binaryMatrix, return leftmost column index(0-indexed) with at least a 1 in it. If such index doesn't exist, return -1.
func leftMostColumnWithOne(_ binaryMatrix: BinaryMatrix) -> Int {
	let rows = binaryMatrix.dimensions()[0]
	let columns = binaryMatrix.dimensions()[1]

	var row = rows - 1
	var column = columns - 1
	while row >= 0, column >= 0 {
		if binaryMatrix.get(row, column) == 0 {
			row -= 1
		} else {
			column -= 1
		}
	}


	return column == columns-1 ? -1 : column+1
}

///Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.
func subarraySum(_ nums: [Int], _ k: Int) -> Int {
	var d: [Int: Int] = [:]
	d[0] = 1

	var sum = 0
	var res = 0

	for i in 0..<nums.count {
		sum += nums[i]
		res += d[sum-k, default: 0]
		d[sum, default: 0] += 1
	}

	return res
}

func rangeBitwiseAnd(_ m: Int, _ n: Int) -> Int {
	var count = 0
	var m = m
	var n = n
	while m != n {
		m >>= 1
		n >>= 1
		count += 1
	}
	return m << count
}
