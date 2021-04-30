//
//  ListNode.swift
//  LeetCode
//
//  Created by Pranshu Goyal on 22/03/20.
//  Copyright © 2020 Pranshu Goyal. All rights reserved.
//

import Foundation

class ListNode {
	var val: Int
	var next: ListNode?

	init(_ val: Int) {
		self.val = val
		self.next = nil
	}

	convenience init?(from array: [Int]) {
		guard !array.isEmpty else {
			return nil
		}
		self.init(array[0])

		guard array.count >= 2 else {
			return
		}

		var head: ListNode? = self
		for int in array[1...] {
			head?.next = ListNode(int)
			head = head?.next
		}
	}

	func getArray() -> [Int] {
		var head: ListNode? = self
		var array: [Int] = []
		while let node = head {
			array.append(node.val)
			head = node.next
		}
		return array
	}
}

func middleNode(_ head: ListNode?) -> ListNode? {
	guard let head = head else {
		return nil
	}

	var fastPointer: ListNode? = head
	var slowPointer: ListNode? = head

	while fastPointer != nil, fastPointer?.next != nil {
		fastPointer = fastPointer?.next?.next
		slowPointer = slowPointer?.next
	}

	return slowPointer
}

func swapPairs(_ head: ListNode?) -> ListNode? {
	if head?.next == nil {
		return head
	} else {
		let newLast = head?.next
		let newHead = swapPairs(head?.next)
		newLast?.next = head
		head?.next = nil
		return newHead
	}
}

func addTwoNumbersReversed(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
	var h1 = l1
	var h2 = l2

	var list: ListNode?
	var head = list
	func addNode(_ val: Int) {
		if list == nil {
			list = ListNode(val)
			head = list
		} else {
			head?.next = ListNode(val)
			head = head?.next
		}
	}

	var carry: Int = 0
	while true {
		guard h1 != nil || h2 != nil else {
			if carry != 0 {
				addNode(carry)
			}
			break
		}

		let sum = (h1?.val ?? 0) + (h2?.val ?? 0) + carry
		addNode(sum%10)
		carry = sum/10

		h1 = h1?.next
		h2 = h2?.next
	}

	return list ?? ListNode(0)
}
