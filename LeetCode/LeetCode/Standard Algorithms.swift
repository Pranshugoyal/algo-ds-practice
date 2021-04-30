//
//  Standard Algorithms.swift
//  LeetCode
//
//  Created by Pranshu Goyal on 13/04/20.
//  Copyright © 2020 Pranshu Goyal. All rights reserved.
//

extension Array {
	/// Use Binary Search to find the index of a new element to be inserted in a sorted array.
    func insertionIndexOf(_ elem: Element, isOrderedBefore: (Element, Element) -> Bool) -> Int {
        var lo = 0
        var hi = self.count - 1
        while lo <= hi {
            let mid = (lo + hi)/2
            if isOrderedBefore(self[mid], elem) {
                lo = mid + 1
            } else if isOrderedBefore(elem, self[mid]) {
                hi = mid - 1
            } else {
                return mid // found at position mid
            }
        }
        return lo // not found, would be inserted at position lo
    }
}

class LRUCache {
	class ListNode {
		var next: ListNode?, prev: ListNode?
		var key: Int, val: Int
		init(_ key: Int, _ val: Int) {
			self.key = key
			self.val = val
		}
	}

    let capacity: Int
    // define head and tail to locate the bottom and top of the linked list
    let head = ListNode(0,0), tail = ListNode(0,0)
    var dict = [Int:ListNode]()

    init(_ capacity: Int) {
        self.capacity = capacity
        head.next = tail
        tail.prev = head
    }

    func get(_ key: Int) -> Int {
        if let listNode = dict[key] {
            //move to end
            moveToBottom(listNode)
            return listNode.val
        }
        return -1
    }

    func put(_ key: Int, _ value: Int) {
        if let existedListNode = dict[key] {
            existedListNode.key = key
            existedListNode.val = value
            //move to end
            moveToBottom(existedListNode)
        }
        else {
            let newListNode = ListNode(key, value)
            freeSpaceIfNeeded()
            // move to end
            moveToBottom(newListNode)
            dict[key] = newListNode
        }
    }

    private func freeSpaceIfNeeded() {
        if let toBeRemoved = head.next, dict.count == capacity {
            head.next = toBeRemoved.next
            toBeRemoved.next?.prev = head
            dict[toBeRemoved.key] = nil
        }
    }

    private func moveToBottom(_ listNode: ListNode) {
        // remove current listNode from linked chain
        let prevListNode = listNode.prev
        let nextListNode = listNode.next
        prevListNode?.next = nextListNode
        nextListNode?.prev = prevListNode

        // shift current listNode to last position(previous of tail)
        let lastListNode = tail.prev
        // connect with last ListNode
        lastListNode?.next = listNode
        listNode.prev = lastListNode
        // connect with tail ListNode
        listNode.next = tail
        tail.prev = listNode
    }
}
