//
//  Heap.swift
//  Coding Prep
//
//  Created by Pranshu Goyal on 29/03/21.
//

import Foundation

///Q703. Kth Largest Element in a Stream
class KthLargest {

    let heap: Heap<Int>
    let size: Int

    init(_ k: Int, _ nums: [Int]) {
        heap = Heap([], order: .min)
        size = k
        for (index, value) in nums.enumerated() {
            if index < k {
                heap.insert(value)
            } else {
                _ = add(value)
            }
        }
    }

    @discardableResult
    func add(_ val: Int) -> Int {
        if heap.storage.count < size {
            heap.insert(val)
        } else if val > heap.peek() {
            _ = heap.extractRoot()
            heap.insert(val)
        }
        return heap.peek()
    }
}
