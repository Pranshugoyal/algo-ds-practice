//
//  Trees.swift
//  LeetCode
//
//  Created by Pranshu Goyal on 12/04/20.
//  Copyright © 2020 Pranshu Goyal. All rights reserved.
//

final class TreeNode {
	var val: Int
	var left: TreeNode?
	var right: TreeNode?

	init(_ val: Int) {
		self.val = val
		self.left = nil
		self.right = nil
	}
}

func diameterOfBinaryTree(_ tree: TreeNode?) -> Int {
	var ans = 1

	@discardableResult
	func depth(_ tree: TreeNode?) -> Int {
		guard let tree = tree else {
			return 0
		}

		let l = depth(tree.left)
		let r = depth(tree.right)
		ans = max(ans, l+r+1)

		return max(depth(tree.left), depth(tree.right)) + 1
	}

	depth(tree)
	return ans - 1
}

func isValidSequence(_ root: TreeNode?, _ arr: [Int]) -> Bool {
	guard let root = root else {
		return arr.isEmpty
	}

	if root.val != arr.first {
		return false
	}

	if arr.count <= 1 && (root.left != nil || root.right != nil) {
		return false
	}

	let suffix = Array(arr[1...])
	return isValidSequence(root.left, suffix) || isValidSequence(root.right, suffix)
}
