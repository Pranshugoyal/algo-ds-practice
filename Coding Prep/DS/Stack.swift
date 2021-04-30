//
//  Stack.swift
//  Coding Prep
//
//  Created by Pranshu Goyal on 28/04/21.
//

import Foundation

class Stack<T> {
	
	var storage: [T] = []

	var isEmpty: Bool {
		return storage.isEmpty
	}

	func push(_ element: T) {
		storage.append(element)
	}

	func pop() -> T? {
		return storage.removeLast()
	}
}
