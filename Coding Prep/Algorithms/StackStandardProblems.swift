//
//	StackStandardProblems.swift
//  Coding Prep
//
//  Created by Pranshu Goyal on 28/04/21.
//

func parenthesisBalancing(_ string: String) -> Bool {
	let open: Set<Character> = ["{", "(", "["]
	let close: [Character: Character] = ["]": "[", "}": "{", ")": "(",]

	let stack = Stack<Character>()
	for c in string {
		if open.contains(c) {
			stack.push(c)
		} else if let match = close[c] {
			if match != stack.pop() {
				return false
			}
		}
	}

	return stack.isEmpty
}

func infixToPostFix(_ expression: String) -> String {
    print(expression)
	let operators: Set<Character> = ["+", "-", "*", "(", ")"]
	let stack = Stack<Character>()

	var output = ""
	for c in expression {
//        print(c, terminator: ", ")
		if !operators.contains(c) {
			output.append(c)
		} else if c == ")" {
			while !stack.isEmpty, let o = stack.pop() {
				guard o != "(" else {
					break
				}

				output.append(o)
			}
		} else {
			while !stack.isEmpty, let o = stack.pop() {
				guard o != "(" else {
					break
				}
				output.append(o)
            }
            stack.push(c)
		}
//        print(output, stack.storage)
	}

	return output
}
