
class Animal {
	func test() {
		print("Animal")
	}

	func call() {
		test()
	}
}

class Mammal: Animal {
	override func test() {
		print("Mammal")
	}
}

class Horse: Mammal {
	override func test() {
		print("Horse")
	}
}

public func testDispatch2() {
	let b: Animal = Horse()
	b.call()
}

//Animal
