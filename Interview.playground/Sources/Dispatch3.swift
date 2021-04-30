
protocol WoodenObject {
	func whatAmI()
}

extension WoodenObject {
	func whatAmI() {
		print("WoodenObject")
	}
}

class Tree: WoodenObject {
	init() {
		self.whatAmI()
	}
}

class Table: Tree {
	func whatAmI() {
		print("Table")
	}
}

public func testDispatch3() {
	Table().whatAmI()
}

public func testWhatAmI() {
    let x: Tree = Table()
    x.whatAmI()
}
