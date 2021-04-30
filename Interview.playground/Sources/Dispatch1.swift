
import Foundation

class Base {
    func test() {
        print("A")
    }
}

class Derived: Base {
    override func test() {
        print("B")
    }
}

public func testDispatch1() {
    let b: Base = Derived()
    b.test()
}
