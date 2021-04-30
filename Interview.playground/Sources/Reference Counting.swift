
class A {
    unowned let b: B
    
    init(beta: B) {
        b = beta
    }
}

class B {
    let i: Int = 10
}

public func testReferenceCount() {
    let a = A(beta: B())
    print(a.b.i)
}
