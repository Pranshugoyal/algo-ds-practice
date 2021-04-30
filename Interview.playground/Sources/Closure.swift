
import Dispatch

public func testClosure() {
    var a: Int = 0
    
    let c: (() -> Void) = { [a] in
        print(a)
    }
    
    a += 1

    DispatchQueue.main.async(execute: c)
}
