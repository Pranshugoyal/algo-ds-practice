import Dispatch

public func testDispatchQueue() {
    
    DispatchQueue.global().sync {
        print("a")
    }
    
    DispatchQueue.global().async {
        print("b")
    }
    
    DispatchQueue.global().sync {
        print("c")
    }
    
    DispatchQueue.main.async {
        
        print("d")
        
        DispatchQueue.global().async {
            print("e")
        }
        
        DispatchQueue.main.sync {
            print("f")
        }
    }
    
    print("g")
}
