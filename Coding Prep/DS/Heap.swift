
public class Heap<T: Comparable> {

    public enum Order {
        case min, max
    }

    public let order: Order
    public var storage: [T]
    
    public convenience init(order: Order = .max) {
        self.init([], order: order)
    }

    public init(_ array: [T], order: Order = .max) {
        self.order = order
        let length = array.count
        storage = array
		let lastParentIndex = parentIndex(of: length-1)
        for i in (0...lastParentIndex).reversed() {
            heapify(on: i)
        }
    }

    func heapify(on index: Int) {

        let left = leftChildIndex(of: index) 
        let right = rightChildIndex(of: index) 
        
        var largest = index
        if left < storage.count && !isInOrder(left, parent: index) {
           largest = left
        } 

        if right < storage.count && !isInOrder(right, parent: largest) {
            largest = right
        }

        if largest != index {
            exchange(index, largest)
            heapify(on: largest)
        }
    }

    public func extractRoot() -> T {
        guard !storage.isEmpty else {
            fatalError("Heap Empty")
        }
        
        let max = peek()
        
        //Replace root with last element and remove last
        storage[0] = storage[storage.count-1]
        storage.removeLast()

        guard !storage.isEmpty else {
            return max
        }

        heapify(on: 0)
        return max
    }

    public func insert(_ key: T) {
        storage.append(key)

        var i = storage.count - 1
        var parent = parentIndex(of: i)
        while i > 0 && !isInOrder(i, parent: parent) {
            exchange(i, parent)
            i = parent
            parent = parentIndex(of: i)
        }
    }

    //Helpers
    func exchange(_ i: Int, _ j: Int) {
        let temp = storage[i]
        storage[i] = storage[j]
        storage[j] = temp
    }

    func isInOrder(_ i: Int, parent: Int) -> Bool {
        if order == .max {
            return storage[parent] >= storage[i]
        } else {
            return storage[parent] <= storage[i]
        }
    }

    //Getters
    public func peek() -> T {
        return storage[0]
    }

    func parentIndex(of i: Int) -> Int {
        return (i-1)/2
    }

    func leftChildIndex(of i: Int) -> Int {
        return 2*i + 1
    }

    func rightChildIndex(of i: Int) -> Int {
        return 2*i + 2
    }
}

