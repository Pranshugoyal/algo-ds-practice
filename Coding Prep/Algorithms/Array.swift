
public func isUnique(_ string: String) -> Bool {
    var map: [Character: Int] = [:]
    for c in string {
       map[c, default: 0] += 1
    }
    for (_, value) in map {
        if value > 1 {
            return false
        }
    }
    return true
}
//print(isUnique("This is false"))

public func reverseArrayInPlace<T>(_ array: inout [T]) {
   guard !array.isEmpty else {
       return
   }

   var left = 0
   var right = array.count - 1

   while left < right {
       let temp = array[right]
       array[right] = array[left]
       array[left] = temp
       left += 1
       right -= 1
   }
}

public func removeDuplicates<T: Hashable>(_ array: [T]) -> [T] {
    var chars: Set<T> = []
    var newArray: [T] = []

    for c in array {
        if !chars.contains(c) {
            newArray.append(c)
            chars.insert(c)
        }
    }
    return newArray
}
//print(removeDuplicates([2,1,2,3,1,5,3,6,7]))

public func binarySearch<T: Comparable>(_ n: Int, in array: [T]) -> Int {
    return 0
}
