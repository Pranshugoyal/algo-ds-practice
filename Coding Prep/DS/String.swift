
public func reverseCStyleString(_ string: String) -> String {
    var newString = ""
    let endIndex = string.index(before: string.endIndex)
    for c in string[..<endIndex].reversed() {
        newString.append(c)
    }
    newString.append(string[endIndex])
    return newString
}
//print(reverseCStyleString("swiftstringT"))

public func areAnagrams(a: String, b: String) -> Bool {
    guard a.count == b.count else {
        return false
    }

    var mapA: [Character: Int] = [:]
    var mapB: [Character: Int] = [:]

    for c in a {
        mapA[c, default: 0] += 1
    }

    for c in b {
        mapB[c, default: 0] += 1
    }

    return mapA == mapB
}
//print(areAnagrams(a: "papaya", b: "paaapyx"))
