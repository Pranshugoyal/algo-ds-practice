
public func set0RowsColumns(_ matrix: [[Int]]) -> [[Int]] {
    var rows: Set<Int> = []
    var columns: Set<Int> = []
    for (rowIndex, row) in matrix.enumerated() {
        for (columnIndex, cell) in row .enumerated(){
            if cell == 0 {
                rows.insert(rowIndex)
                columns.insert(columnIndex)
            }
        }
    }

    var result = matrix
    for row in rows {
        result[row] = Array<Int>(repeating: 0, count: matrix[row].count)
    }

    for column in columns {
        for rowIndex in 0..<result.count {
            result[rowIndex][column] = 0
        }
    }

    return result
}

public func rotateMatrix(_ matrix: [[Int]]) -> [[Int]] {
    let size = matrix.count
    var result = matrix
    for row in 0..<size/2 {
        let first = row
        let last = size - row - 1
        for c in first..<last {
            let offset = c - first
            let topLeftTemp = result[first][c]

            // bottomLeft -> topLeft
            result[first][c] = result[last-offset][first]

            // bottomRight -> bottomLeft
            result[last-offset][first] = result[last][last-offset]

            // topRight -> bottomRight
            result[last][last-offset] = result[c][last]

            // topLeft -> topRight
            result[c][last] = topLeftTemp
        }
    }

    return result
}
