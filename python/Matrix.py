
#https://www.geeksforgeeks.org/find-a-specific-pair-in-matrix/amp/
def findMatrixPair(M, n):
    maxM = [[0]*n for i in range(n)]
    maxM[-1][-1] = M[-1][-1]
    res = None
    
    #last row
    for c in reversed(range(n-1)):
        maxM[-1][c] = max(M[-1][c], maxM[-1][c+1])

    #last column
    for r in reversed(range(n-1)):
        maxM[r][-1] = max(M[r][-1], maxM[r+1][-1])

    #remaining matrix
    for r in reversed(range(n-1)):
        for c in reversed(range(n-1)):
            if res is not None:
                res = max(res, maxM[r+1][c+1] - M[r][c])
            else:
                res = maxM[r+1][c+1] - M[r][c]
            maxM[r][c] = max(M[r][c], maxM[r+1][c], maxM[r][c+1])
    return res

#https://www.geeksforgeeks.org/rotate-a-matrix-by-90-degree-in-clockwise-direction-without-using-any-extra-space/
def rotateMatrixAnticlockwise(M):
    n = len(M)
    for x in range(n//2):
        for y in range(x, n-x-1):
            temp = M[x][y]
            M[x][y] = M[y][n-1-x]
            M[y][n-1-x] = M[n-1-x][n-1-y]
            M[n-1-x][n-1-y] = M[n-1-y][x]
            M[n-1-y][x] = temp
    return M

#https://leetcode.com/problems/range-sum-query-2d-immutable/
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        n, m = len(matrix), len(matrix[0])
        P = [[0]*(m+1) for _ in range(n+1)]
        for r in range(1, n+1):
            for c in range(1, m+1):
                P[r][c] = P[r-1][c] + P[r][c-1] + matrix[r-1][c-1] - P[r-1][c-1]
        self.presums = P

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        P = self.presums
        OD = P[row2+1][col2+1]
        OB = P[row1][col2+1]
        OC = P[row2+1][col1]
        OA = P[row1][col1]
        return OD - OB - OC + OA

#https://leetcode.com/problems/matrix-block-sum/
def matrixBlockSum(mat, k):
    def calculatePresums(matrix):
        n, m = len(matrix), len(matrix[0])
        P = [[0]*(m+1) for _ in range(n+1)]
        for r in range(1, n+1):
            for c in range(1, m+1):
                P[r][c] = P[r-1][c] + P[r][c-1] + matrix[r-1][c-1] - P[r-1][c-1]
        return P

    def sumRegion(P, row1: int, col1: int, row2: int, col2: int):
        OD = P[row2+1][col2+1]
        OB = P[row1][col2+1]
        OC = P[row2+1][col1]
        OA = P[row1][col1]
        return OD - OB - OC + OA

    n, m = len(mat), len(mat[0])
    P = calculatePresums(mat)
    blockSums = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            il, ih = max(i-k, 0), min(i+k, n-1)
            jl, jh = max(j-k, 0), min(j+k, m-1)
            blockSums[i][j] = sumRegion(P, il, jl, ih, jh)
    return blockSums
