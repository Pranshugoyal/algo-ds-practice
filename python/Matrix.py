
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
