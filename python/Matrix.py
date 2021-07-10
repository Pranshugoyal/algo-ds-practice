
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
