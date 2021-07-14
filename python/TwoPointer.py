
#https://www.interviewbit.com/problems/intersection-of-sorted-arrays/
def arrayIntersection(A, B):
    a, b = 0, 0
    res = []
    while a < len(A) and b < len(B):
        if A[a] == B[b]:
            res.append(A[a])
            a += 1
            b += 1
        elif A[a] < B[b]:
            a += 1
        else:
            b += 1
    return res
