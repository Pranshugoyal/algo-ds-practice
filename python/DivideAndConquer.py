
################################################################################
# ------------------------ Divide & Conquer GfG Must Do ---------------------- #
################################################################################

#https://practice.geeksforgeeks.org/problems/binary-search/1
def binarySearch(arr,k):
	def condition(i):
		return arr[i] >= k
	
	left, right = 0, len(arr)-1
	while left < right:
		mid = left + (right-left)//2
		if condition(mid):
			right = mid
		else:
			left = mid + 1
	return left

def bsCeil(k):
    lo, hi = 0, len(arr)-1
    while lo < hi:
        mid = hi - (hi-lo)//2
        if arr[mid] <= k:
            lo = mid
        else:
            hi = mid-1
    return hi if arr[hi] == k else -1

#https://practice.geeksforgeeks.org/problems/find-the-element-that-appears-once-in-sorted-array/0
def findOnce(arr : list, n : int):
	def condition(i) -> bool:
		if i == n-1:
			return arr[i] == arr[i-1]
		elif i == 0:
			return arr[i] != arr[i+1]

		if arr[i] != arr[i-1] and arr[i] != arr[i+1]:
			return True

		if i%2 == 0:
			return arr[i] == arr[i-1]
		else:
			return arr[i] == arr[i+1]
	
	left, right = 0,n-1
	while left < right:
		mid = left + (right-left)//2
		if condition(mid):
			right = mid
		else:
			left = mid+1
	return arr[left]

def findRotationInSortedArray(arr):
	def condition(i) -> bool:
		return arr[i] < arr[0]
	
	left,right = 0, len(arr)-1
	while left < right:
		mid = left + (right-left)//2
		if condition(mid):
			right = mid
		else:
			left = mid+1
	return left

#https://practice.geeksforgeeks.org/problems/search-in-a-rotated-array/0
def findNumberInSortedRotatedArray(arr,k):
	rotationIndex = findRotationInSortedArray(arr)
	leftSearch = binarySearch(arr[:rotationIndex],k)
	if arr[leftSearch] == k:
		return leftSearch
	
	rightSearch = binarySearch(arr[rotationIndex:],k) + rotationIndex
	if arr[rightSearch] == k:
		return rightSearch

	return -1

def mergeSort(arr,l,r):
	if r == l:
		return
	
	mid = l + (r-l)//2
	mergeSort(arr,l,mid)
	mergeSort(arr,mid+1,r)
	
	i,j = l,mid+1
	temp = []
	while i <= mid and j <= r:
		if arr[i] <= arr[j]:
			temp.append(arr[i])
			i += 1
		else:
			temp.append(arr[j])
			j += 1
	
	while i <= mid:
		temp.append(arr[i])
		i += 1

	while j <= r:
		temp.append(arr[j])
		j += 1
	
	i = l
	for n in temp:
		arr[i] = n
		i += 1

################################################################################
# ----------------------------- Leetcode Practice ---------------------------- #
################################################################################

#https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
def kthSmallestMatrixSaddleBackSearch(matrix, k):
    def countLessThanK(n, k):
        r, c = 0, n-1
        count = 0
        while r < n and c >= 0:
            if matrix[r][c] <= k:
                count += c+1
                r += 1
            else:
                c -=1
        return count

    #Binary search for ans
    n = len(matrix)
    lo, hi = matrix[0][0], matrix[-1][-1]
    while lo < hi:
        mid = lo + (hi-lo)//2
        count = countLessThanK(n, mid)
        if count < k:
            lo = mid + 1
        else:
            hi = mid
    return lo

#https://leetcode.com/problems/find-k-th-smallest-pair-distance/
def kthSmallestPairDistance(nums, k):
    nums.sort()
    def countPairsWithDiffLessThan(d):
        i, j = 0,1
        count = 0
        while j < len(nums):
            while nums[j] - nums[i] > d and i <= j:
                i += 1
            count += j - i
            j += 1
            if count >= k:
                return True

        return False

    lo, hi = 0, nums[-1]-nums[0]
    while lo < hi:
        mid = lo + (hi-lo)//2
        if countPairsWithDiffLessThan(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

#https://www.spoj.com/problems/EKO/
def getBladeHeight(heights, n, M):
    heights.sort()
    prefix = [0]*n
    prefix[-1] = heights[-1]
    for i in reversed(range(n-1)):
        prefix[i] = heights[i] + prefix[i+1]

    def countWoodBS(h):
        lo, hi = 0, n-1
        while lo < hi:
            mid = lo + (hi-lo)//2
            if heights[mid] > h:
                hi = mid
            else:
                lo = mid+1
        
        return prefix[lo] - (n-lo)*h

    def countWood(h):
        w = 0
        for th in heights:
            w += max(th-h, 0)
        return w

    lo, hi = 0, max(heights) 
    while lo < hi:
        mid = lo + (hi-lo)//2
        if countWoodBS(mid) < M:
            hi = mid
        else:
            lo = mid + 1

    return lo - 1

#https://www.geeksforgeeks.org/find-a-peak-in-a-given-array/
def findPeak(arr, lo, hi):
    if lo == hi:
        return lo
    elif hi-lo == 1:
        return hi if arr[hi] > arr[lo] else lo

    mid = lo + (hi-lo)//2
    if arr[mid-1] < arr[mid] > arr[mid+1]:
        return mid
    elif arr[mid] < arr[mid+1]:
        return findPeak(arr, mid+1, hi)
    else:
        return findPeak(arr, lo, mid-1)

#https://leetcode.com/problems/find-a-peak-element-ii/
#https://www.geeksforgeeks.org/find-peak-element-2d-array/
def peakFinding2D(mat, lo, hi):
    def maxRowIndex(row):
        return row.index(max(row))

    if lo == hi:
        return [lo, maxRowIndex(mat[lo])]
    elif hi-lo == 1:
        mi = maxRowIndex(mat[lo])
        if mat[lo][mi] > mat[hi][mi]:
            return [lo, mi]
        else:
            return [hi, maxRowIndex(mat[hi])]

    mid = lo + (hi-lo)//2
    mi = maxRowIndex(mat[mid])
    if mat[mid-1][mi] < mat[mid][mi] > mat[mid+1][mi]:
        return [mid, mi]
    elif mat[mid-1][mi] > mat[mid][mi]:
        return peakFinding2D(mat, lo, mid-1)
    else:
        return peakFinding2D(mat, mid+1, hi)

#https://practice.geeksforgeeks.org/problems/count-squares3649/1
def countSquares(N):
    lo, hi = 1, N
    while lo < hi:
        mid = lo + (hi-lo)//2
        if mid**2 >= N:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1

#https://www.geeksforgeeks.org/move-negative-numbers-beginning-positive-end-constant-extra-space/
def partitionOnZero(nums):
    i = 0
    for j in range(len(nums)):
        if nums[j] < 0:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1

#https://www.geeksforgeeks.org/quick-sort/
def quickSort(nums, l, h):
    if l >= h:
        return

    i, p = l, h-1
    for j in range(l, h):
        if nums[j] < nums[p]:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[p] = nums[p], nums[i]
    quickSort(nums, l, i)
    quickSort(nums, i+1, h)

#https://www.geeksforgeeks.org/k-th-element-two-sorted-arrays/
def kthElementInMergedSortedArrays(a, b, k):
    l, r = max(0, k-len(b)), min(k, len(a))
    while l < r:
        mid = r - (r-l)//2
        if k-mid >= len(b) or a[mid-1] <= b[k-mid]:
            l = mid
        else:
            r = mid - 1

    if r == k:
        return a[k-1]
    elif r == 0:
        return b[k-1]
    else:
        return max(a[r-1], b[k-r-1])
