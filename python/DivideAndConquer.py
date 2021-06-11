
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