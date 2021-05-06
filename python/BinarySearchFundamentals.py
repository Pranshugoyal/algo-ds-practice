
#
# BinarySearchFundamentals
#

def binarySearch(list, v):
	if len(list) == 0:
		return None
	
	lo = 0
	hi = len(list) - 1

	while lo <= hi:
		mid = lo + (hi-lo)//2
		if list[mid] == v:
			return mid
		elif list[mid] < v:
			lo = mid + 1
		else:
			hi = mid - 1
	
	return None

def binarySearchOnReverseSorted(list, v):
	if len(list) == 0:
		return None
	
	lo = 0
	hi = len(list) - 1

	while lo <= hi:
		mid = lo + (hi-lo)//2
		if list[mid] == v:
			return mid
		elif list[mid] > v:
			lo = mid + 1
		else:
			hi = mid - 1
	
	return None

def binarySearchOrderUnknown(list, target):
	if len(list) == 0:
		return None

	if len(list) == 1:
		return 0 if target == list[0] else None

	if list[0] < list[-1]:
		return binarySearch(list, target)
	elif list[0] > list[-1]:
		return binarySearchOnReverseSorted(list, target)
	else:
		return 0 if target == list[0] else None

# Nearly Sorted: Elements might be shifted atmost 1 from their actual position
def binarySearchInNearlySortedArray(list, v):
	if len(list) == 0:
		return None
	
	lo = 0
	hi = len(list) - 1
	while lo <= hi:
		mid = lo + (hi-lo)//2
		if list[mid] == v:
			return mid
		elif mid - 1 >= 0 and list[mid-1] == v:
			return mid - 1
		elif mid + 1 < len(list) and list[mid+1] == v:
			return mid + 1
		elif list[mid] < v:
			if mid + 2 < len(list):
				lo = mid + 2
			else:
				return None
		else:
			if mid - 2 >= 0:
				hi = mid - 2
			else:
				return None

def floorOfTargetInSortedArray(list, v):
	if len(list) == 0:
		return 0
	
	lo = 0
	hi = len(list) - 1
	print(v, list)
	while lo <= hi:
		mid = lo + (hi-lo)//2
		print(lo, mid, hi)
		if list[mid] == v:
			return mid
		elif list[mid] > v:
			hi = mid - 1
		else:
			lo = mid + 1

	if lo > hi:
		return hi	

def firstOccuranceInSortedList(list, v):
	if len(list) == 0:
		return None

	lo = 0
	hi = len(list) - 1
	result = None
	while lo <= hi:
		mid = lo + (hi-lo)//2
		if list[mid] == v:
			result = mid
			hi = mid - 1
		elif list[mid] > v:
			hi = mid - 1
		else:
			lo = mid + 1

	return result

def lastOccuranceInSortedList(list, v):
	if len(list) == 0:
		return None

	lo = 0
	hi = len(list) - 1
	result = None
	while lo <= hi:
		mid = lo + (hi-lo)//2
		if list[mid] == v:
			result = mid
			lo = mid + 1
		elif list[mid] > v:
			hi = mid - 1
		else:
			lo = mid + 1

	return result
	
def countOfTargetInSortedList(list, v):
	if len(list) == 0:
		return 0
	
	firstIndex = firstOccuranceInSortedList(list, v)
	lastIndex = lastOccuranceInSortedList(list, v)
	return lastIndex - firstIndex + 1

# Works when there are no duplicate elements in the array
def rotationValueInSortedArray(list):
	if len(list) == 0:
		return None

	if list[-1] >= list[0]:
		return 0

	size = len(list)
	lo = 0
	hi = len(list) - 1
	while lo <= hi:
		mid = lo + (hi-lo)//2
		print(lo, mid, hi)
		e = list[mid]
		if e > list[(mid+1)%size]:
			return (mid + 1)%size
		elif list[(mid-1+size)%size] > e:
			return mid
		elif e > list[0]:
			#left half is sorted, move to right half
			lo = mid + 1
		else:
			#right half is sorted, move to left half
			hi = mid - 1

def findElementInRotatedSortedArray(list, v):
	if len(list) == 0:
		return None

	rotationIndex = rotationValueInSortedArray(list)
	if rotationIndex == 0:
		return binarySearch(list, v)

	leftResult = binarySearch(list[:rotationIndex], v)
	#print(rotationIndex, leftResult, list, list[:rotationIndex])
	if leftResult is None:
		return binarySearch(list[rotationIndex:], v)
	else:
		return leftResult

if __name__ == "__main__":
	list = [5,10,20,30,40]
	print(floorOfTargetInSortedArray(list, 0))
