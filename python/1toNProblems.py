
#
# 1toNProblems
#

def findDuplicateAndMissingNumber(list):
	def swap(i, j):
		(list[i], list[j]) = (list[j], list[i])

	for i in range(len(list)):
		n = list[i]
		while list[n-1] != n:
			swap(i, n-1)
			n = list[i]
	
	return list

if __name__ == "__main__":
	list = [2,11,12,13,14,2,1,1,3,4,3,7,8,6]
	print(findDuplicateAndMissingNumber(list))

