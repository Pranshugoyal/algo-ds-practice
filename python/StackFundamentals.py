#
# Stack.py
# https://www.youtube.com/watch?v=J2X70jj_I1o&list=PL_z_8CaSLPWdeOezg68SKkeLN4-T_jNHd

#Nearest greater to left
def ngl(list):
	stack = []
	result = []

	for i in list:
		while len(stack) > 0 and stack[-1] <= i:
			stack.pop()
		
		if len(stack) == 0:
			result.append(None)
		else:
			result.append(stack[-1])

		stack.append(i)	

	return result
	
#Nearest greater to right
def ngr(list):
	stack = []
	result = [None] * len(list)

	for i in range(len(list)-1,-1,-1):
		while len(stack) > 0 and stack[-1] <= list[i]:
			stack.pop()
		
		if len(stack) == 0:
			result[i] = None
		else:
			result[i] = stack[-1]

		stack.append(list[i])	

	return result

#Nearest smaller to right
def nsr(list):
	stack = []
	result = [None] * len(list)

	for i in range(len(list)-1,-1,-1):
		while len(stack) > 0 and stack[-1] >= list[i]:
			stack.pop()
		
		if len(stack) == 0:
			result[i] = None
		else:
			result[i] = stack[-1]

		stack.append(list[i])	

	return result

#Nearest smaller to left
def nsl(list):
	stack = []
	result = [None] * len(list)

	for i in range(0, len(list)):
		while len(stack) > 0 and stack[-1] >= list[i]:
			stack.pop()
		
		if len(stack) == 0:
			result[i] = None
		else:
			result[i] = stack[-1]

		stack.append(list[i])	

	return result

# Stock Span Problem
def stockSpan(list):
	stack = []
	result = [None]*len(list)

	for i in range(0, len(list)):
		while len(stack) > 0 and list[stack[-1]] <= list[i]:
			stack.pop()

		if len(stack) == 0:
			result[i] = i + 1
		else:
			result[i] = i - stack[-1]

		stack.append(i)

	return result

def maxAreaHistogram(list):
	stack = []
	nsl = [-1] * len(list)
	for i in range(0, len(list)):
		while len(stack) > 0 and list[stack[-1]] >= list[i]:
			stack.pop()
		if len(stack) > 0:
			nsl[i] = stack[-1]
		stack.append(i)

	stack = []
	nsr = [len(list)] * len(list)	
	for i in range(len(list)-1,-1,-1):
		while len(stack) > 0 and list[stack[-1]] >= list[i]:
			stack.pop()
		if len(stack) > 0:
			nsr[i] = stack[-1]
		stack.append(i)

	result = []
	for i in range(0, len(list)):
		area = (nsr[i] - nsl[i] - 1) * list[i]
		result.append(area)	

	return max(result)	

def maxAreaBinaryMatrix(M):
	reducedMatrix = [M[0]]
	for row in M[1:]:
		currentRow = []
		for i in range(0,len(row)):
			if row[i] == 1:
				currentRow.append(reducedMatrix[-1][i] + 1)
			else:
				currentRow.append(0)
		reducedMatrix.append(currentRow)

	maxAreas = []
	for row in reducedMatrix:
		maxAreas.append(maxAreaHistogram(row))
	return max(maxAreas)

def rainWaterTapping(list):
	rml = [list[0]]
	for i in list[1:]:
		if i > rml[-1]:
			rml.append(i)
		else:
			rml.append(rml[-1])
	
	rmr = [list[-1]]
	for i in reversed(list):
		if i > rmr[-1]:
			rmr.append(i)
		else:
			rmr.append(rmr[-1])
	rmr.reverse()

	result = []
	for i in range(0, len(list)):
		height = min(rml[i], rmr[i]) - list[i]
		result.append(height)

	return sum(result)

def rainWaterTappingDP(list):
	leftMax = list[0]
	rightMax = list[-1]

	left = 0
	right = len(list) -1
	waterTapped = 0

	while left < right:
		if leftMax < rightMax:
			if list[left] < leftMax:
				waterTapped += leftMax - list[left]
			else:
				leftMax = list[left]
			left += 1
		else:
			if list[right] < rightMax:
				waterTapped += rightMax - list[right]
			else:
				rightMax = list[right]
			right -= 1
	
	return waterTapped

