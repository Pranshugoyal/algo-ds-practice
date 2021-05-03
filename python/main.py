
def checkPalindrome(string):
	count = len(string)
	for i in range(0, count//2):
		if string[i] != string[count-1-i]: 
			return False
	return True

def checkPalindromeWithStack(string):
	stack = []
	count = len(string)
	for i in range(0, count//2):
		stack.append(string[i])
	
	#print(stack)
	half = 0
	if count%2 == 0:
		half = count//2
	else:
		half = count//2 + 1
	
	for i in range(half, count):
		#print(stack, i, string[i])
		if string[i] != stack.pop():
			return False
	
	return True 

def calculateSpan(price):
	 
	n = len(price)
	# Create a stack and push index of fist element to it
	st = []
	st.append(0)
 
	# Span value of first element is always 1
	S = [0 for i in range(len(price)+1)]
	S[0] = 1
 
	# Calculate span values for rest of the elements
	for i in range(1, n):
		 
		# Pop elements from stack whlie stack is not
		# empty and top of stack is smaller than price[i]
		while( len(st) > 0 and price[st[-1]] <= price[i]):
			st.pop()
 
		# If stack becomes empty, then price[i] is greater
		# than all elements on left of it, i.e. price[0],
		# price[1], ..price[i-1]. Else the price[i] is
		# greater than elements after top of stack
		S[i] = i + 1 if len(st) <= 0 else (i - st[-1])
 
		# Push this element to stack
		st.append(i)
	
	return S[:-1]

def getSpan(prices):
	n = len(prices)
	print(prices)
	span = [1]
	stack = [0]

	for i in range(1,n):
		while len(stack) > 0 and prices[stack[-1]] <= prices[i]:
			stack.pop()

		if len(stack) == 0:
			span.append(i+1)
		else:
			span.append(i - stack[-1])

		stack.append(i)
		print(i, stack)

	return span
	
if __name__ == "__main__":
	#cases = ["aba", "abe", "ababa", "abcba", "abba"]
	#cases = ["abe"]
	#for case in cases:
		#print(checkPalindromeWithStack(case))
	#print(calculateSpan([100,80,60,70,60,75,85]))
	print(getSpan([100,80,60,70,60,75,85]))
