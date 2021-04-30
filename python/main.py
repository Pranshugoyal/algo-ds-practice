
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
	
if __name__ == "__main__":
	cases = ["aba", "abe", "ababa", "abcba", "abba"]
	#cases = ["abe"]
	for case in cases:
		print(checkPalindromeWithStack(case))
