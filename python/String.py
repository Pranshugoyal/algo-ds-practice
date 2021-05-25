
def longestPalindrome(S):
	if len(S) <= 1:
		return S

	def checkPalindrome(s):
		if len(s) == 0:
			return True
		i, j = 0, len(s)-1
		while i <= j:
			if s[i] != s[j]:
				return False
			i += 1
			j -= 1	
		return True
	
	n = len(S)
	lastPalindrome = ""
	for i in range(n):
		minJ = i + len(lastPalindrome) - 1
		for j in range(n-1, minJ, -1):
			if S[i] == S[j] and checkPalindrome(S[i:j+1]):
				if j-i+1 > len(lastPalindrome):
					lastPalindrome = S[i:j+1]
	return lastPalindrome

def longestPalindromeDP(S):
	n = len(S)
	dp = []
	for i in range(n):
		dp.append([None for j in range(n)])
	#dp.append([False for j in range(n)])

	#for i in range(n):
		#dp[i][i] = True
		#if i < n-1:
			#dp[i][i+1] = S[i] == S[i+1]

	def fillDP(i,j, dp, S):
		if i == j:
			return True
		elif j == i+1:
			return S[i] == S[j]
		else:
			return S[i] == S[j] and dp[i+1][j-1]

	#for r in dp:
		#print(r)
	maxL = 1
	ans = (0,0)
	for i in range(n-1,-1,-1):
		for j in range(i,n):
			t = fillDP(i,j, dp, S)
			dp[i][j] = t
			if t and j-i+1 > maxL:
				maxL = j-i+1
				ans = (i,j)
	return S[ans[0]:ans[1]+1]

def checkRotation(a,b,k):
	if len(a) != len(b):
		return False
	
	n = len(a)
	status = True
	for i in range(n):
		ia = (i+k)%n
		iba = (i-k+n)%n
		if a[ia] != b[i] and a[iba] != b[i]:
			return False
	
	return True

def insertionsNeededToMakePalindrome(a):
	if len(a) < 2:
		return 0
	
	n = len(a)
	ra = a[::-1]
	dp = []
	for i in range(n+1):
		dp.append([None for j in range(n+1)])
	
	for i in range(n+1):
		dp[0][i] = 0
		dp[i][0] = 0
	
	for i in range(1,n+1):
		for j in range(1,n+1):
			if ra[j-1] == a[i-1]:
				dp[i][j] = dp[i-1][j-1] + 1
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])

	lcsCount = dp[n][n]
	return n - lcsCount

def longestUniqueSubstring(a):
	if len(a) < 2:
		return a
	
	i, j = 0, 0
	maxLen = 0
	map = {}
	while j < len(a):
		map[a[j]] = map.get(a[j], 0) + 1
		
		while len(map) < j-i+1:
			map[a[i]] -= 1
			if map[a[i]] == 0:
				map.pop(a[i])
			i += 1
		maxLen = max(j-i+1, maxLen)
		j += 1
	return maxLen

def atoi(s):
	def charToI(c):
		n = ord(c)
		if n >= 48 and n <= 57:
			return n - 48
		else:
			return None
	
	multiplier = 1
	n = 0
	if len(s) > 1 and s[0] == "-" and charToI(s[1]) is not None:
		multiplier = -1
		s = s[1:]
	for c in s:
		i = charToI(c)
		if i is not None:
			n = n*10 + i
		else:
			return -1
	return n*multiplier

def strstr(s,x):
	m = len(x)

	for i in range(len(s)-m):
		if s[i:i+m] == x:
			return i
	return -1

def longestCommonPrefix(arr, n):
	minLength = len(arr[0])
	for s in arr:
		minLength = min(minLength, len(s))

	matching = -1
	limitFound = False
	for i in range(minLength):
		c = arr[0][i]
		for s in arr:
			if s[i] != c:
				limitFound = True
				break
		if limitFound:
			break
		else:
			matching = i

	if matching != -1:
		return arr[0][:matching+1]
	else:
		return -1

