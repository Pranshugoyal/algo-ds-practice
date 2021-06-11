
################################################################################
# ----------------------- Bit Manipulation GfG Must Do ----------------------- #
################################################################################

def getFirstSetBit(n):
	count = 1
	while n > 0:
		if n%2 == 1:
			return count
		n = n//2
		count += 1
	return -1

def posOfRightMostDiffBit(m,n):
	diff = m^n
	count = 1
	while diff%2 == 0:
		count += 1
		diff >>= 1
	return count

def checkKthBit(n,k):
	mask = 1 << k
	return n & mask != 0

def toggleBits(N , L , R):
	mask = 1 << (L-1)
	for i in range(L,R+1):
		N ^= mask
		mask = mask << 1
	return N

def maxConsecutiveOnes(N):
	maxOnes = 0
	while N > 0:
		if N%2 == 1:
			n = N+1
			n &= -n
			maxOnes = max(maxOnes,n)
		N >>= 1
	n = maxOnes
	count = 0
	while n%2 == 0:
		count += 1
		n //= 2
	return count

print(maxConsecutiveOnes(14))
