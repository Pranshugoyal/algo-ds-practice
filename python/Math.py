
#https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/amp/
def modInverse(a, m):
    m0 = m
    y = 0
    x = 1
 
    if (m == 1):
        return 0
 
    while (a > 1):
        # q is quotient
        q = a // m
 
        t = m
 
        # m is remainder now, process
        # same as Euclid's algo
        m = a % m
        a = t
        t = y
 
        # Update x and y
        y = x - q * y
        x = t
 
    # Make x positive
    if (x < 0):
        x = x + m0
 
    return x

#https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
def fisherYatesShuffle(nums, n):
    for i in range(n):
        j = random.randrange(i, n)
        nums[i], nums[j] = nums[j], nums[i]
    return nums