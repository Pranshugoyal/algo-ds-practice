
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

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

#https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
def fisherYatesShuffle(nums, n):
    for i in range(n):
        j = random.randrange(i, n)
        nums[i], nums[j] = nums[j], nums[i]
    return nums

class Point:
    def __init__(self, p):
        self.x, self.y = p

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @staticmethod
    def pack(self):
        return [self.x, self.y]

#http://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf
#https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
def threePointOrientation(p1, p2, p3):
    s12 = (p2.y - p1.y)*(p3.x - p2.x)
    s23 = (p3.y - p2.y)*(p2.x - p1.x)
    diff = s12 - s23
    if diff > 0:
        #Clockwise
        return 1
    elif diff < 0:
        #Anti-Clockwise
        return -1
    else:
        #Collinear
        return 0

def isPointOnSegment(p, q, P):
    return min(p.x, q.x) <= P.x <= max(p.x, q.x) and min(p.y, q.y) <= P.y <= max(p.y, q.y)

#https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/amp/
def lineSegmentIntersection(p1, q1, p2, q2):
    o1 = threePointOrientation(p1, q1, p2)
    o2 = threePointOrientation(p1, q1, q2)
    o3 = threePointOrientation(p2, q2, p1)
    o4 = threePointOrientation(p2, q2, q1)

    #General Case
    if o1 != o2 and o3 != o4:
        return True

    #special Cases
    if o1 == 0 and isPointOnSegment(p1, q1, p2):
        return True
    if o2 == 0 and isPointOnSegment(p1, q1, q2):
        return True
    if o3 == 0 and isPointOnSegment(p2, q2, p1):
        return True
    if o4 == 0 and isPointOnSegment(p2, q2, q1):
        return True
    else:
        return False

#https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/amp/
def convexHullJarvis(points):
    n = len(points)

    def leftmostPoint():
        left = points[0]
        for point in points:
            if point.x < left.x:
                left = point
            elif point.x == left.x:
                if point.y > left.y:
                    left = point
        return left

    def findNextPoint(p):
        q = None
        for point in points:
            if point != p:
                q = point
                break

        for point in points:
            if point == p or point == q:
                continue
            o = threePointOrientation(p, point, q)
            if o < 0:
                q = point
            elif o == 0:
                if isPointOnSegment(p, q, point):
                    q = point
        return q

    l = leftmostPoint()
    p, hull = findNextPoint(l), [l]
    while p != l:
        hull.append(p)
        p = findNextPoint(p)
    return hull

#https://leetcode.com/problems/erect-the-fence/solution/
def fencePerimeter(trees):
    if len(trees) < 4:
        return trees
    points = list(map(Point, trees))
    fence = convexHullJarvis(points)
    return list(map(Point.pack, fence))

def findPi(n, size):
    from random import randrange as rr
    from math import gcd, pi

    def isCoprime():
        a, b = rr(size) + 1, rr(size) + 1
        return gcd(a, b) == 1

    coprimePairs = 0
    for _ in range(n):
        coprimePairs += 1 if isCoprime() else 0

    return (6*n/coprimePairs)**(0.5)

print(findPi(int(1_00_00 * (pi)**2), 10000))
