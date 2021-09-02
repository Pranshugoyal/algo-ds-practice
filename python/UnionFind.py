
class UnionFind:

    def __init__(self, sets=[]):
        self.map = {}
        for i in sets:
            self.add(i)

    def add(self, s):
        self.map[s] = s

    def find(self, s):
        p = s
        while self.map[p] != p:
            p = self.map[p]
        return p

    def union(self, s1, s2):
        p1 = self.find(s1)
        p2 = self.find(s2)

        if p1 == p2:
            return

        self.map[p2] = p1

    def areDisjoint(self, s1, s2):
        return self.find(s1) != self.find(s2)

#https://leetcode.com/problems/array-nesting/
def arrayNesting(nums):
    def findLoop(k):
        count = 0
        while nums[k] is not None:
            nums[k], k = None, nums[k]
            count += 1
        return count

    maxLen = 1
    for i in range(len(nums)):
        maxLen = max(maxLen, findLoop(i))
    return maxLen
