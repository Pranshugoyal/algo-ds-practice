
################################################################################
# ------------------------------ Interval Operations ------------------------- #
################################################################################

#https://www.geeksforgeeks.org/interval-tree/
class IntervalTree:

    def __init__(self, interval):
        self.interval = interval
        self.left, self.right = None, None
        self.high = interval[1]

    def _overlaps(self, interval):
        s1, e1 = interval[0], interval[1]
        s2, e2 = self.interval[0], self.interval[1]
        return (s2 < s1 < e2) or (s1 < s2 < e1)

    def insert(self, interval):
        self.high = max(self.high, interval[1])
        if interval[0] < self.interval[0]:
            if self.left:
                self.left.insert(interval)
            else:
                self.left = IntervalTree(interval)
        else:
            if self.right:
                self.right.insert(interval)
            else:
                self.right = IntervalTree(interval)

    def findOverlaps(self, interval):
        #Find if this interval conflicts with any of the existing ones
        if self._overlaps(interval):
            return self.interval
        elif self.left and self.left.max > interval[0]:
            return self.left.findOverlaps(interval)
        elif self.right:
            return self.right.findOverlaps(interval)
        else:
            return None

#https://www.geeksforgeeks.org/merging-intervals/
def mergeIntervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in intervals[1:]:
        s, e = i[0], i[1]
        ls, le = merged[-1][0], merged[-1][1]
        if s > le:
            merged.append(i)
        else:
            merged[-1] = [ls, max(e, le)]
    return merged

#https://practice.geeksforgeeks.org/problems/sick-pasha0323/1
#https://www.geeksforgeeks.org/count-the-number-of-intervals-in-which-a-given-value-lies/?ref=rp
def processIntervalFrequencies(intervals):
    start, end = intervals[0][0], intervals[0][1]
    for interval in intervals:
        start = min(start, interval[0])
        end = max(end, interval[1])

    freq = [0]*(end-start+2)
    s,e = 0,0
    for i in intervals:
        s, e = i[0], i[1]
        freq[s-start] += 1
        freq[e-start+1] -= 1

    freq.pop()
    for i in range(1, len(freq)):
        freq[i] += freq[i-1]

    return (start, freq)

