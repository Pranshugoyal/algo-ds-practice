from typing import List

class Solution:
	def ways(self, pizza: List[str], k: int) -> int:
		self.LIMIT = 1000000007

		#if len(pizza) == 50 and len(pizza[0]) == 50 and k == 8:
			#return 370641831980%self.LIMIT

		self.pizza = pizza
		self.map = {}
		self.countmap = {}
		totalApples = self.countApplesInRange((0,0),(len(pizza), len(pizza[0])))
		return self.findWays((0,0), (len(pizza), len(pizza[0])), k, totalApples)

	def findWays(self, start: (int, int), end: (int, int), k: int, totalApples: int) -> int:
		#print("Check:", start, end, k, totalApples)
		if k == 1:
			#print("Return:", start, end, "base:", 1)
			return 1 if totalApples > 0 else 0

		if (start, end, k) in self.map:
			#print("Cache hit:", start, end, k, totalApples)
			return self.map[(start, end, k)]

		totalWays = 0

		#Row cuts
		rows = end[0] - start[0]
		#print("Number of rows:", rows)
		if rows > 1: 
			for size in range(1,rows):
				r = self.isRowCutValid(start, end, size, k, totalApples) 
				if r > 0:
					ways = self.findWays((start[0]+size, start[1]), end, k-1, r)
					totalWays += ways
					#print("row cut size:", size, "in:", start, end, "ways:", ways)

		#Column cuts
		columns = end[1] - start[1]
		#print("Number of columns:", columns)
		if columns > 1: 
			for size in range(1,columns):
				r = self.isColumnCutValid(start, end, size, k, totalApples) 
				if r > 0:
					ways = self.findWays((start[0], start[1]+size), end, k-1, r)
					totalWays += ways
					#print("Column cut size:", size, "in:", start, end, "ways:", ways)

		#print("Return:", start, end, "Ways:", totalWays)
		self.map[(start, end, k)] = totalWays%self.LIMIT
		return self.map[(start, end, k)]

	def isRowCutValid(self, start: (int, int), end: (int, int), size: int, k: int, totalApples: int) -> int:
		return self.isCutValid(start, (start[0]+size, end[1]), k, totalApples)

	def isColumnCutValid(self, start: (int, int), end: (int, int), size: int, k: int, totalApples: int) -> int:
		return self.isCutValid(start, (end[0], start[1]+size), k, totalApples)

	def isCutValid(self, start: (int, int), end: (int, int), k: int, totalApples: int) -> int:
		cutApples = self.countApplesInRange(start, end)

		if cutApples == 0:
			return -1

		remainingApples = totalApples - cutApples
		if remainingApples >= k - 1:
			return remainingApples
		else:
			return -1

	def countApplesInRange(self, start: (int, int), end: (int, int)) -> int:
		if (start, end) in self.countmap:
			#print("Count cache hit:", start, end)
			return self.countmap[(start, end)]

		count = 0
		for row in self.pizza[start[0]:end[0]]:
			for c in row[start[1]:end[1]]:
				if c == "A":
					count += 1	
		self.countmap[(start, end)] = count
		return count
#class end

list = ["..A.A.AAA...AAAAAA.AA..A..A.A......A.AAA.AAAAAA.AA",
		"A.AA.A.....AA..AA.AA.A....AAA.A........AAAAA.A.AA.",
		"A..AA.AAA..AAAAAAAA..AA...A..A...A..AAA...AAAA..AA",
		"....A.A.AA.AA.AA...A.AA.AAA...A....AA.......A..AA.",
		"AAA....AA.A.A.AAA...A..A....A..AAAA...A.A.A.AAAA..",
		"....AA..A.AA..A.A...A.A..AAAA..AAAA.A.AA..AAA...AA",
		"A..A.AA.AA.A.A.AA..A.A..A.A.AAA....AAAAA.A.AA..A.A",
		".AA.A...AAAAA.A..A....A...A.AAAA.AA..A.AA.AAAA.AA.",
		"A.AA.AAAA.....AA..AAA..AAAAAAA...AA.A..A.AAAAA.A..",
		"A.A...A.A...A..A...A.AAAA.A..A....A..AA.AAA.AA.AA.",
		".A.A.A....AAA..AAA...A.AA..AAAAAAA.....AA....A....",
		"..AAAAAA..A..A...AA.A..A.AA......A.AA....A.A.AAAA.",
		"...A.AA.AAA.AA....A..AAAA...A..AAA.AAAA.A.....AA.A",
		"A.AAAAA..A...AAAAAAAA.AAA.....A.AAA.AA.A..A.A.A...",
		"A.A.AA...A.A.AA...A.AA.AA....AA...AA.A..A.AA....AA",
		"AA.A..A.AA..AAAAA...A..AAAAA.AA..AA.AA.A..AAAAA..A",
		"...AA....AAAA.A...AA....AAAAA.A.AAAA.A.AA..AA..AAA",
		"..AAAA..AA..A.AA.A.A.AA...A...AAAAAAA..A.AAA..AA.A",
		"AA....AA....AA.A......AAA...A...A.AA.A.AA.A.A.AA.A",
		"A.AAAA..AA..A..AAA.AAA.A....AAA.....A..A.AA.A.A...",
		"..AA...AAAAA.A.A......AA...A..AAA.AA..A.A.A.AA..A.",
		".......AA..AA.AAA.A....A...A.AA..A.A..AAAAAAA.AA.A",
		".A.AAA.AA..A.A.A.A.A.AA...AAAA.A.A.AA..A...A.AAA..",
		"A..AAAAA.A..A..A.A..AA..A...AAA.AA.A.A.AAA..A.AA..",
		"A.AAA.A.AAAAA....AA..A.AAA.A..AA...AA..A.A.A.AA.AA",
		".A..AAAA.A.A.A.A.......AAAA.AA...AA..AAA..A...A.AA",
		"A.A.A.A..A...AA..A.AAA..AAAAA.AA.A.A.A..AA.A.A....",
		"A..A..A.A.AA.A....A...A......A.AA.AAA..A.AA...AA..",
		".....A..A...A.A...A..A.AA.A...AA..AAA...AA..A.AAA.",
		"A...AA..A..AA.A.A.AAA..AA..AAA...AAA..AAA.AAAAA...",
		"AA...AAA.AAA...AAAA..A...A..A...AA...A..AA.A...A..",
		"A.AA..AAAA.AA.AAA.A.AA.A..AAAAA.A...A.A...A.AA....",
		"A.......AA....AA..AAA.AAAAAAA.A.AA..A.A.AA....AA..",
		".A.A...AA..AA...AA.AAAA.....A..A..A.AA.A.AA...A.AA",
		"..AA.AA.AA..A...AA.AA.AAAAAA.....A.AA..AA......A..",
		"AAA..AA...A....A....AA.AA.AA.A.A.A..AA.AA..AAA.AAA",
		"..AAA.AAA.A.AA.....AAA.A.AA.AAAAA..AA..AA.........",
		".AA..A......A.A.AAA.AAAA...A.AAAA...AAA.AAAA.....A",
		"AAAAAAA.AA..A....AAAA.A..AA.A....AA.A...A.A....A..",
		".A.A.AA..A.AA.....A.A...A.A..A...AAA..A..AA..A.AAA",
		"AAAA....A...A.AA..AAA..A.AAA..AA.........AA.AAA.A.",
		"......AAAA..A.AAA.A..AAA...AAAAA...A.AA..A.A.AA.A.",
		"AA......A.AAAAAAAA..A.AAA...A.A....A.AAA.AA.A.AAA.",
		".A.A....A.AAA..A..AA........A.AAAA.AAA.AA....A..AA",
		".AA.A...AA.AAA.A....A.A...A........A.AAA......A...",
		"..AAA....A.A...A.AA..AAA.AAAAA....AAAAA..AA.AAAA..",
		"..A.AAA.AA..A.AA.A...A.AA....AAA.A.....AAA...A...A",
		".AA.AA...A....A.AA.A..A..AAA.A.A.AA.......A.A...A.",
		"...A...A.AA.A..AAAAA...AA..A.A..AAA.AA...AA...A.A.",
		"..AAA..A.A..A..A..AA..AA...A..AA.AAAAA.A....A..A.A"]

if __name__ == "__main__":
	#for row in list:
		#print(row)
	print(Solution().ways(list,8))
