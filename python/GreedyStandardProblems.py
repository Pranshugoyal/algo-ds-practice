
################################################################################
# ---------------------------- Interval Selection ---------------------------- #
################################################################################

#https://www.geeksforgeeks.org/activity-selection-problem-greedy-algo-1/
def intervalSelection(start, finish) -> int:
	n = len(start)
	if n <= 1:
		return n

	intervals = []
	for i in range(n):
		intervals.append((start[i], finish[i]))

	intervals.sort(key=lambda x: x[1])
	selected = [intervals[0]]
	for i in range(1,n):
		if intervals[i][0] >= selected[-1][1]:
			selected.append(intervals[i])

	return len(selected)

#https://www.geeksforgeeks.org/job-sequencing-problem/
def jobSequencingProblem(jobs) -> int:
	n = len(jobs)
	if n <= 1:
		return sum(jobs)

	def findSlot(slots,job) -> int:
		for slot in range(job[1]-1,-1,-1):
			if slots[slot] == None:
				return slot
		return -1

	jobs.sort(key=lambda x: x[2], reverse=True)
	chosen = [None for x in range(n)]
	for job in jobs:
		#find slot for this job
		slot = findSlot(chosen, job)
		if slot != -1:
			chosen[slot] = job[0]

	final = []
	for job in chosen:
		if job != None:
			final.append(job)
	return final

jobs = [['a', 2, 100],  # Job Array
       ['b', 1, 19],
       ['c', 2, 27],
       ['d', 1, 25],
       ['e', 3, 15]]
print(jobSequencingProblem(jobs))
