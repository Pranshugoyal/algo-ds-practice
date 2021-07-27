
class Cell:
    def __init__(self, val=None):
        self.val = val
        if val is not None:
            self.pos = set([val])
        else:
            self.pos = set([i+1 for i in range(9)])

    def __str__(self):
        return str(self.val) if self.val is not None else "x"

    def copy(self):
        cell = Cell(self.val)
        cell.pos = self.pos.copy()
        return cell

    def setVal(self, val):
        self.val = val
        self.pos = set([val])

    def removePos(self, taken):
        if self.val is not None:
            return False
        else:
            self.pos -= taken
            if len(self.pos) == 1:
                self.val = list(self.pos)[0]
                return True
            else:
                return False

def solveSudoku(sudoku):
    lastUpdate = True
    while lastUpdate:
        lastUpdate = updatePass(sudoku)

    if fillCount(sudoku) < 81:
        return backtrackSolve(sudoku)
    else:
        return sudoku

def backtrackSolve(sudoku):
    def chooseNext(new):
        minpos = 9
        chosen = None
        for row in range(9):
            for column in range(9):
                lp = len(new[row][column].pos)
                if lp >= 2 and lp < minpos:
                    minpos = lp
                    chosen = (row, column)
            if minpos == 2:
                return chosen
        return chosen

    new = copy(sudoku)
    while fillCount(new) < 81:
        r,c = chooseNext(new)
        new[r][c].val = new[r][c].pos.pop()
        testVal = new[r][c].val
        new = solveSudoku(new)
        if not isValid(new):
            sudoku[r][c].pos.remove(testVal)
            new = copy(sudoku)

def printSudoku(sudoku):
    for row in sudoku:
        print([str(cell) for cell in row])
    print("Filled:", fillCount(sudoku), "\n")

def copy(sudoku):
    new = []
    for row in sudoku:
        new.append([cell.copy() for cell in row])
    return new

def fillCount(sudoku):
    count = 0
    for row in sudoku:
        for cell in row:
            if cell.val is not None:
                count += 1
    return count

def updatePass(sudoku):
    def updateRows():
        updates = False
        for row in sudoku:
            taken = set([cell.val for cell in row])
            for cell in row:
                updates = cell.removePos(taken) or updates 
        return updates

    def updateColumns():
        updates = False
        for c in range(9):
            column = [sudoku[row][c] for row in range(9)]
            taken = set([cell.val for cell in column])
            for cell in column:
                updates = cell.removePos(taken) or updates 
        return updates

    def updateBlocks():
        updates = False
        for (br, bc) in blockGenerator(0,0):
            block = [sudoku[r][c] for r,c in blockGenerator(br, bc)]
            taken = set([cell.val for cell in block])
            for cell in block:
                updates = cell.removePos(taken) or updates 
        return updates

    updates = False
    updates = updates or updateRows()
    updates = updates or updateColumns()
    updates = updates or updateBlocks()
    return updates

def isValid(sudoku):
    def rowCheck():
        for row in sudoku:
            values = set()
            for cell in row:
                if cell.val not in values:
                    values.add(cell.val)
                else:
                    return False
        return True

    def columnCheck():
        for column in range(9):
            values = set()
            for row in range(9):
                val = sudoku[row][column].val 
                if val not in values:
                    values.add(val)
                else:
                    return False

    def blockCheck():
        for r, c in blockGenerator(0, 0):
            values = set()
            for i, j in blockGenerator(r, c):
                val = sudoku[i][j].val
                if val not in values:
                    values.add(val)
                else:
                    return False
        return True

    return rowCheck() and columnCheck() and blockCheck()

def blockGenerator(r, c):
    for i in range(r*3, (r+1)*3):
        for j in range(c*3, (c+1)*3):
            yield i, j

def readSudoku():
    sudoku = []
    for row in range(9):
        s = "Enter row " + str(row+1) + " : "
        data = input(s).split()
        sudoku.append([])
        for val in data:
            if val == "x":
                sudoku[-1].append(Cell())
            else:
                sudoku[-1].append(Cell(int(val)))
    return sudoku

def testSudoku():
    sudoku = []
    arr = [ "xxxx8xx7x",
            "x58x3x1xx",
            "xxxxxxxxx",

            "x26xxxx9x",
            "4xxxxxxx6",
            "7xxx293xx",

            "xx7xxx9xx",
            "1xx2x3xxx",
            "x6xxxxx54"]

    for data in arr:
        sudoku.append([])
        for val in data:
            if val == "x":
                sudoku[-1].append(Cell())
            else:
                sudoku[-1].append(Cell(int(val)))
    return sudoku
