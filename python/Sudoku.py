
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
            oldCount = len(self.pos)
            self.pos -= taken
            if len(self.pos) == 1:
                self.val = list(self.pos)[0]
            return len(self.pos) != oldCount

class Sudoku:

    def __init__(self, data):
        self.data = data

    @staticmethod
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
        return Sudoku(sudoku)

    def __str__(self):
        repr = ""
        for row in self.data:
            repr += str([str(cell) for cell in row]) + "\n"
        repr += "Filled: " + str(self.fillCount())
        return repr

    def __copy__(self):
        new = []
        for row in self:
            new.append([cell.copy() for cell in row])
        return Sudoku(new)

    def fillCount(self):
        count = 0
        for row in self.data:
            for cell in row:
                if cell.val is not None:
                    count += 1
        return count

    def isValid(self):
        def rowCheck():
            for row in self.data:
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
                    val = self.data[row][column].val 
                    if val not in values:
                        values.add(val)
                    else:
                        return False
            return True

        def blockCheck():
            for r, c in Sudoku.blockGenerator(0, 0):
                values = set()
                for i, j in Sudoku.blockGenerator(r, c):
                    val = self.data[i][j].val
                    if val not in values:
                        values.add(val)
                    else:
                        return False
            return True

        return rowCheck() and columnCheck() and blockCheck()

    @staticmethod
    def blockGenerator(r, c):
        for i in range(r*3, (r+1)*3):
            for j in range(c*3, (c+1)*3):
                yield i, j

    def updatePass(self):
        sudoku = self.data
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
            for (br, bc) in Sudoku.blockGenerator(0,0):
                block = [sudoku[r][c] for r,c in Sudoku.blockGenerator(br, bc)]
                taken = set([cell.val for cell in block])
                for cell in block:
                    updates = cell.removePos(taken) or updates 
            return updates

        updates = False
        updates = updates or updateRows()
        updates = updates or updateColumns()
        updates = updates or updateBlocks()
        return updates

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
            if val == "x" or val == ".":
                sudoku[-1].append(Cell())
            else:
                sudoku[-1].append(Cell(int(val)))
    return Sudoku(sudoku)
