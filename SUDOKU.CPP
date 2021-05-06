#include <stdio.h>
#include <conio.h>

typedef struct /*Declaration*/
{
	int final, value[9];
}
element;

typedef struct
{
	element unit[9][9];
}
sudoku;
sudoku real, fake;

void solve(void);
void rowchk(int, int);
void columnchk(int, int);
void blockchk(int, int);
void row(void);
void column(void);
void block(void);
void hit_trial(void);
int check(void);
void print(void);

void main()
{
	int c, i, j, k;

	for (i = 0; i < 9; i++) /*initiating final to 0*/
		for (j = 0; j < 9; real.unit[i][j].final = 0, j++);

	for (i = 0; i < 9; i++) /*initialising value array*/
		for (j = 0; j < 9; j++)
			for (k = 0; k < 9; real.unit[i][j].value[k++] = k);

	input:
		for (k = 0; k < 9;)
		{
			print();
			printf("\n");
			printf("\nPress n for next,p for previous row or enter if done\n\nRow %d,Enter column number: ", k + 1);

			c = (getch() - 48);
			if (c == 30 || c == 62)
			{
				k++;
				continue;
			}
			else if (c == 32 || c == 64)
			{
				k--;
				continue;
			}
			else if (c == -35)
			{
				k = 9;
				continue;
			}

			printf("%d\nEnter digit to be filled: ", c);
			i = getch() - 48;
			real.unit[k][c - 1].final = i;
			for (j = 0; j < 10; j++)
				real.unit[k][c - 1].value[j] = 0;
			real.unit[k][c - 1].value[i - 1] = i;

			if (real.unit[k][c - 1].final < 1 || real.unit[k][c - 1].final > 9)
			{
				real.unit[k][c - 1].final = 0;
				for (j = 0; j < 10; j++)
					real.unit[k][c - 1].value[j] = j + 1;
			}
		}
	if (check())
		goto input;

	clrscr();
	printf("\nSolving. . .");

	solve();
	for (i = 0; i < 9; i++)
		hit_trial();

	print();
	getch();
}

void solve(void)
{
	int i, j, k;
	for (k = 1; k < 16; k++)
		for (i = 0; i < 9; i++)
			for (j = 0; j < 9; j++)
			{
				if (real.unit[i][j].final == 0)
					rowchk(i, j);
				row();
			}
}

void print()
{
	int i, j;
	clrscr();
	for (i = 0; i < 9; i++)
	{
		if (i > 1 && i % 3 == 0)
			printf("\n                        ----------|---------|----------\n                          ");
		else
			printf("\n                                  |         |\n                          ");
		for (j = 0; j < 9; j++)
		{
			if (real.unit[i][j].final < 1 || real.unit[i][j].final > 9)
			{
				if (j > 1 && j != 8 && j % 3 == 2) printf("%c | ", 22);
				else printf("%c  ", 22);
			}
			else
			{
				if (j > 1 && j != 8 && j % 3 == 2) printf("%d | ", real.unit[i][j].final);
				else printf("%d  ", real.unit[i][j].final);
			}
		}
	}
}

void rowchk(int i, int j)
{
	int v, c;
	for (v = 0; v < 9; v++)
		if (real.unit[i][j].value[v] != 0)
			for (c = 0; c < 9; c++)
				if (j != c && real.unit[i][j].value[v] == real.unit[i][c].final)
				{
					real.unit[i][j].value[v] = 0;
					break;
				}
	columnchk(i, j);
}

void columnchk(int i, int j)
{
	int v, r;
	for (v = 0; v < 9; v++)
		if (real.unit[i][j].value[v] != 0)
			for (r = 0; r < 9; r++)
				if (i != r && real.unit[i][j].value[v] == real.unit[r][j].final)
				{
					real.unit[i][j].value[v] = 0;
					break;
				}
	blockchk(i, j);
}

void blockchk(int i, int j)
{
	int v, bi, bj, i2, j2;

	for (v = 0; v < 9; v++)
		if (real.unit[i][j].value[v] != 0)
			for (i2 = 1, bi = i / 3 * 3; i2 < 4; i2++, bi++)
				for (j2 = 1, bj = j / 3 * 3; j2 < 4; j2++, bj++)
					if (real.unit[i][j].value[v] == real.unit[bi][bj].final)
					{
						real.unit[i][j].value[v] = 0;
						break;
					}
}

void row()
{
	int count, i, j, k, r;
	for (r = 0; r < 9; r++)
	{
		for (i = 1; i < 10; i++)
		{
			for (j = 0, count = 0; j < 9 && count < 2; j++)
				if (real.unit[r][j].value[i - 1] == i)
				{
					count++;
					k = j;
				}

			if (count == 1)
			{
				real.unit[r][k].final = i;
				for (j = 0; j < 9; j++)
					real.unit[r][k].value[j] = 0;
				real.unit[r][k].value[i - 1] = i;
			}
		}
	}
	column();
}

void column()
{
	int i, j, k, c, count;
	for (c = 0; c < 9; c++)
	{
		for (i = 1; i < 10; i++)
		{
			for (j = 0, count = 0; j < 9 && count < 2; j++)
				if (real.unit[j][c].value[i - 1] == i)
				{
					count++;
					k = j;
				}

			if (count == 1)
			{
				real.unit[k][c].final = i;
				for (j = 0; j < 9; j++)
					real.unit[k][c].value[j] = 0;
				real.unit[k][c].value[i - 1] = i;
			}
		}
	}
	block();
}

void block()
{
	int bi, bj, i, j, k, l, m, n, count;
	for (bi = 0; bi < 3; bi++)
		for (bj = 0; bj < 3; bj++)
		{
			for (i = 1; i < 9; i++)
			{
				count = 0;
				for (j = 0; j < 3; j++)
					for (m = 0; m < 3 && count < 2; m++)
						if (real.unit[bi *3 + j][bj *3 + m].value[i - 1] == i)
						{
							count++;
							l = bi *3 + j;
							n = bj *3 + m;
						}

				if (count == 1)
				{
					real.unit[l][n].final = i;
					for (m = 0; m < 9; m++)
						real.unit[l][n].value[m] = 0;
					real.unit[l][n].value[i - 1] = i;
				}
			}
		}
}

int check()
{
	int i, j, r, c, br, bc;

	for (r = 0; r < 9; r++)
		for (c = 0; c < 9; c++)
			if (real.unit[r][c].final != 0)
				for (i = c + 1; i < 9; i++)
					if (real.unit[r][c].final == real.unit[r][i].final)
						return 1;

	for (c = 0; c < 9; c++)
		for (r = 0; r < 9; r++)
			if (real.unit[r][c].final != 0)
				for (i = r + 1; i < 9; i++)
					if (real.unit[r][c].final == real.unit[i][c].final)
						return 1;

	for (br = 0; br < 3; br++)
		for (bc = 0; bc < 3; bc++)
			for (r = 0; r < 3; r++)
				for (c = 0; c < 3; c++)
					if (real.unit[br *3 + r][bc *3 + c].final != 0)
						for (i = br *3 + r + 1; i < br *3 + 3; i++)
							for (j = bc *3 + c + 1; j < bc *3 + 3; j++)
								if (real.unit[br *3 + r][bc *3 + c].final == real.unit[i][j].final)
									return 1;

	return 0;
}

void hit_trial()
{
	int r, c, v, i, j;

	for (r = 0; r < 9; r++)
		for (c = 0; c < 9; c++)
			if (real.unit[r][c].final == 0)
				for (v = 0; v < 9; v++)
					if (real.unit[r][c].value[v] == v + 1)
					{
						fake = real;
						for (j = 0; j < 9; j++)
							real.unit[r][c].value[j] = 0;
						real.unit[r][c].final = real.unit[r][c].value[v] = v + 1;
						solve();
						if (check())
						{
							fake.unit[r][c].value[v] = 0;
							real = fake;
							solve;
						}
					}
}
