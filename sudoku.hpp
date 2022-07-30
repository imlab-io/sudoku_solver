// implementation of the backtrace algorithm for sudoku solving
// https: //www.geeksforgeeks.org/sudoku-backtracking-7/

// try to find an empty cell, return 1 if found
int find_empty_cell(int grid[9][9], int *row, int *col)
{
    row[0] = 0;
    col[0] = 0;

    int r = 0, c = 0;
    for (r = 0; r < 9; r++)
    {
        for (c = 0; c < 9; c++)
        {
            if (grid[r][c] == 0)
            {
                row[0] = r;
                col[0] = c;
                return 1;
            }
        }
    }
    return 0;
}
// check row, col and cell for the given number value
// if the value does  not exist in one of them, it is safe to try
int is_safe(int grid[9][9], int row, int col, int value)
{
    int r = 0, c = 0, d = 0;
    // check cols and rows
    for(d = 0; d < 9; d++)
    {
        if(grid[d][col] == value || grid[row][d] == value)
        {
            return 0;
        }
    }
    // check cell
    row = 3 * (int)(row/3);
    col = 3 * (int)(col/3);

    for (r = 0; r < 3; r++)
    {
        for (c = 0; c < 3; c++)
        {
            if (grid[row + r][col + c] == value)
            {
                return 0;
            }
        }
    }

    return 1;
}

// try to solve the sudoku using backtracing algorithm
int solve_sudoku(int grid[9][9])
{
    int row = 0, col = 0;
    if (find_empty_cell(grid, &row, &col))
    {
        int value = 0;
        for(value = 1; value <= 9; value++)
        {
            // is it safe to use #value for the (row,col)
            if(is_safe(grid, row, col, value))
            {
                // assign the value and try to solve rest
                grid[row][col] = value;
                if (solve_sudoku(grid))
                {
                    return 1;
                }

                // if we are here assigned value causes to return 0
                grid[row][col] = 0;
            }
        }

        // this means we cannot assign any value to the current cell, this will detect invalid sudokus
        return 0;
    }

    // all the cells are filled
    return 1;
}
