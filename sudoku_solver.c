#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include "iocore.h"
#include "imcore.h"
#include "lacore.h"
#include "cvcore.h"
#include "mlcore.h"
#include "sudoku.hpp"

// invert input image
void bwinvert(matrix_t *in)
{
    int i = 0;
    for (i = 0; i < volume(in); i++)
    {
        atui8(in, i) = 255 - atui8(in, i);
    }
}

// highlights the given keyPoints with the given color (input must be three channel image)
void highlight(matrix_t *image, vector_t *keyPoints, struct color_t highlightColor)
{
    int i = 0;

    // make the whole image darker
    for (i = 0; i < volume(image); i++)
    {
        atui8(image, i) = 0.3 * atui8(image, i);
    }

    // highlight the given points
    for (i = 0; i < length(keyPoints); i++)
    {
        struct point_t p = at(struct point_t, keyPoints, i);

        atui8(image, (int)p.y, (int)p.x, 0) = highlightColor.blue;
        atui8(image, (int)p.y, (int)p.x, 1) = highlightColor.green;
        atui8(image, (int)p.y, (int)p.x, 2) = highlightColor.red;
    }
}


void highlight_peaks(matrix_t *image, vector_t *peaks, struct color_t highlightColor)
{
    uint32_t i = 0;
    for (i = 0; i < length(peaks); i++)
    {
        // get the point
        struct point_t p = at(struct point_t, peaks, i);
        
        // two corners of the line
        struct point_t pt1, pt2;

        // find a and b
        float a = cos(p.y * 3.14159265359 / 180), b = sin(p.y * 3.14159265359 / 180);

        // find a point on the line
        float x0 = a * p.x, y0 = b * p.x;

        // construct two corners
        pt1.x = round(x0 + 1000 * (-b));
        pt1.y = round(y0 + 1000 * (a));
        pt2.x = round(x0 - 1000 * (-b));
        pt2.y = round(y0 - 1000 * (a));

        draw_line(image, pt1, pt2, highlightColor, 1);
    }
}

// given polar angles (degrees) and radius, finds the intersection point in cartesian coordinates
int find_intersection(float theta1, float rho1, float theta2, float rho2, float *x, float *y)
{
    x[0] = 0;
    y[0] = 0;

     // convert degree to radian
    float t1 = theta1 * 3.14159265359 / 180;
    float t2 = theta2 * 3.14159265359 / 180;

    // if the lines are parallel to each other, no intersection point is possible
    if(sin(t1 - t2) == 0.0f)
    {
        return 0;
    }
    // compute the intersection point
    else
    {
        x[0] = (rho2 * sin(t1) - rho1 * sin(t2)) / sin(t1 - t2);
        y[0] = (rho1 * cos(t2) - rho2 * cos(t1)) / sin(t1 - t2);
    }
    
    return 1;
}

int find_max_prediction(matrix_t *prediction, int grid[9][9], int *row, int *col)
{
    float max_score = -100000;
    int max_value = -1;

    row[0] = 0;
    col[0] = 0;

    // use the prediction probabilities
    int idx = 0, class = 0, r = 0, c = 0;
    for (r = 0; r < 9; r++)
    {
        for (c = 0; c < 9; c++)
        {
            // if the grid is empty
            if(grid[r][c] < 0)
            {
                for( class = 0; class < 10; class++)
                {
                    // try all classes and find the max score for the cell (r,c)
                    float score = at(float, prediction, idx, class, 0);

                    if (score > max_score)
                    {
                        max_score = score;
                        row[0] = r;
                        col[0] = c;
                        max_value = class;
                    }
                }
            }

            // go to the next element
            idx++;
        }
    }

    return max_value;
}

void fill_sudoku_grid(matrix_t *prediction, int grid[9][9])
{
    int row = 0, col = 0;

    // fill grid with -1
    for(row = 0; row < 9; row++)
    {
        for(col = 0; col < 9; col++)
        {
            grid[row][col] = -1;
        }
    }

    // find the max probability cell
    int max_value;
    while( (max_value = find_max_prediction(prediction, grid, &row, &col)) >= 0)
    {
        if (max_value == 0 || is_safe(grid, row, col, max_value))
        {
            grid[row][col] = max_value;
        }
        else
        {
            atf(prediction, 9*row + col, max_value, 0) = -1000000;
        }
    }    
}

void predict_multiclass(matrix_t *feature, matrix_t *label, struct ann_t *net)
{
    uint32_t n_class = cols(label);
    uint32_t n_sample = rows(feature);

    // create an temporary output matrix
    matrix_t *output = matrix_create(float, n_sample, n_class);

    // train a ann classifier for each stage
    uint32_t sample = 0, class = 0;
    // get the predictions
    ann_predict(feature, output, net);

    // find the class label for each sample and class
    for (sample = 0; sample < n_sample; sample++)
    {
        for (class = 0; class < n_class; class ++)
        {
            atf(label, sample, class) += atf(output, sample, class);
        }
    }

    // remove the unused matrix
    matrix_free(&output);
}

// given grayscale aligned and scaled (360x360x1) sudoku image, fills the grid
void recognize_digits(matrix_t *sudoku, int grid[9][9])
{
    int CellSize = 40;

    struct feature_t *extractor = feature_create(CV_HOG, CellSize, CellSize, 1, "-block:2x2 -cell:4x4 -stride:1x1 -nbins:18");

    struct ann_t *testNet = ann_read("..//data//trained_model.ann");

    // allocate space for the features
    matrix_t *features = matrix_create(float, 81, feature_size(extractor), 1);

    // allocate space for gray scale image
    matrix_t *sudokuGray = matrix_create(uint8_t, rows(sudokuGray), cols(sudokuGray), 1);
    rgb2gray(sudoku, sudokuGray);

     // allocate space for temp cell image
    matrix_t *cellImage = matrix_create(uint8_t, CellSize, CellSize, 1);

    int i = 0, j = 0, idx = 0;
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            struct rectangle_t crop_region = rectangle(j * CellSize, i * CellSize, CellSize, CellSize, 0);
            imcrop(sudokuGray, crop_region, cellImage);

            // imwrite(cellImage, imlab_filename("test//digit","bmp"));

            // extract the feature of the cell i,j and write it into the idx th row of the feature vector
            feature_extract(cellImage, extractor, data(float, features, idx++, 0));
        }
    }

    // create output matrix for the labels
    matrix_t *label_predicted = matrix_create(float, 81, 10, 1);

    // do classification/regression
    predict_multiclass(features, label_predicted, testNet);

    // fill the grid using the resulting scores
    fill_sudoku_grid(label_predicted, grid);

    // clean the grayscale copy of the image
    matrix_free(&sudokuGray);
}

// find the hough peaks and return it as a vector
vector_t *find_hough_peaks(vector_t *keyPoints, uint32_t width, uint32_t height)
{
    uint32_t mapSize = 2 * sqrt(width*width + height*height) + 1;
    uint32_t mapSizeHalf = (mapSize - 1) / 2;

    int minTheta = -30;
    int maxTheta = 120;

    // create hough accumulator array
    matrix_t *hough = matrix_create(float, (maxTheta - minTheta + 1), mapSize);

    int i = 0, j = 0;
    for (i = 0; i < length(keyPoints); i++)
    {
        struct point_t p = at(struct point_t, keyPoints, i);

        int theta = 0;
        for (theta = minTheta; theta <= maxTheta; theta++)
        {
            float t = (float)(theta) * 3.14159265359 / 180;
            int rho = p.x * cos(t) + p.y * sin(t);

            atf(hough, theta - minTheta, mapSizeHalf + rho, 0) += 1.0f;
        }
    }

    matrix_t *hough_image = matrix_create(uint8_t, rows(hough), cols(hough), 1);
    matrix2image(hough, 0, hough_image);
    imwrite(hough_image, "hough.bmp");

    // threshold the hough map and find the line candidates
    vector_t *candidates = vector_create(struct point_t);
    int theta,rho;
    for(theta = 0; theta < rows(hough); theta++)
    {
        for(rho = 0; rho < cols(hough); rho++)
        {
            if(atf(hough, theta, rho) > 0.4 * minimum(width,height))
            {
                struct point_t p = point(rho - mapSizeHalf, theta + minTheta,0);
                vector_push(candidates, &p);
            }
        }
    }

    // merge too close candidates together
    vector_t *candidates_merged = point_merge(candidates, 5, 1);

    // print debug information
    printf("Total %d line candidates reduced to %d!\n", length(candidates), length(candidates_merged));

    // remove unused arrays
    vector_free(&candidates);
    matrix_free(&hough);

    // return the hough lines
    return candidates_merged;
}


// find the four sides that constructs the sudoku grid
vector_t* find_corners(vector_t *peaks, struct point_t corners[4])
{
    struct point_t edge[4] = {{.x = 1e6, .y = 0},{.x = -1e6, .y = 0},{.x = 1e6, .y = 0},{.x = -1e6, .y = 0}};

    // find the edge peaks for theta > 45 and theta < 45
    uint32_t i = 0;
    for (i = 0; i < length(peaks); i++)
    {
        struct point_t p = at(struct point_t, peaks, i);

        float rho = p.x;
        float theta = p.y;

        // right or left line
        if(theta < 45)
        {
            if(rho < edge[2].x)
            {
                edge[2].x = rho;
                edge[2].y = theta;
            }

            if(rho > edge[3].x)
            {
                edge[3].x = rho;
                edge[3].y = theta;
            }
        }
        // top or bottom
        else
        {
            if (rho < edge[0].x)
            {
                edge[0].x = rho;
                edge[0].y = theta;
            }

            if (rho > edge[1].x)
            {
                edge[1].x = rho;
                edge[1].y = theta;
            }
        }
    }

    //top,bottom,right,left
    int i0 = find_intersection(edge[0].y, edge[0].x, edge[2].y, edge[2].x, &corners[0].x, &corners[0].y);
    int i1 = find_intersection(edge[0].y, edge[0].x, edge[3].y, edge[3].x, &corners[1].x, &corners[1].y);
    int i2 = find_intersection(edge[1].y, edge[1].x, edge[3].y, edge[3].x, &corners[2].x, &corners[2].y);
    int i3 = find_intersection(edge[1].y, edge[1].x, edge[2].y, edge[2].x, &corners[3].x, &corners[3].y);

    // return the four edge(top,bottom,right,left) of the sudoku
    return vector_create(struct point_t, 4, edge);
}

void highlight_cells(matrix_t *sudoku, struct color_t highlightColor)
{
    // paint grids for visibility
    uint32_t step = rows(sudoku) / 9;
    uint32_t i = 0;
    for(i = 0; i < 8; i++)
    {
        struct point_t v1 = point((i + 1) * step, 0, 0);
        struct point_t v2 = point((i + 1) * step, cols(sudoku) - 1, 0);

        struct point_t h1 = point(0, (i + 1) * step, 0);
        struct point_t h2 = point(rows(sudoku) - 1, (i + 1) * step, 0);

        draw_line(sudoku, v1, v2, highlightColor, 1);
        draw_line(sudoku, h1, h2, highlightColor, 1);
    }
}

void display_sudoku(int grid[9][9])
{
    int i,j;

    // pretty print the sudoku grid
    for (i = 0; i < 9; i++)
    {
        printf("|");
        for (j = 0; j < 9; j++)
        {
            if(grid[i][j] != 0) 
            {
                printf(" %d ", grid[i][j]);
            }
            else
            {
                printf(" * ");
            }

            if(j == 2 || j == 5)
            {
                printf("|");
            }
        }
        printf("|\n");

        if(i == 2 || i == 5)
        {
           for (j = 0; j < 9; j++)
           {
               printf("---");
           }
           printf("----\n");
        }
    }
}

int main(int argc, unsigned char *argv[]) {

    // read the test image
    unsigned char filename[256] = "..//data//1.bmp";
    if(argc > 1) {
        strncpy(filename, argv[1], 256);
    }
    // write the grid image to file
    int num = 0;
    sscanf(filename, "%*[^0-9]%d", &num);

    printf("Processing File: %s\n", filename);

    matrix_t *image = imread(filename);
    matrix_t *image_copy = imread(filename);
    matrix_t *gray_image = matrix_create(uint8_t);
    matrix_t *binary_image = matrix_create(uint8_t);

    // convert image into grayscale
    rgb2gray(image, gray_image);
    imbinarize(gray_image, 32, 32, 0, binary_image);
    bwinvert(binary_image);

    uint32_t numberOfComponenets = 0;
    vector_t **connnectionList = bwconncomp(binary_image, &numberOfComponenets);

    // print some debug information
    printf("Number of connected components: %d\n", numberOfComponenets);

    // find the largest connected component block (which is assumed to be the sudoku)
    uint32_t largestPixelCount = 0;
    uint32_t largestID = 0;

    uint32_t i = 0;
    for (i = 0; i < numberOfComponenets; i++)
    {
        if (length(connnectionList[i]) > largestPixelCount)
        {
            largestPixelCount = length(connnectionList[i]);
            largestID = i;
        }
    }

    // paint it for better visualization
    // highlight(image, connnectionList[largestID], RGB(120, 200, 120));
    // sprintf(filename, "grid_highlighted_%d.bmp", num);
    // imwrite(image, filename);

    // find the hough peaks
    vector_t *peaks = find_hough_peaks(connnectionList[largestID], width(image), height(image));

    // highlight the peaks as line
    // highlight_peaks(image, peaks, RGB(255, 200, 120));
    // sprintf(filename, "all_lines_%d.bmp", num);
    // imwrite(image, filename);

    // find the four corners of the sudoku
    struct point_t corners[4];
    vector_t *edges = find_corners(peaks, corners);
    // highlight_peaks(image_copy, edges, RGB(255, 100, 0));

    // draw corners too
    // draw_point(image_copy, corners[0], RGB(50, 50, 200), 15);
    // draw_point(image_copy, corners[1], RGB(50, 50, 200), 15);
    // draw_point(image_copy, corners[2], RGB(50, 50, 200), 15);
    // draw_point(image_copy, corners[3], RGB(50, 50, 200), 15);

    // sprintf(filename, "all_edges_%d.bmp", num);
    // imwrite(image_copy, filename);

    // perspective transform the sudoku to square
    matrix_t *sudoku = matrix_create(uint8_t, 360, 360, 3);
    struct point_t destination[4] = {{.x = 0, .y = 0},{.x = 359, .y = 0},{.x = 359, .y = 359},{.x = 0, .y = 359}};

    // correct the image
    matrix_t *transform = pts2tform(corners, destination, 4);
    imtransform(image, transform, sudoku);

    // write the resulting image
    sprintf(filename, "sudoku_%d.bmp", num);
    imwrite(sudoku, filename); 

    // find the numbers in sudoku cells
    int grid[9][9] = {0};
    recognize_digits(sudoku, grid);

    // find the hough lines
    printf("Detected Sudoku!\n");
    display_sudoku(grid);

    int result = solve_sudoku(grid);

    if(result == 1)
    {
        printf("Solved Sudoku!\n");
        display_sudoku(grid);
    }
    else
    {
        printf("Sudoku not solved!\n");
    }

    return 0;
}