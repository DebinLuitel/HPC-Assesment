#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


// function prototypes
int *find_number_of_rows_and_columns(const char *file_name);
void *multiply_matrices(void *args);

// to limit the number of input threads to the dimensions of the matrices
int max_threads_allowed;

int threadCount;

// structure to store the individual martix elements
typedef struct
{
    double x;
} unit;

// structure to store the individual matrix rows, columns
// and the reference to the matrix elements
typedef struct
{
    int rows;
    int cols;
    unit *x;
} matrix;

matrix A, B, C, _C, target;
pthread_mutex_t mutex;

int thread_counter = 0;

//creates a matrix (target) with a specified number of rows and columns and sets each element of the matrix to zero.

matrix create_matrix(int rows, int cols)
{
    // matrix target;
    int i, j;
    double temp_data;

    target.rows = rows;
    target.cols = cols;
    target.x = (unit *)malloc(rows * cols * sizeof(unit));
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            temp_data = 0.0F;
            (target.x + i * target.cols + j)->x = temp_data;
        }
    return target;
}
//This function takes in a matrix and writes the contents of the matrix to a file called DebinOutput.txt. The function iterates through each element of the matrix and writes the value to the file.

void writeinFile(matrix write_Matrix){
     int rows = write_Matrix.rows;
    int cols = write_Matrix.cols;
     FILE *fptr;
     fptr=fopen("DebinOutput.txt","w");
    int i, j;
     for (i = 0; i < rows; i++)
    {
        fprintf(fptr,"[  ");
        
        for (j = 0; j < cols; j++){
          fprintf(fptr,"%lf  ", (write_Matrix.x + i * cols + j)->x);
        }

        
        fprintf(fptr,"]\n");
        
    }
    fprintf(fptr,"\n\n");
}
//This function prints a matrix (represented by the structure "matrix") to the console. It loops through the matrix's elements row by row, printing each one.

void display_matrix(matrix displayable_matrix)
{
    int rows = displayable_matrix.rows;
    int cols = displayable_matrix.cols;
    int i, j;
   

    for (i = 0; i < rows; i++)
    {
        printf("[  ");
        
        for (j = 0; j < cols; j++){
          printf("%lf  ", (displayable_matrix.x + i * cols + j)->x);
        }
        printf("]\n");
        
    }
    printf("\n\n");
}
//This function calculates the value of one unit in the result matrix of two given matrices A and B by looping through all columns of matrix A and multiplying the elements with the corresponding elements of matrix B.

double calculate_one_matrix_unit(int first, int second)
{
    int i;
    double res = 0.0F;
    for (i = 0; i < A.cols; i++)
    {
        res += (A.x + first * A.cols + i)->x * (B.x + i * B.cols + second)->x;
    }

    return res;
}

//This function is used to multiply two matrices using threads.
//It locks a mutex to check for an empty cell in the result matrix, assigns its coordinates to firstNum and secondNum, sets the cell to 1 and unlocks the mutex.
//It then calculates the value of the empty cell and sets the result to the same cell. 

void *multiply_matrices(void *param) 
{
    while (1)
    {
        int firstNum;
        int secondNum;
        int i, j, flag = 0, close = 0;
        double res;

        pthread_mutex_lock(&mutex);
        for (i = 0; i < _C.rows; i++)
        {
            for (j = 0; j < _C.cols; j++)
            {
                if ((_C.x + i * _C.cols + j)->x == 0.0F)
                {
                    firstNum = i;
                    secondNum = j;
                    (_C.x + i * _C.cols + j)->x = 1.0F;
                    close = 1;
                    break;
                }
            }
            if (close == 1)
                break;
            else if (i == _C.rows - 1)
                flag = 1;
        }
        pthread_mutex_unlock(&mutex);

        if (flag == 1)
            pthread_exit(NULL);
        res = calculate_one_matrix_unit(firstNum, secondNum);
        (C.x + firstNum * C.cols + secondNum)->x = res;
    }
    pthread_exit(NULL);
}

//This is a main function that takes two file names and a number of threads as arguments. It then reads the two files and stores the values in matrices. 
//It also checks if the number of threads entered is valid (greater than 0) otherwise it prints an error message and exits.

void main(int argc, char *argv[])
{
     threadCount = strtol(argv[3], NULL, 10);

    if (threadCount <= 0)
    {
        printf("plese enter number of threads after file name");
       exit(0);
    }

    FILE *fp1, *fp2 = NULL;
    int row, col;
    double matval = 0.0;
    int c;

    int MatrixA_Row, MatrixA_Col, MatrixB_Row, MatrixB_Col, MatrixC_Row, MatrixC_Col;

    char *File1 = argv[1];
    char *File2 = argv[2];

    fp1 = fopen(File1, "r");
    fp2 = fopen(File2, "r");

    if (fp1 != NULL && fp2 != NULL)
    {
        int *p;
        int *q;

        p = find_number_of_rows_and_columns(File1);

        MatrixA_Row = *(p + 0);
        MatrixA_Col = *(p + 1);

    

        q = find_number_of_rows_and_columns(File2);

        MatrixB_Row = *(q + 0);
        MatrixB_Col = *(q + 1);

        // output matrix C is the combination of rows from matrix A and columns from matrix B

        MatrixC_Row = MatrixA_Row;
        MatrixC_Col = MatrixB_Col;

    

        printf("\nMatrix A \tRows: %d, Columns: %d\n", MatrixA_Row, MatrixA_Col);
        printf("Matrix B \t Rows: %d, Columns: %d\n", MatrixB_Row, MatrixB_Col);
        printf("Output Matrix\t Rows: %d, Columns: %d\n\n", MatrixA_Row, MatrixB_Col);

//checking if the number of columns in MatrixA is equal to the number of rows in MatrixB. If so, it allocates memory for each of the two matrices and stores their data.
//It also creates a target matrix and stores the data from matrix A to it. 

        if (MatrixA_Col == MatrixB_Row)
        {
            // elements/values present in each matrix
            int matA_elements = MatrixA_Row * MatrixA_Col;
            int matB_elements = MatrixB_Row * MatrixB_Col;
            int matC_elements = MatrixC_Row * MatrixC_Col;

            // dynamic memory allocation
            double *Matrix_A = (double *)malloc(matA_elements * sizeof(double));
            double *Matrix_B = (double *)malloc(matB_elements * sizeof(double));

            matrix target_matA;
            target_matA.rows = MatrixA_Row;
            target_matA.cols = MatrixA_Col;
            target_matA.x = (unit *)malloc(MatrixA_Row * MatrixA_Col * sizeof(unit));

            if (Matrix_A == NULL || Matrix_B == NULL)
            {
                printf("\nError! memory not allocated.\n");
                exit(0);
            }

            // Scanning the file and storing the matrix A data in allocated memory
            int counter = 0;
            for (row = 0; row < MatrixA_Row; row++)
            {
                for (col = 0; col < MatrixA_Col; col++)
                {
                    fscanf(fp1, "%lf,", Matrix_A + counter);
                    (target_matA.x + row * target_matA.cols + col)->x = *(Matrix_A + counter);
                    counter++;
                }
            }

            printf("\nMatrix A elements >>> \n");
            A = target_matA;
            display_matrix(A);

            matrix target_matB;
            target_matB.rows = MatrixB_Row;
            target_matB.cols = MatrixB_Col;
            target_matB.x = (unit *)malloc(MatrixB_Row * MatrixB_Col * sizeof(unit));

            // Scanning the file and storing the matrix B data in allocated memory
             counter = 0;
            for (row = 0; row < MatrixB_Row; row++)
            {
                for (col = 0; col < MatrixB_Col; col++)
                {
                    fscanf(fp2, "%lf,", Matrix_B + counter);
                    (target_matB.x + row * target_matB.cols + col)->x = *(Matrix_B + counter);
                    counter;
                }
            }

            printf("Matrix B elements >>> \n");
            B = target_matB;
            display_matrix(B);

            int i;
            C = create_matrix(A.rows, B.cols);
            for (i = 0; i < C.cols * C.rows; i++)
            {
                (C.x + i)->x = 0.0F;
            }

            _C = create_matrix(A.rows, B.cols);
            for (i = 0; i < _C.cols * _C.rows; i++)
            {
                (_C.x + i)->x = 0.0F;
            }

            pthread_t thread_id[threadCount];

            printf("Creating threads and computing the matrix multiplication...\n\n");

            pthread_mutex_init(&mutex, NULL);

//creating "threadCount" number of threads and making them run the "multiply_matrices" function.
// Once all threads have completed, it prints the output matrix C on a file, deallocates all the memory used, and closes the two files.

            for (int m = 0; m < threadCount; m++)
            {
                pthread_create(&thread_id[m], NULL, multiply_matrices, NULL);
            }

            for (int n = 0; n < threadCount; n++)
            {
                pthread_join(thread_id[n], NULL);
            }

            printf("\n\nOutput matrix C is printed on file output.txt \n");
            writeinFile(C);

            // deallocating the memory
            free(Matrix_A);
            free(Matrix_B);
            free(target_matA.x);
            free(target_matB.x);
            free(target.x);
        }
        else
        {
            printf("\nOops! the column of matrix A is not equal to the row of matrix B, thus matrices cannot be multiplied.\n");
        }

        fclose(fp1);
        fclose(fp2);
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}


// function to find the number of rows and columns of each matrix from the files
int *find_number_of_rows_and_columns(const char *file_name)
{
    FILE *fp = fopen(file_name, "r");
    int newRows = 1;
    int newCols = 1;
    char ch;

    static int rows_cols[10];

    while (!feof(fp))
    {
        ch = fgetc(fp);

        if (ch == '\n')
        {
            newRows++;
            // rows_cols[0] = newCols;
            newCols = 1;
        }
        else if (ch == ',')
        {
            newCols++;
        }
    }
    rows_cols[0] = newRows;
    rows_cols[1] = newCols;

    // printf("\nRows: %d, Cols: %d\n", rows_cols[0], rows_cols[1]);

    return rows_cols;
}
