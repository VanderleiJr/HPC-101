%%writefile laplace_omp.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define ITER_MAX 3000          // número máximo de iterações
#define CONV_THRESHOLD 1.0e-5f // limite de convergência

// grid a ser resolvida
double **grid;

// grid auxiliar
double **new_grid;

// tamanho de cada lado do grid
int size;

// numero de threads
int num_threads;

// numero de iterações
int iter;

// erro
double err;

// retorna o valor máximo
double max(double a, double b){
    if(a > b)
        return a;
    return b;
}

// retorna o valor absoluto do número
double absolute(double num){
    if(num < 0)
        return -1.0 * num;
    return num;
}

// alocando memória para a grid
void allocate_memory(){
    grid = (double **) malloc(size * sizeof(double *));
    new_grid = (double **) malloc(size * sizeof(double *));

    for(int i = 0; i < size; i++){
        grid[i] = (double *) malloc(size * sizeof(double));
        new_grid[i] = (double *) malloc(size * sizeof(double));
    }
}

// inicializando a grid
void initialize_grid(){
    // seed para o gerador aleatório
    srand(10);

    int linf = size / 2;
    int lsup = linf + size / 10;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // inicializa região de calor no centro do grid
            if ( i >= linf && i < lsup && j >= linf && j < lsup)
                grid[i][j] = 100;
            else
                grid[i][j] = 0;
            new_grid[i][j] = 0.0;
        }
    }
}

// Jacobi iteration
void jacobi_iteration(){
    err = 0.0;

    // Paralelize os dois loops aninhados
    #pragma omp parallel for reduction(max:err)
    for(int i = 1; i < size-1; i++){
        for(int j = 1; j < size-1; j++){
            new_grid[i][j] = 0.25 * (grid[i][j+1] + grid[i][j-1] +
                                     grid[i-1][j] + grid[i+1][j]);

            err = max(err, absolute(new_grid[i][j] - grid[i][j]));
        }
    }

    // Fora da região paralela, copie os valores de new_grid para grid
    for(int i = 1; i < size-1; i++){
        for(int j = 1; j < size-1; j++){
            grid[i][j] = new_grid[i][j];
        }
    }
}

int main(int argc, char *argv[]){

    if(argc != 3){
        printf("Usage: ./laplace_seq N T\n");
        printf("N: The size of each side of the domain (grid)\n");
        printf("T: The number of threads to use\n");
        exit(-1);
    }

    // variables to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    size = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    // usuário define a quantidade de threads
    omp_set_num_threads(num_threads);

    // allocate memory to the grid (matrix)
    allocate_memory();

    // set grid initial conditions
    initialize_grid();

    err = 1.0;
    iter = 1;

    printf("Jacobi relaxation calculation: %d x %d grid\n", size, size);

    // get the start time
    gettimeofday(&time_start, NULL);

    // Jacobi iteration
    // This loop will end if either the maximum change reaches below a set threshold (convergence)
    // or a fixed number of maximum iterations have completed
    while ( err > CONV_THRESHOLD && iter <= ITER_MAX ) {

        err = 0.0;

        // call the jacobi_iteration function
        jacobi_iteration();

        iter++;

    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                       (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("\nKernel executed in %lf seconds with %d iterations and error of %0.10lf\n", exec_time, iter, err);

    return 0;
}
