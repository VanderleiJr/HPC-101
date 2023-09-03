#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define ITER_MAX 3000          // número máximo de iterações
#define CONV_THRESHOLD 1.0e-5f // limite de convergência

// grid a ser resolvida
double **grid;

// grid auxiliar
double **new_grid;

// tamanho de cada lado do grid
int size;

// estrutura de argumentos para a função da thread
typedef struct {
    int start_row;
    int end_row;
    double max_err;
} ThreadArgs;

// número de threads
int num_threads;

// retorna o valor máximo
double max(double a, double b) {
    if (a > b)
        return a;
    return b;
}

// retorna o valor absoluto do número
double absolute(double num) {
    if (num < 0)
        return -1.0 * num;
    return num;
}

// alocando memória para a grid
void allocate_memory() {
    grid = (double **)malloc(size * sizeof(double *));
    new_grid = (double **)malloc(size * sizeof(double *));

    for (int i = 0; i < size; i++) {
        grid[i] = (double *)malloc(size * sizeof(double));
        new_grid[i] = (double *)malloc(size * sizeof(double));
    }
}

// inicializando a grid
void initialize_grid() {
    // seed para o gerador aleatório
    srand(10);

    int linf = size / 2;
    int lsup = linf + size / 10;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // inicializa região de calor no centro do grid
            if (i >= linf && i < lsup && j >= linf && j < lsup)
                grid[i][j] = 100;
            else
                grid[i][j] = 0;
            new_grid[i][j] = 0.0;
        }
    }
}

// interações do método de Jacobi em uma quantidade de colunas
void *jacobi_iteration(void *args) {
    ThreadArgs *threadArgs = (ThreadArgs *)args;
    int start_row = threadArgs->start_row;
    int end_row = threadArgs->end_row;
    double max_err = 0.0;
    // calcula a equação de Laplace para determinar o próximo valor
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < size - 1; j++) {
            new_grid[i][j] = 0.25 * (grid[i][j + 1] + grid[i][j - 1] +
                                     grid[i - 1][j] + grid[i + 1][j]);
            max_err = max(max_err, absolute(new_grid[i][j] - grid[i][j]));
        }
    }
    threadArgs->max_err = max_err; // Atualiza max_err na estrutura de argumentos
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: ./laplace_seq N T\n");
        printf("N: The size of each side of the domain (grid)\n");
        printf("T: Number of threads\n");
        exit(-1);
    }

    // variaveis para medir o tempo
    struct timeval time_start;
    struct timeval time_end;

    size = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    // alocando a memória do grid
    allocate_memory();

    // inicializando o grid
    initialize_grid();

    double err = 1.0;
    int iter = 0;

    printf("Jacobi relaxation calculation: %d x %d grid\n", size, size);

    // iniciando o tempo
    gettimeofday(&time_start, NULL);

    // iterações de Jacobi
    // este loop termina se a alteração máxima atingir abaixo de um limite definido (convergência)
    // ou um número fixo de iterações for feito

    while (err > CONV_THRESHOLD && iter <= ITER_MAX) {
        err = 0.0;

        // cria um array com os id das threads
        pthread_t threads[num_threads];
        ThreadArgs threadArgs[num_threads];

        // cria e executa as threads
        for (int i = 0; i < num_threads; i++) {
            // calcula a quantidade de colunas para essa thread
            int chunk_size = size / num_threads;
            threadArgs[i].start_row = i * chunk_size + 1;
            threadArgs[i].end_row = (i == num_threads - 1) ? size - 1 : (i + 1) * chunk_size + 1;
            threadArgs[i].max_err = 0.0; // Inicializa max_err para cada thread
            pthread_create(&threads[i], NULL, jacobi_iteration, (void *)&threadArgs[i]);
        }

        // junta as threads e calcula o erro máximo
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
            err = max(err, threadArgs[i].max_err); // Atualiza err com o erro máximo global
        }

        // copia os próximos valores para a próxima iteração
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                grid[i][j] = new_grid[i][j];
            }
        }
        iter++;
    }

    // pega o tempo final
    gettimeofday(&time_end, NULL);

    double exec_time = (double)(time_end.tv_sec - time_start.tv_sec) +
                       (double)(time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("\nKernel executed in %lf seconds with %d iterations and error of %0.10lf\n", exec_time, iter, err);

    return 0;
}
