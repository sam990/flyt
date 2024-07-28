#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>


int main(int argc, char* argv[]) {

    pthread_barrier_t *barrier;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s shm-name num-processes\n", argv[0]);
        exit(1);
    }

    int num_processes = atoi(argv[2]);

    int fd = shm_open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
        fprintf(stderr, "Error creating shared_mem fd\n");
        return -1;
    }

    ftruncate(fd, sizeof(pthread_barrier_t));

    barrier = mmap(0, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    pthread_barrierattr_t attr;
    pthread_barrierattr_init(&attr);
    pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

    pthread_barrier_init(barrier, &attr, num_processes + 1);

    fprintf(stderr, "Init\n");


    fputc('0', stdout);
    fflush(stdout);

    // sleep(5);

    int c = fgetc(stdin);

    pthread_barrier_wait(barrier);
    pthread_barrier_destroy(barrier);

    shm_unlink(argv[1]);

    fprintf(stderr, "Exiting: %c\n", c);

    return 0;

}