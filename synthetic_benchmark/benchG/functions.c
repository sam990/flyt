#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>

#include "civetweb.h"

#include "functions_map.h"

#define BATCH_WAIT_TIME_MS 20

#define MAX_QUERY_LEN 128

#define POST_DATA_BYTES 256

#define MAX_BATCH_SIZE 256

#define SLA_TIME_MS 300

int init_device_vars(size_t, size_t);
void executeFunc(uint8_t* data, int* responses, int num_requests);


static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t clear_cond = PTHREAD_COND_INITIALIZER;

int num_active_requests = 0;
uint8_t client_total_data[MAX_BATCH_SIZE * POST_DATA_BYTES];
int client_responses[MAX_BATCH_SIZE];
int batch_executing = 0;
static int responses_pending_count = 0;

static pthread_t clear_batch_thread_id;

static int num_violations = 0;

void* increase_provisioning(void * arg) {
    int sockfd;
    struct sockaddr_in servaddr;

    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    unsigned long long time_ms = current_time.tv_sec * 1000 + current_time.tv_usec / 1000;

    FILE* fp = fopen("reconfig_log.txt", "a");
    fprintf(fp, "%llu\n", time_ms);
    fclose(fp);
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set server address
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("10.129.27.228");
    servaddr.sin_port = htons(32578);

    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) != 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }

    uint8_t data[1] = {1};
    // Perform operations on the socket
    write(sockfd, data , 1);

    read(sockfd, data, 1);

    if (data[0] == 0) {
        printf("Provisioning increased\n");
    } else {
        printf("Provisioning not increased\n");
    }

    // Close the socket
    close(sockfd);

    return NULL;
    
}


int execute_batch() {
    while (responses_pending_count > 0) {
        pthread_cond_wait(&clear_cond, &mutex);
    }

    batch_executing = 1;
    if (num_active_requests > 0) {

        // print 
        // for (int i = 0; i < num_active_requests; i++) {
        //     printf("Request %d: ", i);
        //     for (int j = 0; j < POST_DATA_BYTES; j++) {
        //         printf("%x ", client_total_data[i * POST_DATA_BYTES + j]);
        //     }
        //     printf("\n");
        // }

        printf("Executing batch of %d requests\n", num_active_requests);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        executeFunc(client_total_data, client_responses, num_active_requests);
        clock_gettime(CLOCK_MONOTONIC, &end);
        long long elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
        if (elapsed_ms > SLA_TIME_MS) {
            num_violations++;
            printf("SLA violation: %lld ms\n", elapsed_ms);
        }
        if (num_violations >= 3) {
            pthread_t provisioning_thread;
            num_violations = 0;
            pthread_create(&provisioning_thread, NULL, increase_provisioning, NULL);
        }
    }
    responses_pending_count = num_active_requests;
    num_active_requests = 0;
    batch_executing = 0;
    pthread_cond_broadcast(&cond);
    return 0;
}

void empty_signal_handler(int sig) {

}

void* clear_batch_thread(void *arg) {
    signal(SIGUSR1, empty_signal_handler);
    const struct timespec wait_time = {0, BATCH_WAIT_TIME_MS * 1000000ull};
    while (1) {
        int intr = nanosleep(&wait_time, NULL);
        if (intr == -1) {
            printf("nanosleep interrupted\n");
        }
        pthread_mutex_lock(&mutex);
        execute_batch();
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}


int dummy_function(struct mg_connection* conn, const char* query_str, char* response_dest, int response_max_len, int* response_len) {
    pthread_mutex_lock(&mutex);

    if (num_active_requests >= MAX_BATCH_SIZE) {
        pthread_kill(clear_batch_thread_id, SIGUSR1);
    }

    while (num_active_requests >= MAX_BATCH_SIZE) {
        pthread_cond_wait(&cond, &mutex);
    }

    int self_id = num_active_requests;
    num_active_requests++;

    mg_read(conn, client_total_data + self_id * POST_DATA_BYTES, POST_DATA_BYTES);


    while (responses_pending_count == 0) {
        pthread_cond_wait(&cond, &mutex);
    }

    int result = client_responses[self_id];
    responses_pending_count--;

    if (responses_pending_count == 0) {
        pthread_cond_broadcast(&clear_cond);
    }

    pthread_mutex_unlock(&mutex);

    *response_len = snprintf(response_dest, response_max_len, "%d", result);

    return 1;
}

int init_functions() {
    init_device_vars(POST_DATA_BYTES, MAX_BATCH_SIZE);
    pthread_create(&clear_batch_thread_id, NULL, clear_batch_thread, NULL);
    return 0;
}


FUNCTION_MAPS(
    DEFINE_FUNC_MAP("/dummy", dummy_function);
)





