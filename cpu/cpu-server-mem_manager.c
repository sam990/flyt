/* Copyright (c) 2024-2026 SynerG Lab, IITB */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>

#define MAX_GPU_MEMORY_SIZE	24*1024*1024*1024ull
#define HOST_PAGE_SIZE	4096
#define SHIFT_VALUE 48ull
#define ARRAY_SIZE MAX_GPU_MEMORY_SIZE/HOST_PAGE_SIZE  // Define the array size - 24GB / 4K page size.
#define VIRTUAL_MEMORY_START	((uint64_t)1 << SHIFT_VALUE)

#define SHM_NAME "/flyt-mem-file"  // Shared memory name
#define SEM_NAME "/flyt-mem-sem"  // Semaphore name
#define MEM_LOCK "/tmp/flyt-mem-lck"


// Shared memory pointer and semaphore
unsigned char *bit_array;
sem_t *sem;
int mem_lck_fd = -1;
int mem_shd_fd = -1;

// Helper function to get the value of a specific bit
bool get_bit(int index) {
    return (bit_array[index / 8] >> (index % 8)) & 1;
}

// Helper function to set a specific bit to 1
void set_bit(int index) {
    bit_array[index / 8] |= (1 << (index % 8));
}

// Helper function to unset (clear) a specific bit to 0
void unset_bit(int index) {
    bit_array[index / 8] &= ~(1 << (index % 8));
}

// Function to return the index of the first 0 in the array
int find_first_zero() {
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (!get_bit(i)) {
            return i;
        }
    }
    return -1;  // Return -1 if no 0 is found
}

// Function to set the next n values starting from index x to 1
void set_next_n_values(int start, int n) {
    for (int i = 0; i < n; i++) {
        if (start + i < ARRAY_SIZE) {
            set_bit(start + i);
        }
    }
}

// Function to unset the next n values starting from index x to 0
void unset_next_n_values(int start, int n) {
    for (int i = 0; i < n; i++) {
        if (start + i < ARRAY_SIZE) {
            unset_bit(start + i);
        }
    }
}

int check_next_n_zero_bits(int start, int n) {
    for (int i = 0; i < n; i++) {
        if (start + i < ARRAY_SIZE) {
            if(get_bit(start + i) ) {
		    return 1;
	    }
        }
    }
    return 0;
}

// Function to find the starting index of the first sequence of n consecutive 0 bits
int find_next_n_zero_bits(int n) {
    int consecutive_zeros = 0;
    int start_index = -1;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (!get_bit(i)) {
            if (consecutive_zeros == 0) {
                start_index = i;  // Start of a potential sequence
            }
            consecutive_zeros++;
            if (consecutive_zeros == n) {
                return start_index;
            }
        } else {
            consecutive_zeros = 0;  // Reset the count if a 1 is encountered
        }
    }
    return -1;  // Return -1 if no such sequence is found
}

int init_shared_resources() {
    int is_creator = 0;

    if ( mem_lck_fd != -1) {
	    printf("Memory is already initialized.\n");
	    return 1;
    }

    mem_lck_fd = open(MEM_LOCK, O_CREAT | O_RDWR, 0666);
    if (mem_lck_fd == -1) {
        perror("Failed to open memory lock file");
        return 1;
    }

    // Try to acquire the write lock (this is for the first process to run)
    int flock_err = flock(mem_lck_fd, LOCK_EX| LOCK_NB);
    if (flock_err != 0) {
	if (errno != EWOULDBLOCK) {
       		perror("Failed to acquire exculsive lock");
       		close(mem_lck_fd);
       		return 1;
	}
	flock_err = flock(mem_lck_fd, LOCK_SH); // wait for shared lock
	if (flock_err != 0) {
       		perror("Failed to acquire shared lock");
       		close(mem_lck_fd);
       		return 1;
	}
    }
    else {
        is_creator = 1;
    }

    // here i have either shared or exclusive lock

    // Open or create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        exit(1);
    }

    if (is_creator == 1) {
        // First process to create the shared memory
        if (ftruncate(shm_fd, ARRAY_SIZE / 8) == -1) {
            perror("ftruncate");
            exit(1);
        }
    }

    // Map the shared memory
    bit_array = mmap(NULL, ARRAY_SIZE / 8, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (bit_array == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Initialize the memory only if this process is the creator
    if (is_creator) {
        memset(bit_array, 0, ARRAY_SIZE / 8);

	// Reset the semaphore: unlink and recreate
        if (sem_unlink(SEM_NAME) == -1) {
            if (errno != ENOENT) { // Ignore error if semaphore does not exist
                perror("sem_unlink");
                exit(EXIT_FAILURE);
            }
        } else {
            printf("Semaphore unlinked successfully.\n");
        }

        // Recreate the semaphore with the initial value
        sem = sem_open(SEM_NAME, O_CREAT | O_EXCL, 0666, 1);
        if (sem == SEM_FAILED) {
            perror("sem_open exclusive");
            exit(EXIT_FAILURE);
        }
    }
    else {
	// Open the existing semaphore
        sem = sem_open(SEM_NAME, 0);
        if (sem == SEM_FAILED) {
            perror("sem_open shared");
            exit(EXIT_FAILURE);
        }
    }

    // Now acquire shared lock.
    flock_err = flock(mem_lck_fd, LOCK_SH); // wait for shared lock
    return 0;
}

// Function to clean up shared resources
void cleanup_shared_resources() {
    if (munmap(bit_array, ARRAY_SIZE / 8) == -1) {
        perror("munmap");
    }
    flock(mem_lck_fd, LOCK_UN);

    close(mem_lck_fd);

    if (shm_unlink(SHM_NAME) == -1) {
        perror("shm_unlink");
    }
    sem_close(sem);
    if (sem_unlink(SEM_NAME) == -1) {
        perror("sem_unlink");
    }

}


void *get_next_uva(size_t size) {

    size_t nbits = size / HOST_PAGE_SIZE;
    void *ptr = NULL;
    // Lock the semaphore before accessing shared memory
    if (sem_wait(sem) == -1) {
        perror("sem_wait");
        exit(1);
    }

    // Critical Section
    int sequence_start = find_next_n_zero_bits(nbits);
    if (sequence_start != -1) {
	size_t offset = sequence_start * HOST_PAGE_SIZE;
	ptr = (void *)((uint64_t)VIRTUAL_MEMORY_START + offset);
        printf("First zero found at index: %d ptr = %p\n", sequence_start, ptr);
        set_next_n_values(sequence_start, nbits);
    }

    // Unlock the semaphore after finishing
    if (sem_post(sem) == -1) {
        perror("sem_post");
        exit(1);
    }

    return ptr;
}

int get_uva_addr(void *ptr, size_t size) {
    size_t nbits = size / HOST_PAGE_SIZE;

    if (ptr == NULL)
	return 1;

    size_t offset = ((uint64_t)ptr - (uint64_t)VIRTUAL_MEMORY_START);
    offset = offset / HOST_PAGE_SIZE;

    // Lock the semaphore before accessing shared memory
    if (sem_wait(sem) == -1) {
        perror("sem_wait");
        exit(1);
    }

    int not_found = check_next_n_zero_bits(offset, nbits);

    if (not_found == 0) {
	    set_next_n_values(offset, nbits);
    }
    sem_post(sem);

    return not_found;
}

int free_uva_addr(void *ptr, size_t size) {
	size_t nbits = size / HOST_PAGE_SIZE;

    if (ptr == NULL)
	return 1;

    size_t offset = ((uint64_t)ptr - (uint64_t)VIRTUAL_MEMORY_START);
    offset = offset / HOST_PAGE_SIZE;
	
    // Lock the semaphore before accessing shared memory
    if (sem_wait(sem) == -1) {
        perror("sem_wait");
        exit(1);
    }

    if(get_bit(offset) == 0) {
        printf("Something is really really wrong");
	sem_post(sem);
	return 1;
    }

    unset_next_n_values(offset, nbits);

    sem_post(sem);
    return 0;
}


