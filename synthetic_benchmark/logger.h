#ifndef SHARED_MEMORY_LOGGER_H
#define SHARED_MEMORY_LOGGER_H

#include <iostream>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>

#define FIFO_NAME "/tmp/flyt_shmem.log"

class FlytLogger {
public:
    // Constructor takes only the PID
    explicit FlytLogger(pid_t pid)
        : pid(pid), fifo_fd(-1) {
        // Open and initialize shared memory
        initSharedMemory();
	//printf("Initializing thread\n");
    }

    ~FlytLogger() {
        close(fifo_fd);
    }

    // Log a function's metrics
    void writeLatencyLog(  int latency, 
		    int upperlimit, 
		    int lowerlimit ) {
        // Register a function
	if(fifo_fd == -1) {
		pid = getpid();
		initSharedMemory();
	}
	printf("logging\n");
        std::string command = formatLogLatencyCommand(pid, latency, upperlimit, lowerlimit);
        writeMessage(command);
	//close(fifo_fd);
	//fifo_fd = -1;
    }

    // Log a function's metrics
    void writeMemoryLog(  int memsize, 
		    int success ) {
        // Register a function
	if(fifo_fd == -1) {
		pid = getpid();
		initSharedMemory();
	}
	printf("logging mem alloc\n");
        std::string command = formatLogMemoryCommand(pid, memsize, success);
        writeMessage(command);
	//close(fifo_fd);
	//fifo_fd = -1;
    }

private:
    pid_t pid;
    int fifo_fd;

    void initSharedMemory() {
        // Open the shared memory object
        fifo_fd = open(FIFO_NAME, O_WRONLY);
        if (fifo_fd == -1) {
            perror("shared memory file open");
            throw std::runtime_error("Failed to open shared memory");
        }
    }

    void writeMessage(const std::string& message) {
        if (write(fifo_fd, message.c_str(), message.size()) < message.size()) {
	    perror("Failed to write to fifo");
            throw std::runtime_error("Failed to write to fifo");
        }
    }

    std::string formatLogLatencyCommand(unsigned int pid, 
		    int metrics,
		    int upperlimit,
		    int lowerlimit) {
        return "LATENCY " + std::to_string(pid) + " " + 
		std::to_string(metrics) + " ",
		std::to_string(upperlimit) + " ",
		std::to_string(lowerlimit) + "\n";
    }

    std::string formatLogMemoryCommand(unsigned int pid, 
		    int memsize,
		    int success) {
        return "MEMORY " + std::to_string(pid) + " " + 
		std::to_string(memsize) + " ",
		std::to_string(success) + "\n";
    }
};

#endif // SHARED_MEMORY_LOGGER_H

