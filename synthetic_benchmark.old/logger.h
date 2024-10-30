#ifndef SHARED_MEMORY_LOGGER_H
#define SHARED_MEMORY_LOGGER_H

#include <iostream>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>

class FlytLogger {
public:
    // Constructor takes only the PID
    explicit FlytLogger(pid_t pid)
        : pid(pid), fifo_name("/tmp/flyt_shmem.log"), send(0) {
        // Open and initialize shared memory
        initSharedMemory();
	printf("Initializing thread\n");
        // Register the application
	std::string command = formatStartCommand(pid, 1);
	writeMessage(command);
    }

    ~FlytLogger() {
        // Unmap and close the shared memory
	std::string command = formatStopCommand(pid);
	writeMessage(command);
        close(fifo_fd);
    }

    // Log a function's metrics
    void writelog(const std::string& functionName, 
		    unsigned int metrics, 
		    unsigned int upperlimit, 
		    unsigned int lowerlimit, 
		    unsigned int spike) {
        // Register a function
	if(fifo_fd == -1) {
		pid = pthread_self();
		fifo_name = "/tmp/flyt_shmem.log";
		initSharedMemory();
		if(send == 0) {
		    std::string command = formatStartCommand(pid, 1);
		    writeMessage(command);
		    send = 1;
		}
	}
	printf("logging\n");
	if(addFunctionName(functionName)) {
	    logFunctionName(functionName, upperlimit, lowerlimit, spike);
	}
        std::string command = formatLogCommand(pid, functionName, metrics);
	printf("writing\n");
        writeMessage(command);
	close(fifo_fd);
	fifo_fd = -1;
    }

private:
    pid_t pid;
    std::string fifo_name;
    int fifo_fd;
    int send;
    std::unordered_set<std::string> function_names;

    void initSharedMemory() {
        // Open the shared memory object
        fifo_fd = open(fifo_name.c_str(), O_WRONLY);
        if (fifo_fd == -1) {
            perror("shared memory file open");
            throw std::runtime_error("Failed to open shared memory");
        }
    }

    // Method to add a function name if it's unique
    bool addFunctionName(const std::string& functionName) {
        // Check if the function name is already in the set
        if (function_names.find(functionName) != function_names.end()) {
            return false;
        }
        // Add the function name to the set
        function_names.insert(functionName);
	return true;
    }

    void logFunctionName(std::string name, 
		    unsigned int upperlimit, 
		    unsigned int  lowerlimit, 
		    unsigned int count) {
        std::string command = formatFnInitCommand(pid, name, upperlimit, lowerlimit, count);
        writeMessage(command);
    }

    void writeMessage(const std::string& message) {
        if (write(fifo_fd, message.c_str(), message.size()) < message.size()) {
	    perror("Failed to write to fifo");
            throw std::runtime_error("Failed to write to fifo");
        }
    }

    static std::string formatStartCommand(unsigned int pid, 
		    unsigned int count ) {
        return "START " + std::to_string(pid) + " " + std::to_string(count) + "\n";
    }

    static std::string formatFnInitCommand(unsigned int pid, 
		    const std::string& functionName, 
		    unsigned int upperThreshold, 
		    unsigned int lowerThreshold, 
		    unsigned int spikeCount) {
        return "FN_INIT " + std::to_string(pid) + " " + functionName + " " +
               std::to_string(upperThreshold) + " " + std::to_string(lowerThreshold) + " " +
               std::to_string(spikeCount) + "\n";
    }

    static std::string formatLogCommand(unsigned int pid, 
		    const std::string& functionName, 
		    unsigned int metrics) {
        return "LOG " + std::to_string(pid) + " " + functionName + " " + std::to_string(metrics) + "\n";
    }
    
    static std::string formatStopCommand(unsigned int pid) {
        return "STOP " + std::to_string(pid) + "\n";
    }

};

#endif // SHARED_MEMORY_LOGGER_H

