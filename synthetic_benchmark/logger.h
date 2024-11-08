#ifndef SHARED_MEMORY_LOGGER_H
#define SHARED_MEMORY_LOGGER_H

#include <iostream>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_set>

#define INTERVAL 250 //ms
class FlytLogger {
public:
    // Constructor takes only the PID
    explicit FlytLogger(pid_t pid)
        : pid(pid), fifo_name("/tmp/flyt_shmem.log"), send(0) {
        // Open and initialize shared memory
        initSharedMemory();
	printf("Initializing thread\n");
        // Register the application
	std::string command = formatStartCommand(pid, INTERVAL);
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
		    int metrics, 
		    int upperlimit, 
		    int lowerlimit, 
		    int spike) {
        // Register a function
	if(fifo_fd == -1) {
		pid = getpid();
		fifo_name = "/tmp/flyt_shmem.log";
		initSharedMemory();
		if(send == 0) {
		    std::string command = formatStartCommand(pid, INTERVAL);
		    writeMessage(command);
		    send = 1;
		}
	}
	printf("logging\n");
	if(addFunctionName(functionName)) {
	    logFunctionName(functionName, upperlimit, lowerlimit, spike);
	}
	int avg = (upperlimit + lowerlimit)/2;
	float metrics_float = (float)(metrics - avg) / avg * 100;
	//metrics = (int)metrics_float;
	if(metrics > 200) {
		metrics = 200;
	}
        std::string command = formatLogCommand(pid, functionName, metrics);
	printf("writing metrics for pid: %4d %4d avg: %4d spike: %4d command %s\n", pid, metrics, avg, spike, command.c_str());
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
		    int upperlimit, 
		    int  lowerlimit, 
		    int count) {
        std::string command = formatFnInitCommand(pid, name, upperlimit, lowerlimit, count);
        writeMessage(command);
    }

    void writeMessage(const std::string& message) {
        if (write(fifo_fd, message.c_str(), message.size()) < message.size()) {
	    perror("Failed to write to fifo");
            throw std::runtime_error("Failed to write to fifo");
        }
    }

    std::string formatStartCommand(unsigned int pid, 
		    int count ) {
        return "START " + std::to_string(pid) + " " + std::to_string(count) + "\n";
    }

    std::string formatFnInitCommand(unsigned int pid, 
		    const std::string& functionName, 
		    int upperThreshold, 
		    int lowerThreshold, 
		    int spikeCount) {
        return "FN_INIT " + std::to_string(pid) + " " + functionName + " " +
               std::to_string(upperThreshold) + " " + std::to_string(lowerThreshold) + " " +
               std::to_string(spikeCount) + "\n";
    }

    std::string formatLogCommand(unsigned int pid, 
		    const std::string& functionName, 
		    int metrics) {
        return "LOG " + std::to_string(pid) + " " + functionName + " " + std::to_string(metrics) + "\n";
    }
    
    std::string formatStopCommand(unsigned int pid) {
        return "STOP " + std::to_string(pid) + "\n";
    }

};

#endif // SHARED_MEMORY_LOGGER_H

