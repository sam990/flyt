#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <string>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <limits>

#include <toml.hpp>

#define CLMGR_CONFIG_PATH "../configs/client-mgr.toml"  // Replace with your actual path

#include "logger.h"

#define GRID_SIZE 64
#define BLOCK_SIZE 192
#define ITERATIONS 10000
#define NUM_THREADS 1
#define NUM_LAUNCHES 40
#define LAUNCH_TIME_THRESHOLD 10.0f // Time threshold in milliseconds

std::string SERVER_IP;
#define SERVER_PORT 32578      // The same port as in the Rust server
int client_id = -1;


#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA Runtime API error: " << cudaGetErrorString(err)   \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

std::string get_server_ip(const std::string& config_path) {
    try {
        // Load the TOML configuration file
        auto config = toml::parse(config_path);

        auto ipaddr = toml::find<std::string>(config, "resource-manager", "address");
        return ipaddr;

    } catch (const toml::syntax_error & err) {
        std::cerr << "Failed to parse config file: " << err.what() << std::endl;
        return "";
    } catch (const std::out_of_range& e) {
        std::cerr << "Key not found in config: " << e.what() << std::endl;
        return "";
    }
}

int send_msg(unsigned char message[2]) {
    struct sockaddr_in server_addr;

    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation error\n";
        return EXIT_FAILURE;
    }

    // Set up server address structure
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);

    // Convert and set IP address
    if (inet_pton(AF_INET, SERVER_IP.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address or address not supported\n";
        close(sock);
        return EXIT_FAILURE;
    }

    // Connect to server
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed\n";
        close(sock);
        return EXIT_FAILURE;
    }

    std::cout << "Sending msg "<< (int)message[0] << ", " << (int)message[1] <<" to the server!\n";

    // Send a single byte to the server
    ssize_t bytes_sent = send(sock, message, sizeof(message[0])*2, 0);
    if (bytes_sent < 0) {
        std::cerr << "Send failed\n";
        close(sock);
        return EXIT_FAILURE;
    }

    std::cout << "Sent message: " << (int)message[0] << std::endl;

    // Now wait for a response from the server
    unsigned char response[1];  // Buffer to hold the response from the server
    ssize_t bytes_received = recv(sock, response, sizeof(response), 0);
    if (bytes_received == -1) {
        std::cerr << "Error receiving response: " << strerror(errno) << std::endl;
        return -1;  // Return error if receiving failed
    }

    std::cout << "Received response: " << (int)response[0] << std::endl;

    close(sock);

    return 0;
}

int get_input(int *grid_size, int *block_size, int *cmd, unsigned char *args) {
    std::cout << "\tPlease select an option:\n";
    std::cout << "\t\t1. Increase SM cores by 8\n";
    std::cout << "\t\t2. Decrease SM cores by 8\n";
    std::cout << "\t\t3. Enable Horizontal scale\n";
    std::cout << "\t\t4. Migrate SMs\n";
    std::cout << "\t\t5. Change grid and block size\n";
    std::cout << "\t\t6. Continue kernel execution based on previous specs\n";
    std::cout << "\t\t7. Exit application\n";
    std::cout << "\t\t8. Get current reserved SMs for this app\n";
    std::cout << "\t\t9. Disable Horizontal scale\n";
    std::cout << "\tEnter your choice (1-8): ";

    std::cin >> *cmd;
    int old_grid_size = *grid_size, old_block_size = *block_size;
    //*grid_size = GRID_SIZE;
    //*block_size = BLOCK_SIZE;

    // Handle invalid input
    if (std::cin.fail() || *cmd < 1 || *cmd > 9) {
        std::cout << "Invalid selection. Please enter a number between 1 and 5.\n";
        std::cin.clear(); // Clear the error flag
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
	*cmd = 6;
	return 0;
    }

    switch (*cmd) {
        case 1: // Increase SM cores
        case 2: // Decrease SM cores
        case 3: // Enable horizontal scaling
        case 9: // Disable horizontal scaling
        case 4: // Migrate application
	    *args = client_id;
            return 0;
        case 5:
            std::cout << "Enter new grid size ( 1 - " << GRID_SIZE << "): ";
            std::cin >> *grid_size;
            if (*grid_size < 1 || *grid_size > GRID_SIZE) {
		*grid_size = old_grid_size;
                std::cout << "Grid size out of range retaing old size!" << *grid_size << std::endl;
            }

            std::cout << "Enter new block size ( 1 - " << BLOCK_SIZE << "): ";
            std::cin >> *block_size;
            if (*block_size < 1 || *block_size > BLOCK_SIZE) {
		*block_size = old_block_size;
                std::cout << "Block size out of range retaing old size!" << *grid_size << std::endl;
            }
	    *cmd = 6;
            return 0;

        case 6: // Execute kernel
	    *cmd = 6;
            return 0;
        case 7: // Exit
            return -1;
        case 8: // Get SM cores
            return 0;
	default:
	    break;
    }

    *cmd = 6;
    return 0;

}

int get_active_sm_count() {

	int count = 0;
	CHECK_CUDA(cudaGetDeviceCount(&count));

	return count;
}

// Define the kernel with arithmetic operations
__global__ void workload_kernel(long *data) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x + threadIdx.x + 32;

    for (long k = 0; k < ITERATIONS; k++) {
        for (long j = 0; j < ITERATIONS; j++) {
            i += b;
            i -= b;
            i *= b;
        }
    }

    data[idx] += 2 * i;
    // d_a[idx] += i + d_b[idx];
}

cudaEvent_t startEvent, stopEvent;
long *d_a, *d_b, *d_c;

void kernelOperation (cudaStream_t stream, int grid_size, int block_size) {
        cudaError_t err;

	std::cout << "Kernel launching with Grid size = "<<grid_size << " Block size = " << block_size << std::endl;

        // Launch the kernel
        workload_kernel<<<grid_size, block_size, 0, stream>>>(d_a);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA add Error: " << cudaGetErrorString(err) << std::endl;
	    //cudaStreamDestroy(stream);
	    //cudaStreamCreate(&stream);
            //return ;
        }
        cudaStreamSynchronize(stream); // Ensure the kernel launch is completed
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA sync Error: " << cudaGetErrorString(err) << std::endl;
	    //cudaStreamDestroy(stream);
	    //cudaStreamCreate(&stream);
            //return ;
        }
}


// Function to launch kernels in a specific CUDA stream
void kernelLaunchFunction(int numLaunches, float launchTimeThreshold) {
    cudaStream_t stream;

    // Allocate memory for device arrays
    CHECK_CUDA(cudaMalloc(&d_a, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_b, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_c, GRID_SIZE * BLOCK_SIZE * sizeof(long)));

    // Initialize device arrays
    CHECK_CUDA(cudaMemset(d_a, 0, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMemset(d_b, 1, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMemset(d_c, 0, GRID_SIZE * BLOCK_SIZE * sizeof(long)));

    // Convert the current thread ID to a string
    std::stringstream ss;
    ss << std::this_thread::get_id();
    std::string filename = ss.str() + ".txt";
    
    // Open the file in write mode
    std::ofstream file(filename, std::ios::out | std::ios::app);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
 

	CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

    int grid_size = GRID_SIZE, block_size = BLOCK_SIZE, cmd = -1;
    unsigned char args;
    //file << "Thread id "<< pthread_self() << " flag " << flag << std::endl;
    while (true) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start;

	if (get_input(&grid_size, &block_size, &cmd, &args) == 0) {
            std::cout << "Command selected: " << cmd << "\n";
	    start = std::chrono::high_resolution_clock::now();
	    unsigned char message[2];
            switch (cmd ) {
		    case 1: // Incrase SM cores
			    {
            			std::cout << "Increasing SM cores by 8.\n";
			        message[0] = {1u};
			        message[1] = args;
			        send_msg(message);
			    }
			    break;
		    case 2: // Decrease SM cores
			    {
            			std::cout << "Decreasing SM cores by 8.\n";
			        message[0] = {3u};
			        message[1] = args;
			        send_msg(message);
			    }
			    break;
		    case 3: // Horizontal scale
			    {
            			std::cout << "Enable Horizontal scaling.\n";
			        message[0] = {4u};
			        message[1] = 1;
			        send_msg(message);
			    }
			    std::cout << "Any new application would now be horizontaly scaled.\n";
			    break;
		    case 9: // Disable Horizontal scale
			    {
            			std::cout << "Disable Horizontal scaling.\n";
			        message[0] = {4u};
			        message[1] = '0';
			        send_msg(message);
			    }
			    std::cout << "Any new application would now be on grouped VM.\n";
			    break;
		    case 4: // migrate
			    {
            			std::cout << "Migrating SMs.\n";
			        message[0] = {2u};
			        message[1] = args;
			        send_msg(message);
			    }
			    break;
		    case 8:
			    std::cout << "current used SMs are " << get_active_sm_count() << std::endl;
			    break;
		    default: // New grid size;
	    		    kernelOperation(stream, grid_size, block_size);
			    break;
            }
        } else {
            std::cout << "Exiting application.\n";
	    break;
        }

	auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        float elapsedTime;
	std::chrono::duration<double> diff = end - start;
	elapsedTime = (float) (1000 * diff.count());

	std::cout << "elapsedTime (ms) " << elapsedTime <<std::endl;

    }

    CHECK_CUDA(cudaStreamDestroy(stream));

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    try {
        SERVER_IP = get_server_ip(CLMGR_CONFIG_PATH);
        std::cout << "Server IP: " << SERVER_IP << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Enter application client-id: ";
    std::cin >> client_id;
    // Launch kernel functions in multiple threads with separate CUDA streams
    kernelLaunchFunction( NUM_LAUNCHES, LAUNCH_TIME_THRESHOLD);

    return 0;
}

