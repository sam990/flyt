#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <limits>
#include <cstdio>
#include <ctime>
#include <condition_variable>

#include <toml.hpp>

#define CLMGR_CONFIG_PATH "../configs/client-mgr.toml"  // Replace with your actual path

#include "logger.h"

#define MAX_N 16384

#define GRID_SIZE 64
#define BLOCK_SIZE 192
#define NUM_LAUNCHES 1
#define MAX_NUM_LAUNCHES 64
#define LAUNCH_TIME_THRESHOLD 10.0f // Time threshold in milliseconds

std::string SERVER_IP;
#define SERVER_PORT 32578      // The same port as in the Rust server
int client_id = -1;

int sm_cores = 8;

// Mutex to protect shared variables
std::mutex mtx, cv_mtx;
std::condition_variable cv;
bool operationCompleted = false;

int new_matrix_size = MAX_N, new_block_size = BLOCK_SIZE;  // Default parameters
int new_numLaunches = NUM_LAUNCHES;
int new_cmd = 6;  // Default command
unsigned char new_args[10];  // Default args

std::streambuf* original_cout_buffer = nullptr; // To store the original stream buffer

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
    std::ofstream console_file("/dev/tty", std::ios::app);
    std::ostream console_out(console_file.rdbuf());  // U

    console_out.flush();

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

    console_out << "Sending msg "<< (int)message[0] << ", " << (int)message[1] <<" to the server!\n";

    // Send a single byte to the server
    ssize_t bytes_sent = send(sock, message, sizeof(message[0])*2, 0);
    if (bytes_sent < 0) {
        std::cerr << "Send failed\n";
        close(sock);
        return EXIT_FAILURE;
    }

    console_out << "Sent message: " << (int)message[0] << std::endl;

    // Now wait for a response from the server
    unsigned char response[1];  // Buffer to hold the response from the server
    ssize_t bytes_received = recv(sock, response, sizeof(response), 0);
    if (bytes_received == -1) {
        std::cerr << "Error receiving response: " << strerror(errno) << std::endl;
        return -1;  // Return error if receiving failed
    }

    console_out << "Received response: " << (int)response[0] << std::endl;
    console_out.flush();

    close(sock);

    return 0;
}

int get_input(int *matrix_size, int *block_size, int *cmd, unsigned char *args, int *numLaunches) {
    // Step 4: Create a new `std::ostream` instance to write to `/dev/tty` (console)
    std::ofstream console_file("/dev/tty", std::ios::app);
    std::ostream console_out(console_file.rdbuf());  // U

    console_out.flush();
    //std::ostream console_out(original_cout_buffer);
    console_out << "\tPlease select an option:\n";
    console_out << "\t\t1. Increase SM cores by 8\n";
    console_out << "\t\t2. Decrease SM cores by 8\n";
    console_out << "\t\t3. Enable Horizontal scale\n";
    console_out << "\t\t4. Migrate SMs\n";
    console_out << "\t\t5. Change matrix and block size\n";
    console_out << "\t\t6. Continue kernel execution based on previous specs\n";
    console_out << "\t\t7. Exit application\n";
    console_out << "\t\t8. Get current reserved SMs for this app\n";
    console_out << "\t\t9. Disable Horizontal scale\n";
    console_out << "\t\t10. Change Num launches\n";
    console_out << "\tEnter your choice (1-10): ";

    int max_option = 10;

    console_out.flush();

    std::cin >> *cmd;
    int old_matrix_size = *matrix_size, old_block_size = *block_size, old_numLaunches = *numLaunches;

    // Handle invalid input
    if (std::cin.fail() || *cmd < 1 || *cmd > max_option) {
        console_out << "Invalid selection. Please enter a number between 1 and 5.\n";
	console_out.flush();
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
            console_out << "Enter new matrix size ( 1 - " << MAX_N << "): ";
	    console_out.flush();
            std::cin >> *matrix_size;
            if (*matrix_size < 1 || *matrix_size > MAX_N) {
		*matrix_size = old_matrix_size;
                console_out << "Matrix size out of range retaing old size!" << *matrix_size << std::endl;
            }

            console_out << "Enter new block size ( 1 - " << BLOCK_SIZE << "): ";
	    console_out.flush();
            std::cin >> *block_size;
            if (*block_size < 1 || *block_size > BLOCK_SIZE) {
		*block_size = old_block_size;
                console_out << "Block size out of range retaing old size!" << *block_size << std::endl;
            }
	    *cmd = 6;
            return 0;
        case 10:
            console_out << "Enter new number of iterations ( 1- " << MAX_NUM_LAUNCHES << "): ";
	    console_out.flush();
            std::cin >> *numLaunches;
            if (*numLaunches < 1 || *numLaunches > MAX_NUM_LAUNCHES) {
		*numLaunches = old_numLaunches;
                console_out << "No of launches out of range retaing old size!" << *numLaunches << std::endl;
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

__global__ void matrixMulKernel(long* C, long* A, long* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        long value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


void matrixMultiplication(cudaStream_t *stream, int block_size, long *A, long *B, long *C, int width, int numLaunches) {
    cudaError_t err;

    // Define block and grid dimensions
    // Determine block and grid sizes
    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size, (width + block_size - 1) / block_size);

    //std::cout << "Kernel launching with Grid size = " << grid.x
    //          << ", Block size = " << block.x << std::endl;

    // Launch the kernel
    for (int i = 0; i< numLaunches; i++)
    	matrixMulKernel<<<grid, block, 0, stream[i]>>>(A, B, C, width);

    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Matrix Multiplication Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Synchronize stream to ensure the kernel is completed
    for (int i = 0; i< numLaunches; i++)
        cudaStreamSynchronize(stream[i]);

    // Check for errors after synchronization
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Sync Error: " << cudaGetErrorString(err) << std::endl;
    }
}

// Function to launch kernels in a specific CUDA stream
void kernelLaunchThread(int numLaunches, float launchTimeThreshold) {
    int width = MAX_N;

    // Allocate matrices on host
    long *h_A = new long[width * width];
    long *h_B = new long[width * width];
    long *h_C = new long[width * width];

    // Initialize matrices A and B with random values
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // Allocate device memory for matrices
    long *d_A, *d_B, *d_C;

    // Allocate memory for device arrays
    CHECK_CUDA(cudaMalloc(&d_A, width * width * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_B, width * width * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_C, width * width * sizeof(long)));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, width * width * sizeof(long), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, width * width * sizeof(long), cudaMemcpyHostToDevice));

    // Create a stream for the kernel
    cudaEvent_t startEvent, stopEvent;
    cudaStream_t stream[MAX_NUM_LAUNCHES];


    for (int i = 0; i < MAX_NUM_LAUNCHES; i++)
        CHECK_CUDA(cudaStreamCreate(&stream[i]));

    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    int matrix_size = width, block_size = BLOCK_SIZE, cmd = -1;
    numLaunches = NUM_LAUNCHES;
    unsigned char args;

    std::ofstream console_file("/dev/tty", std::ios::app);
    std::ostream console_out(console_file.rdbuf());  // U

    console_out.flush();
    while (true) {
	// Notify and wake up if any,
	{
	    std::unique_lock<std::mutex> lock(cv_mtx);
	    operationCompleted = true;
        }
	cv.notify_one(); //
            printf("notifying  \n");
	std::chrono::time_point<std::chrono::high_resolution_clock> start;

	{
            // Lock the mutex before accessing shared variables
            std::lock_guard<std::mutex> lock(mtx);

	    matrix_size = new_matrix_size;
	    block_size = new_block_size;
	    cmd = new_cmd;
	    numLaunches = new_numLaunches;
	    args = new_args[0];

	    /* Reset the command to execute */
	    new_cmd = 6;
	}

        console_out << "Command selected: " << cmd << "\n";
	start = std::chrono::high_resolution_clock::now();
	unsigned char message[2];
        switch (cmd ) {
	    case 1: // Incrase SM cores
		    {
       			console_out << "Increasing SM cores by 8.\n";
		        message[0] = {1u};
		        message[1] = args;
		        send_msg(message);
			sm_cores = sm_cores + 8;
			if (sm_cores > 64) //ub-12
				sm_cores = 64;

			// Sleep to prevent overloading the CPU while waiting for input
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			cudaDeviceSynchronize();
			continue;
		    }
		    break;
	    case 2: // Decrease SM cores
		    {
       			console_out << "Decreasing SM cores by 8.\n";
		        message[0] = {3u};
		        message[1] = args;
		        send_msg(message);
			sm_cores = sm_cores - 8;
			if (sm_cores < 4) //ub-12
				sm_cores = 4;

			// Sleep to prevent overloading the CPU while waiting for input
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			cudaDeviceSynchronize();
			continue;
		    }
		    break;
	    case 3: // Horizontal scale
		    {
           		console_out << "Enable Horizontal scaling.\n";
		        message[0] = {4u};
		        message[1] = 1;
		        send_msg(message);
		    }
		    console_out << "Any new application would now be horizontaly scaled.\n";
		    break;
	    case 9: // Disable Horizontal scale
		    {
           		console_out << "Disable Horizontal scaling.\n";
		        message[0] = {4u};
		        message[1] = '0';
		        send_msg(message);
		    }
		    console_out << "Any new application would now be on grouped VM.\n";
		    break;
	    case 4: // migrate
		    {
           		console_out << "Migrating SMs.\n";
		        message[0] = {2u};
		        message[1] = args;
		        send_msg(message);
		    }
		    break;
	    case 8:
		    console_out << "current used SMs are " << get_active_sm_count() << std::endl;
		    break;
	    default: // New grid size;
    		    matrixMultiplication(stream, block_size, d_A, d_B, d_C, matrix_size, numLaunches);

		    // Copy the result matrix from device to host
                    cudaMemcpy(h_C, d_C, width * width * sizeof(long), cudaMemcpyDeviceToHost);

		    break;
        }

	if (cmd == 7) {
            console_out << "Exiting application.\n";
	    break;
        }

	auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        float elapsedTime;
	std::chrono::duration<double> diff = end - start;
	elapsedTime = (float) (1000 * diff.count());

	// Convert to time_t for human-readable format
        std::time_t currentTime = std::chrono::system_clock::to_time_t(end);

        // Print the formatted time
	std::cout << "Current time: " << std::put_time(std::localtime(&currentTime), "%Y-%m-%d %H:%M:%S")  \
	        << " matrix_size: " << matrix_size  \
		<< " block_size: " << block_size  \
		<< " Loop_count: " << numLaunches \
		<< " SM_cores: " << sm_cores \
		<< " elapsedTime_(ms): " << elapsedTime <<std::endl;

	console_out.flush();
	std::cout.flush();

	cudaDeviceSynchronize();


    }

    for (int i = 0; i < MAX_NUM_LAUNCHES; i++)
      CHECK_CUDA(cudaStreamDestroy(stream[i]));

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int get_input_hardcoded(int *matrix_size, int *block_size, int *cmd, unsigned char *args, int *numLaunches) {
    static int matrix_idx = 0, gpu_idx = 0, launch_idx = 0;
    int gpu_array[] = {8, 16, 24, 32, 40, 48, 56, 64, 56, 48, 40, 32, 24, 8};
    int matrix_array[] = {2048, 4096, 6144, 7168, 8192, 9216, 10240, 12288};
    int launch_array[] = {1, 4};

    /* Fixed values */
    *args = client_id;
    *block_size = 8;

    // Get current values
    int gpu = gpu_array[gpu_idx];
    int matrix = matrix_array[matrix_idx];
    int launch = launch_array[launch_idx];

    printf("gpu %d sm_core %d, matrix %d matrix_size %d numLaunces %d launch %d\n",
		    gpu, sm_cores, matrix, *matrix_size, *numLaunches, launch);
    if (sm_cores > gpu) {
	    *cmd = 2;
	    return 0;
    }
    else if (sm_cores < gpu) {
	    *cmd = 1;
	    return 0;
    }

    if (matrix != *matrix_size) {
	    *cmd = 5;
	    *matrix_size = matrix;
    }

    *numLaunches = launch;
    *cmd = 6;

    // Increment indices in nested order
    launch_idx++;
    if (launch_idx >= sizeof(launch_array) / sizeof(launch_array[0])) {
        launch_idx = 0; // Reset launch index
        matrix_idx++;
        if (matrix_idx >= sizeof(matrix_array) / sizeof(matrix_array[0])) {
            matrix_idx = 0; // Reset matrix index
            launch_idx = 0; // Reset launch index
            gpu_idx++;
            if (gpu_idx >= sizeof(gpu_array) / sizeof(gpu_array[0])) {
                *cmd = 7; // Exit application
            }
        }
    }

    return 0;
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

    int matrix_size = MAX_N, block_size = 8, numLaunches = NUM_LAUNCHES;  // Default parameters
    int cmd = 6;  // Default command
    unsigned char args[10];  // Default args

    std::ofstream file_out;                        // File output stream

    char *file_path = "flyt_log.txt";

    // Redirect std::cout to a file
    file_out.open(file_path, std::ios::out | std::ios::trunc);
    if (!file_out.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    // Save the original buffer and redirect std::cout
    original_cout_buffer = std::cout.rdbuf();
    std::cout.rdbuf(file_out.rdbuf());

    // Redirect C printf output to the same log file
    freopen(file_path, "a", stdout);  // "a" to append
    freopen(file_path, "a", stderr);  // "a" to append


    // Launch kernel functions in multiple threads with separate CUDA streams
    std::thread loopThread( kernelLaunchThread, NUM_LAUNCHES, LAUNCH_TIME_THRESHOLD);

    while (true) {
        int ret = get_input_hardcoded(&matrix_size, &block_size, &cmd, args, &numLaunches);

	{
            // Lock the mutex before accessing shared variables
            std::lock_guard<std::mutex> lock(mtx);

	    new_matrix_size = matrix_size;
	    new_numLaunches = numLaunches;
	    new_block_size = block_size;
	    new_cmd = cmd;
	    new_args[0] = args[0];
	}

	// wait for 5 reading values to be completed before next change 
	for (int i = 0; i < 10; i++ )
	// wait for other thread to consume the command.
	{
            printf("I am waiting %d\n", i);
	    std::unique_lock<std::mutex> lock(cv_mtx);
	    operationCompleted = false;
            cv.wait(lock, [] { return operationCompleted; }); // Wait until ready becomes true
	}

	if (ret != 0)
		break; // Exit loop

        // Sleep to prevent overloading the CPU while waiting for input
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Join the input thread before exiting
    loopThread.join();

    return 0;
}

