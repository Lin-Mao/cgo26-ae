#include <iostream>
#include <cuda_runtime.h>
#include <csignal>
#include <cstdlib>


static size_t max_used_size = 0;
static size_t total_size = 0;

// Signal handler function
void handleSignal(int signal) {
    if (signal == SIGINT) {
        printf("----------------------------------------\n");
        printf("Memory capacity: %.3f MB (%lu bytes, %.3f GB)\n", (float)total_size / (1024 * 1024), total_size, (float)total_size / (1024 * 1024 * 1024));
        printf("Max used size:  %.3f MiB (%lu bytes, %.3f GB)\n", (float)(max_used_size) / (1024 * 1024), max_used_size, (float)(max_used_size) / (1024 * 1024 * 1024));
        printf("Min Free size:  %.3f MiB (%lu bytes, %.3f GB)\n", (float)(total_size - max_used_size) / (1024 * 1024), total_size - max_used_size, (float)(total_size - max_used_size) / (1024 * 1024 * 1024));
        std::exit(0); // Exit the program with success code
    }
}

int main() {
     // Register the signal handler for SIGINT (Ctrl+C)
    std::signal(SIGINT, handleSignal);

    std::cout << "Press Ctrl+C to exit." << std::endl;

    size_t free_size = 0;
    while (true) {
        cudaMemGetInfo(&free_size, &total_size);
        if (total_size - free_size > max_used_size) {
            max_used_size = total_size - free_size;
        }
    }

    return 0;
}