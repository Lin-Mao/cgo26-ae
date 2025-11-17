#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    size_t free_size, total_size;

    while (true) {
        // Sleep for 0.5 seconds
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        cudaMemGetInfo(&free_size, &total_size);

        std::cout << "Free memory: "
                  << static_cast<float>(free_size) / (1024 * 1024) << " MiB ("
                  << free_size << " bytes, "
                  << static_cast<float>(free_size) / (1024 * 1024 * 1024) << " GB)" 
                  << std::endl;
    }

    return 0;
}