#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// 默认参数
#define BLOCK_SIZE (8 * 1024 * 1024)          // 8MB per block
#define NUM_BLOCKS 237
#define TOTAL_SIZE (BLOCK_SIZE * NUM_BLOCKS)  // 总大小

// 检查 CUDA 调用返回值
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

void measure_vllm(int iterations) {
    printf("vllm 模式: 每次传输单个 block (%d bytes)，共 %d 个 block，每次迭代执行 %d 次\n",
           BLOCK_SIZE, NUM_BLOCKS, iterations);

    // 分配一个 pinned 内存的 host block（复用同一个内存块）
    float* h_block = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_block, BLOCK_SIZE));

    // 初始化 host block 数据（可选）
    for (size_t i = 0; i < BLOCK_SIZE / sizeof(float); ++i) {
        h_block[i] = static_cast<float>(i);
    }

    // 预热 GPU
    {
        float* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, BLOCK_SIZE));
        CUDA_CHECK(cudaMemcpy(d_tmp, h_block, BLOCK_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float totalTimeMs = 0.0f;
    size_t totalBytes = (size_t)BLOCK_SIZE * NUM_BLOCKS * iterations;

    // 循环 iterations，每个 iteration 对每个 block 单独调用 memcpy
    for (int iter = 0; iter < iterations; iter++) {
        for (int blk = 0; blk < NUM_BLOCKS; blk++) {
            float* d_block = nullptr;
            CUDA_CHECK(cudaMalloc(&d_block, BLOCK_SIZE));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, 0));
            CUDA_CHECK(cudaMemcpy(d_block, h_block, BLOCK_SIZE, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsedMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
            totalTimeMs += elapsedMs;

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_block));
        }
    }

    float avgTimePerBlockMs = totalTimeMs / (iterations * NUM_BLOCKS);
    float speedGBs = (float)totalBytes / (totalTimeMs / 1000.0f) / 1e9f;

    printf("vllm 模式: 总传输 %0.2f GB, 总耗时 %0.2f s, 平均每个 block 耗时 %0.2f ms\n",
           (float)totalBytes / 1e9f, totalTimeMs / 1000.0f, avgTimePerBlockMs);
    printf("传输速率: %0.2f GB/s\n", speedGBs);

    CUDA_CHECK(cudaFreeHost(h_block));
}

void measure_dattn(int iterations) {
    printf("dattn 模式: 一次性传输连续区域，总大小 %d bytes\n", TOTAL_SIZE);

    // 分配连续的 pinned 内存，大小为 TOTAL_SIZE
    float* h_large = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_large, TOTAL_SIZE));

    // 初始化数据（可选）
    for (size_t i = 0; i < TOTAL_SIZE / sizeof(float); ++i) {
        h_large[i] = static_cast<float>(i);
    }

    // 在 GPU 上分配相同大小的连续内存
    float* d_large = nullptr;
    CUDA_CHECK(cudaMalloc(&d_large, TOTAL_SIZE));

    // 预热 GPU
    CUDA_CHECK(cudaMemcpy(d_large, h_large, TOTAL_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTimeMs = 0.0f;
    size_t totalBytes = (size_t)TOTAL_SIZE * iterations;

    for (int iter = 0; iter < iterations; iter++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(d_large, h_large, TOTAL_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        totalTimeMs += elapsedMs;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    float avgTimeMs = totalTimeMs / iterations;
    float speedGBs = (float)totalBytes / (totalTimeMs / 1000.0f) / 1e9f;

    printf("dattn 模式: 总传输 %0.2f GB, 总耗时 %0.2f s, 平均耗时 %0.2f ms\n",
           (float)totalBytes / 1e9f, totalTimeMs / 1000.0f, avgTimeMs);
    printf("传输速率: %0.2f GB/s\n", speedGBs);

    CUDA_CHECK(cudaFree(d_large));
    CUDA_CHECK(cudaFreeHost(h_large));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s [vllm|dattn] [iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* mode = argv[1];
    int iterations = 1;
    if (argc >= 3) {
        iterations = atoi(argv[2]);
    }

    if (strcmp(mode, "vllm") == 0) {
        measure_vllm(iterations);
    } else if (strcmp(mode, "dattn") == 0) {
        measure_dattn(iterations);
    } else {
        printf("Unknown mode: %s\n", mode);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}