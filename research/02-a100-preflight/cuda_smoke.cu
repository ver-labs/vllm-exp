#include <cuda_runtime.h>
#include <cstdio>

__global__ void write_value(int* value) { *value = 42; }

#define CHECK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
    return 1; \
  } \
} while (0)

int main() {
  cudaDeviceProp properties;
  CHECK(cudaGetDeviceProperties(&properties, 0));
  std::printf("Device: %s; capability: %d.%d\n", properties.name,
              properties.major, properties.minor);
  if (properties.major != 8 || properties.minor != 0) {
    std::fprintf(stderr, "Expected the allocated A100-class SM80 device.\n");
    return 1;
  }
  int* device_value = nullptr;
  CHECK(cudaMalloc(&device_value, sizeof(int)));
  write_value<<<1, 1>>>(device_value);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  int value = 0;
  CHECK(cudaMemcpy(&value, device_value, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(device_value));
  if (value != 42) return 1;
  std::puts("SM80 compile, kernel launch, synchronization and copy: PASS");
  return 0;
}
