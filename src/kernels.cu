// kernels.cu
// CUDA kernels for gpu_image_filter: separable Gaussian blur, Sobel edge
// detection, and a 3x3 sharpening convolution. All kernels operate on
// interleaved 8-bit RGB images and use clamp-to-edge boundary handling.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "kernels.cuh"

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
              cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

namespace {

constexpr int kMaxRadius = 32;               // Supports sigma up to ~10.7.
constexpr int kMaxDiameter = 2 * kMaxRadius + 1;
__constant__ float c_weights[kMaxDiameter];  // 1D Gaussian weights.

__device__ __forceinline__ int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ unsigned char ClampToByte(float v) {
  return static_cast<unsigned char>(ClampInt(static_cast<int>(v + 0.5f), 0,
                                              255));
}

// ---------------------------------------------------------------------
// Gaussian blur: horizontal pass (uint8 RGB -> float RGB)
// ---------------------------------------------------------------------
__global__ void GaussianHorizontalKernel(const unsigned char* in, float* out,
                                          int width, int height, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum[3] = {0.f, 0.f, 0.f};
  for (int k = -radius; k <= radius; ++k) {
    int sx = ClampInt(x + k, 0, width - 1);
    float w = c_weights[k + radius];
    const unsigned char* px = &in[(y * width + sx) * 3];
    sum[0] += w * px[0];
    sum[1] += w * px[1];
    sum[2] += w * px[2];
  }
  float* po = &out[(y * width + x) * 3];
  po[0] = sum[0];
  po[1] = sum[1];
  po[2] = sum[2];
}

// ---------------------------------------------------------------------
// Gaussian blur: vertical pass (float RGB -> uint8 RGB)
// ---------------------------------------------------------------------
__global__ void GaussianVerticalKernel(const float* in, unsigned char* out,
                                        int width, int height, int radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum[3] = {0.f, 0.f, 0.f};
  for (int k = -radius; k <= radius; ++k) {
    int sy = ClampInt(y + k, 0, height - 1);
    float w = c_weights[k + radius];
    const float* px = &in[(sy * width + x) * 3];
    sum[0] += w * px[0];
    sum[1] += w * px[1];
    sum[2] += w * px[2];
  }
  unsigned char* po = &out[(y * width + x) * 3];
  po[0] = ClampToByte(sum[0]);
  po[1] = ClampToByte(sum[1]);
  po[2] = ClampToByte(sum[2]);
}

// ---------------------------------------------------------------------
// Sobel edge detection
// ---------------------------------------------------------------------
__global__ void GrayscaleKernel(const unsigned char* in, float* gray,
                                 int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const unsigned char* px = &in[(y * width + x) * 3];
  gray[y * width + x] =
      0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2];
}

__global__ void SobelKernel(const float* gray, unsigned char* out, int width,
                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float p[3][3];
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int sx = ClampInt(x + dx, 0, width - 1);
      int sy = ClampInt(y + dy, 0, height - 1);
      p[dy + 1][dx + 1] = gray[sy * width + sx];
    }
  }
  float gx = (p[0][2] + 2.f * p[1][2] + p[2][2]) -
             (p[0][0] + 2.f * p[1][0] + p[2][0]);
  float gy = (p[2][0] + 2.f * p[2][1] + p[2][2]) -
             (p[0][0] + 2.f * p[0][1] + p[0][2]);
  float mag = sqrtf(gx * gx + gy * gy);
  unsigned char v = ClampToByte(mag);
  unsigned char* po = &out[(y * width + x) * 3];
  po[0] = v;
  po[1] = v;
  po[2] = v;
}

// ---------------------------------------------------------------------
// Sharpen (3x3 unsharp kernel), per channel
// ---------------------------------------------------------------------
__global__ void SharpenKernel(const unsigned char* in, unsigned char* out,
                               int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const float kKernel[3][3] = {
      {0.f, -1.f, 0.f}, {-1.f, 5.f, -1.f}, {0.f, -1.f, 0.f}};

  float sum[3] = {0.f, 0.f, 0.f};
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int sx = ClampInt(x + dx, 0, width - 1);
      int sy = ClampInt(y + dy, 0, height - 1);
      float w = kKernel[dy + 1][dx + 1];
      const unsigned char* px = &in[(sy * width + sx) * 3];
      sum[0] += w * px[0];
      sum[1] += w * px[1];
      sum[2] += w * px[2];
    }
  }
  unsigned char* po = &out[(y * width + x) * 3];
  po[0] = ClampToByte(sum[0]);
  po[1] = ClampToByte(sum[1]);
  po[2] = ClampToByte(sum[2]);
}

dim3 BlockDim() { return dim3(16, 16); }
dim3 GridDim(int width, int height) {
  dim3 block = BlockDim();
  return dim3((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
}

float TimedLaunch(const std::function<void()>& fn) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

}  // namespace

float LaunchGaussianBlur(const unsigned char* d_in, unsigned char* d_out,
                          int width, int height, float sigma) {
  int radius = static_cast<int>(ceilf(3.f * sigma));
  radius = radius < 1 ? 1 : (radius > kMaxRadius ? kMaxRadius : radius);
  int diameter = 2 * radius + 1;

  std::vector<float> weights(diameter);
  float sum = 0.f;
  for (int i = -radius; i <= radius; ++i) {
    float w = expf(-(i * i) / (2.f * sigma * sigma));
    weights[i + radius] = w;
    sum += w;
  }
  for (float& w : weights) w /= sum;
  CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights.data(),
                                 diameter * sizeof(float)));

  float* d_intermediate = nullptr;
  CUDA_CHECK(cudaMalloc(&d_intermediate,
                         sizeof(float) * width * height * 3));

  dim3 grid = GridDim(width, height);
  dim3 block = BlockDim();

  float ms = TimedLaunch([&]() {
    GaussianHorizontalKernel<<<grid, block>>>(d_in, d_intermediate, width,
                                               height, radius);
    GaussianVerticalKernel<<<grid, block>>>(d_intermediate, d_out, width,
                                             height, radius);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_intermediate));
  return ms;
}

float LaunchSobel(const unsigned char* d_in, unsigned char* d_out, int width,
                   int height) {
  float* d_gray = nullptr;
  CUDA_CHECK(cudaMalloc(&d_gray, sizeof(float) * width * height));

  dim3 grid = GridDim(width, height);
  dim3 block = BlockDim();

  float ms = TimedLaunch([&]() {
    GrayscaleKernel<<<grid, block>>>(d_in, d_gray, width, height);
    SobelKernel<<<grid, block>>>(d_gray, d_out, width, height);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_gray));
  return ms;
}

float LaunchSharpen(const unsigned char* d_in, unsigned char* d_out,
                     int width, int height) {
  dim3 grid = GridDim(width, height);
  dim3 block = BlockDim();
  float ms = TimedLaunch([&]() {
    SharpenKernel<<<grid, block>>>(d_in, d_out, width, height);
  });
  CUDA_CHECK(cudaGetLastError());
  return ms;
}
