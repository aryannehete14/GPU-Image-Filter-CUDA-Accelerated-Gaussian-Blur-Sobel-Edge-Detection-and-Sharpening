// kernels.cuh
// Host-callable launcher functions that wrap the CUDA kernels used by
// gpu_image_filter. Each function operates on device pointers already
// allocated/copied by the caller (see main.cu) and reports its own
// elapsed GPU time in milliseconds via the returned float, so the
// caller can log proof-of-execution timing per operation.
#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// Separable Gaussian blur (2-pass: horizontal then vertical) on an
// interleaved RGB uint8 image. `sigma` controls blur strength; the
// kernel radius is derived from sigma (radius = ceil(3*sigma)).
float LaunchGaussianBlur(const unsigned char* d_in, unsigned char* d_out,
                          int width, int height, float sigma);

// Sobel edge-magnitude detector. Internally converts to grayscale,
// applies the 3x3 Gx/Gy Sobel operators, and writes the gradient
// magnitude (clamped to [0,255]) replicated across all 3 channels.
float LaunchSobel(const unsigned char* d_in, unsigned char* d_out, int width,
                   int height);

// 3x3 unsharp-mask style sharpening filter, applied per channel.
float LaunchSharpen(const unsigned char* d_in, unsigned char* d_out,
                     int width, int height);

#endif  // KERNELS_CUH_
