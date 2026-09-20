// main.cu
// gpu_image_filter: a small CUDA command-line tool that applies a GPU
// image-processing kernel (Gaussian blur, Sobel edge detection, or
// sharpen) to a PPM image.
//
// Usage:
//   gpu_image_filter --input <in.ppm> --output <out.ppm> --op {blur|sobel|sharpen}
//                     [--sigma <float>] [--log <path>]
//
// GPU Specialization Capstone Project.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "kernels.cuh"
#include "ppm_io.h"

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

struct Args {
  std::string input;
  std::string output;
  std::string op = "blur";
  float sigma = 2.0f;
  std::string log_path;
};

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog
      << " --input <in.ppm> --output <out.ppm> --op {blur|sobel|sharpen} "
         "[--sigma <float>] [--log <path>]\n"
      << "  --input   Path to a binary PPM (P6) input image (required)\n"
      << "  --output  Path to write the resulting PPM image (required)\n"
      << "  --op      Operation to run: blur, sobel, or sharpen "
         "(default: blur)\n"
      << "  --sigma   Gaussian sigma, only used by --op blur "
         "(default: 2.0)\n"
      << "  --log     Optional path to append a CSV log line with timing "
         "info\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto NeedsValue = [&](const char* flag) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "ERROR: " << flag << " requires a value\n";
        return false;
      }
      return true;
    };
    if (a == "--input") {
      if (!NeedsValue("--input")) return false;
      args->input = argv[++i];
    } else if (a == "--output") {
      if (!NeedsValue("--output")) return false;
      args->output = argv[++i];
    } else if (a == "--op") {
      if (!NeedsValue("--op")) return false;
      args->op = argv[++i];
    } else if (a == "--sigma") {
      if (!NeedsValue("--sigma")) return false;
      args->sigma = std::stof(argv[++i]);
    } else if (a == "--log") {
      if (!NeedsValue("--log")) return false;
      args->log_path = argv[++i];
    } else if (a == "--help" || a == "-h") {
      return false;
    } else {
      std::cerr << "ERROR: unknown argument '" << a << "'\n";
      return false;
    }
  }
  if (args->input.empty() || args->output.empty()) {
    std::cerr << "ERROR: --input and --output are required\n";
    return false;
  }
  if (args->op != "blur" && args->op != "sobel" && args->op != "sharpen") {
    std::cerr << "ERROR: --op must be one of blur, sobel, sharpen\n";
    return false;
  }
  return true;
}

void AppendLog(const std::string& log_path, const Args& args,
                const Image& image, float gpu_ms, double total_ms) {
  if (log_path.empty()) return;
  bool file_exists = std::ifstream(log_path).good();
  std::ofstream log(log_path, std::ios::app);
  if (!file_exists) {
    log << "input,output,op,sigma,width,height,gpu_ms,total_ms\n";
  }
  log << args.input << "," << args.output << "," << args.op << ","
      << args.sigma << "," << image.width << "," << image.height << ","
      << gpu_ms << "," << total_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "ERROR: no CUDA-capable GPU detected.\n";
    return EXIT_FAILURE;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (SM " << prop.major << "."
            << prop.minor << ")\n";

  Image image;
  if (!ReadPPM(args.input, &image)) return EXIT_FAILURE;
  std::cout << "Loaded " << args.input << " (" << image.width << "x"
            << image.height << ")\n";

  Image result;
  result.width = image.width;
  result.height = image.height;
  result.channels = 3;
  result.data.resize(image.data.size());

  unsigned char* d_in = nullptr;
  unsigned char* d_out = nullptr;
  size_t bytes = image.NumBytes();
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, image.data.data(), bytes,
                         cudaMemcpyHostToDevice));

  cudaEvent_t total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventRecord(total_start));

  float gpu_ms = 0.f;
  if (args.op == "blur") {
    gpu_ms = LaunchGaussianBlur(d_in, d_out, image.width, image.height,
                                 args.sigma);
  } else if (args.op == "sobel") {
    gpu_ms = LaunchSobel(d_in, d_out, image.width, image.height);
  } else {
    gpu_ms = LaunchSharpen(d_in, d_out, image.width, image.height);
  }

  CUDA_CHECK(cudaMemcpy(result.data.data(), d_out, bytes,
                         cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(total_stop));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  float total_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));

  if (!WritePPM(args.output, result)) return EXIT_FAILURE;

  std::cout << "Operation: " << args.op << "\n"
            << "Kernel time: " << gpu_ms << " ms\n"
            << "Total (H2D+kernel+D2H) time: " << total_ms << " ms\n"
            << "Wrote " << args.output << "\n";

  AppendLog(args.log_path, args, image, gpu_ms, total_ms);
  return EXIT_SUCCESS;
}
