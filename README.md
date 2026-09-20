# GPU Image Filter — CUDA at Scale Capstone Project

A CUDA command-line tool that applies GPU-accelerated image-processing
filters — **Gaussian blur**, **Sobel edge detection**, and **sharpen** — to
images, using custom CUDA kernels (no NPP/OpenCV image-processing calls;
the convolution, grayscale conversion, and edge-detection math are
hand-written CUDA C++).

Built for the GPU Specialization Capstone Project.

---

## 1. Overview

`gpu_image_filter` is a small, self-contained CLI tool. It:

1. Loads an 8-bit RGB image (PPM format).
2. Copies it to GPU memory.
3. Runs one of three CUDA kernels on it:
   - **Gaussian blur** — separable 2-pass convolution (horizontal then
     vertical), with the 1D Gaussian weights computed on the host and
     uploaded to `__constant__` GPU memory for fast, cached reads.
   - **Sobel edge detection** — grayscale conversion kernel followed by a
     3×3 Sobel Gx/Gy convolution kernel that outputs gradient magnitude.
   - **Sharpen** — a 3×3 unsharp-mask convolution kernel applied per
     color channel.
4. Copies the result back to the host and writes it to disk, reporting
   GPU kernel time and total (transfer + compute) time.

It can be run once on a single image, or in a batch over every image in
`data/` and every operation (via `run.sh`) to generate proof-of-execution
artifacts.

## 2. Repository layout

```
.
├── Makefile              # Build rules (make / make clean)
├── run.sh                # Builds + batch-runs on all images in data/
├── README.md             # This file
├── src/
│   ├── main.cu            # CLI parsing, host orchestration, timing, I/O
│   ├── kernels.cuh         # Kernel launcher declarations
│   ├── kernels.cu          # __global__ CUDA kernels (blur/sobel/sharpen)
│   ├── ppm_io.h             # PPM image struct + read/write declarations
│   └── ppm_io.cu             # PPM read/write implementation
├── data/
│   └── README.md         # Where to get sample images + how to convert them
└── output/                # Generated images + log.csv land here (gitignored placeholder kept via .gitkeep)
```

## 3. Requirements

- An NVIDIA GPU + CUDA Toolkit (`nvcc`) — developed/tested against CUDA
  10.x–12.x. This matches the Coursera CUDA lab environment.
- A C++14-capable host compiler (installed automatically with the CUDA
  Toolkit).
- (Optional) [ImageMagick](https://imagemagick.org/) to convert
  PNG/JPEG sample images into the PPM format this tool reads.

No other third-party libraries are required — image I/O is a small
hand-written binary-PPM reader/writer (`src/ppm_io.*`), so there is
nothing extra to install on the GPU lab machine.

## 4. Build

```bash
make
```

This produces a `gpu_image_filter` binary in the project root. To target
a specific GPU compute capability (default is `sm_50`, which is broadly
compatible):

```bash
make ARCH=sm_75      # e.g. for a Turing GPU
```

To clean build artifacts and generated outputs:

```bash
make clean
```

## 5. Usage (CLI)

```
./gpu_image_filter --input <in.ppm> --output <out.ppm> --op {blur|sobel|sharpen} [--sigma <float>] [--log <path>]
```

| Flag       | Required | Description                                              |
|------------|----------|------------------------------------------------------------|
| `--input`  | Yes      | Path to a binary PPM (P6) input image                     |
| `--output` | Yes      | Path to write the resulting PPM image                     |
| `--op`     | No       | `blur`, `sobel`, or `sharpen` (default: `blur`)            |
| `--sigma`  | No       | Gaussian sigma, used only by `--op blur` (default: `2.0`) |
| `--log`    | No       | Path to a CSV file to append timing info to (for proof of execution across many runs) |

### Example

```bash
./gpu_image_filter --input data/lena.ppm --output output/lena_blur.ppm \
  --op blur --sigma 3.0 --log output/log.csv
```

Sample output:

```
GPU: Tesla T4 (SM 7.5)
Loaded data/lena.ppm (512x512)
Operation: blur
Kernel time: 0.184 ms
Total (H2D+kernel+D2H) time: 1.021 ms
Wrote output/lena_blur.ppm
```

## 6. Getting sample images

See `data/README.md` for public-domain / Creative-Commons image sources
(USC SIPI database, UCI ML repository, CC Search) and the one-line
ImageMagick command to convert PNG/JPEG into the PPM format this project
reads.

## 7. Proof of execution: `run.sh`

```bash
./run.sh
```

This script:

1. Runs `make clean && make` to build fresh.
2. Iterates over every `*.ppm` file in `data/`.
3. Runs **all three** operations (`blur`, `sobel`, `sharpen`) on each
   image — i.e., many executions across multiple images in one batch,
   satisfying the "proof of execution on multiple pieces of data"
   requirement.
4. Writes every output image to `output/` and appends one CSV row per
   run to `output/log.csv` (image, operation, dimensions, GPU kernel
   time, total time), giving a clear, timestamped record that the code
   actually executed on the GPU across many inputs.

Commit the contents of `output/` (images + `log.csv`) back to this
repository as your proof-of-execution artifacts.

## 8. Algorithms / kernels

- **Gaussian blur** — Implemented as a *separable* convolution: instead
  of an O(r²) 2D convolution per pixel, it runs two O(r) 1D passes
  (horizontal, then vertical), reducing work for radius `r` from `r²` to
  `2r` per pixel. Weights are precomputed on the host and pushed once
  per call into GPU `__constant__` memory, which is cached and
  broadcast-read efficiently by all threads in a warp.
- **Sobel edge detection** — Two kernels: a grayscale-luminance
  conversion (`0.299R + 0.587G + 0.114B`), then a 3×3 Sobel Gx/Gy
  convolution computing gradient magnitude `sqrt(Gx² + Gy²)`, clamped to
  `[0, 255]`.
- **Sharpen** — A classic 3×3 unsharp-mask kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]`
  applied independently to each RGB channel.
- All kernels use one CUDA thread per output pixel, a 16×16 thread block,
  and clamp-to-edge boundary handling for out-of-bounds sampling.
- Each kernel launch is timed with CUDA events (`cudaEventRecord` /
  `cudaEventElapsedTime`) for both GPU-only time and full
  host-to-device + compute + device-to-host time.

## 9. Project description / write-up

> _Fill in your own project write-up here — see the "Project Description"
> brief for what to cover. Suggested prompts: What did you build and
> why? What was the biggest technical challenge (e.g. boundary handling,
> choosing block size, separable convolution)? What would you do with
> more time (e.g. shared-memory tiling, multi-GPU, more filters)?_

## 10. License

MIT License — see `LICENSE`.

## 11. Acknowledgments

- Coursera "CUDA at Scale for the Enterprise" course project template.
- Sample image sources: USC SIPI Image Database, UCI ML Repository,
  Creative Commons Search (see `data/README.md`).
