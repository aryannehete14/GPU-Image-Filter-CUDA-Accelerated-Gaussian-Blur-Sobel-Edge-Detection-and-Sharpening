// ppm_io.h
// Minimal, dependency-free reader/writer for binary PPM (P6) images.
// PPM is used instead of PNG/JPEG so the project has zero external
// image-library dependencies. Any PNG/JPEG can be converted to PPM with
// ImageMagick:   convert input.png -compress none output.ppm
//                convert input.jpg output.ppm
#ifndef PPM_IO_H_
#define PPM_IO_H_

#include <string>
#include <vector>

struct Image {
  int width = 0;
  int height = 0;
  int channels = 3;  // Always 3 (RGB) for this project.
  std::vector<unsigned char> data;  // Row-major, interleaved RGB.

  size_t NumPixels() const { return static_cast<size_t>(width) * height; }
  size_t NumBytes() const { return NumPixels() * channels; }
};

// Reads a binary PPM (P6) file. Returns true on success.
bool ReadPPM(const std::string& path, Image* image);

// Writes a binary PPM (P6) file. Returns true on success.
bool WritePPM(const std::string& path, const Image& image);

#endif  // PPM_IO_H_
