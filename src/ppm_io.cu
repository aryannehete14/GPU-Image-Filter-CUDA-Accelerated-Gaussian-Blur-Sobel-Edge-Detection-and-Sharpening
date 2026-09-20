// ppm_io.cu
// Host-only implementation, compiled by nvcc alongside the kernel code.
#include "ppm_io.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

// Reads the next whitespace-delimited token from a PPM header, skipping
// any '#' comment lines, per the NetPBM spec.
std::string NextToken(std::ifstream* file) {
  std::string token;
  char c;
  while (file->get(c)) {
    if (c == '#') {
      std::string discard;
      std::getline(*file, discard);
      continue;
    }
    if (isspace(static_cast<unsigned char>(c))) {
      if (!token.empty()) break;
      continue;
    }
    token += c;
  }
  return token;
}

}  // namespace

bool ReadPPM(const std::string& path, Image* image) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "ERROR: could not open input file: " << path << std::endl;
    return false;
  }

  std::string magic = NextToken(&file);
  if (magic != "P6") {
    std::cerr << "ERROR: unsupported PPM format '" << magic
               << "' (only binary P6 is supported). Convert with e.g.\n"
               << "  convert input.png ppm_output.ppm" << std::endl;
    return false;
  }

  int width = std::stoi(NextToken(&file));
  int height = std::stoi(NextToken(&file));
  int maxval = std::stoi(NextToken(&file));
  if (maxval != 255) {
    std::cerr << "ERROR: only 8-bit PPM (maxval=255) is supported."
               << std::endl;
    return false;
  }

  // Consume the single whitespace character after maxval before raw data.
  file.get();

  image->width = width;
  image->height = height;
  image->channels = 3;
  image->data.resize(static_cast<size_t>(width) * height * 3);
  file.read(reinterpret_cast<char*>(image->data.data()), image->data.size());
  if (!file) {
    std::cerr << "ERROR: unexpected end of file while reading pixel data."
               << std::endl;
    return false;
  }
  return true;
}

bool WritePPM(const std::string& path, const Image& image) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "ERROR: could not open output file: " << path << std::endl;
    return false;
  }
  file << "P6\n" << image.width << " " << image.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(image.data.data()),
             image.data.size());
  return static_cast<bool>(file);
}
