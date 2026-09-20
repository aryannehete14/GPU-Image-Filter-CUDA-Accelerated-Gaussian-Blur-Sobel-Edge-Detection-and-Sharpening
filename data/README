# data/

Place binary PPM (P6) images here before running `./run.sh`.

## Where to get sample images

- USC SIPI Image Database: https://sipi.usc.edu/database/database.php
- UC Irvine ML Repository (e.g. MNIST, CMU Face images): https://archive-beta.ics.uci.edu
- Creative Commons Search: https://search.creativecommons.org

## Converting to PPM

Most downloaded images will be PNG/JPEG/TIFF. Convert them with
[ImageMagick](https://imagemagick.org/):

```bash
sudo apt-get install imagemagick   # if not already installed

# Single file
convert lena.png lena.ppm

# Batch convert everything in this folder
for f in *.png *.jpg; do
  [ -e "$f" ] || continue
  convert "$f" "${f%.*}.ppm"
done
```

Only 8-bit binary PPM (`P6`, maxval 255) is supported by this project's
minimal reader — ImageMagick's default `convert` output satisfies this.
