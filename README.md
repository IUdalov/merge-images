# merge-images
Unix like command line utility. Applies the outline of one image to another.

### Target platform
Linux

### Build dependencies
1. OpenCV
2. pkg-config

### Build
1. `cd src && make`

### Usage
`./merge <source image> <image traits> [ -o <output>] [-s transparency] [-h]`
1. `-h` - print help.
2. `-s` - set transparency of traits, default is 30%.
3. `-o` - set output file name, default is `result.jpg`
