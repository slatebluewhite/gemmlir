# cnn3

A small convolutional network, exported early and kept because `docs/pipeline.md`
quotes its timing. `_raw.mlir` is what torch-mlir produced, `_cal.mlir` the same
after `scripts/calibrate.py` annotated it, and `_data.h` the input and the
reference output PyTorch gave for it.

    ../../scripts/compile.sh cnn3_cal.mlir --quantize -o /tmp/cnn3.o
