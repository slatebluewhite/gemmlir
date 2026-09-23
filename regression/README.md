# The regression gate

Every change to this project is accepted or rejected here. The rule is strict
on purpose: **byte for byte**, not "close". A relative L2 has hidden a
convolution wrong in one pixel and a resadd that loses one row in four; a byte
comparison saw both.

## Once

```bash
export/export_all.sh          # writes models/: 14 models, *_cal.mlir + *_data.h
```

Random weights on random noise at 64x64, from a fixed seed, so the inputs are
the same every time. They cannot classify anything -- `demo/` is the half that
does -- and they do not need to: a wrong rewrite moves the output whatever the
weights are, and the small input keeps a fourteen-model sweep near an hour.

## For each change

```bash
gate.sh build   /tmp/before           # on the tree before the change
# ...make the change, rebuild gemmlir-opt and the runtime...
gate.sh build   /tmp/after
gate.sh compare /tmp/after /tmp/before
```

A build directory keeps its own runtime objects, so a change to the runtime is
gated exactly like a change to the compiler.

`compare` is **one board session** that does, for every model:

| | | |
|---|---|---|
| 1 | new object vs gemmini.h's own CPU implementation of the same calls | 40 runs, byte for byte |
| 2 | new object vs the previous build, both on the accelerator | 40 runs, byte for byte |
| 3 | both builds timed, alternated A B A B, minimum of two | |

and prints the verdict and the table. A change is accepted when every model is
**0 of 40 against the CPU reference**. The column against the previous build
says whether the change moved any byte at all; for a rewrite that claims to be
exact it should be 0 too.

## Things that will waste an afternoon

- **One board session at a time.** Two concurrent `tssh.py` sessions once took
  the U280 off the network. `compare` refuses to start if one is running. Stop
  a session by its PID, never `pkill -f tssh.py`.
- **The board's `/tmp` is a tmpfs** that hides the NFS share. `/mnt2` is a bind
  mount of `/` that reaches it, and a reboot of the board loses it. `compare`
  remounts it; by hand it is `sudo mount --bind / /mnt2`.
- **A mismatch on a model whose object did not change** is the board's
  intermittent `resadd_i8` fault (16x64, only MobileNetV2 has the shape), not
  your change. Compare the objects before reverting anything.
- **Two sessions do not compare.** The same object has read 150 ms in one and
  243 ms in another. Only readings taken side by side mean anything, which is
  why both builds are timed in the same session.

## Files

| | |
|---|---|
| `gate.sh` | build and compare |
| `reduce.py` | a gate log to a verdict and a table |
| `export/` | the exporters, one per model family, and `export_all.sh` |
| `harness/` | `dump_main.c` (writes the output bytes), `time_main.c`, the GPT's two (it takes token ids), and `ticks.h`, which reads the timebase from the device tree instead of assuming a clock |
