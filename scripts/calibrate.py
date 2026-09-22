#!/usr/bin/env python3
"""Record activation ranges from a PyTorch model and annotate its linalg IR.

gemmlir derives a weight's int8 scale from the constant itself, but an
activation's range is only known by running the model. This exports the model
through torch-mlir and writes ``gemmlir.activation_scale`` onto each
``linalg.matmul``, which ``--force-quantized-matmul`` then uses instead of its
fallback.

    from calibrate import calibrate
    mlir = calibrate(model, example_input, calib_inputs)

Matmuls are matched to layers by position -- torch-mlir emits them in execution
order -- and the match is **checked against the shapes**. If they disagree the
call fails rather than annotating the wrong operation, because a scale on the
wrong layer is worse than no scale at all.

Needs torch and torch-mlir; see docs/pipeline.md for getting those installed.
"""

import re

__all__ = ["calibrate", "activation_ranges", "normalize"]

# A matmul, whose K x N is its own, or an NHWC convolution, whose analogue is
# (in-channels, out-channels). A depthwise filter is (KH, KW, C) and has no
# output-channel dimension of its own -- the two are the same C. A global
# average pool is a matmul by then: `--average-pool-to-matmul` turns it into
# `ones(1, H*W) x image(H*W, C)` -- or, for a batch of images, the
# `batch_matmul` of the same thing -- so it lines up as (H*W, C) either way.
_OP = re.compile(
    r"linalg\.(?P<op>batch_matmul|matmul|conv_2d_nhwc_hwcf"
    r"|depthwise_conv_2d_nhwc_hwc)"
    r"(?P<attrs>\s*\{[^}]*\})?\s*ins\([^:]*:\s*"
    r"tensor<(?P<lhs>[0-9x]+)xf32>,\s*tensor<(?P<rhs>[0-9x]+)xf32>"
)


def _contracted(match):
    """The (in, out) pair the calibrated layer should agree with."""
    rhs = [int(d) for d in match.group("rhs").split("x")]
    if match.group("op") == "matmul":
        return rhs[0], rhs[1]
    if match.group("op") == "batch_matmul":
        # (batch, K, N): the batch is not contracted.
        return rhs[1], rhs[2]
    if match.group("op") == "depthwise_conv_2d_nhwc_hwc":
        return rhs[-1], rhs[-1]
    return rhs[-2], rhs[-1]


def _is_global(mod, shape):
    """True when this pooling layer covers the whole image, which is the only
    shape `--average-pool-to-matmul` turns into a contraction."""
    h, w = int(shape[-2]), int(shape[-1])
    size = getattr(mod, "output_size", None)
    if size is not None:
        size = size if isinstance(size, (tuple, list)) else (size, size)
        return tuple(size) == (1, 1)
    k = mod.kernel_size
    k = k if isinstance(k, (tuple, list)) else (k, k)
    return tuple(k) == (h, w) and not mod.padding


_EXPAND = re.compile(
    r"(tensor\.expand_shape\s+%[\w.]+\s*\[\[.*?\]\])\s*"
    r"(:\s*tensor<[^>]*>\s*into\s*tensor<([0-9x]+)x[a-z0-9]+>)")


_CAST = re.compile(r"^\s*(%[\w.]+) = tensor\.cast (%[\w.]+) : "
                   r"(tensor<[^>]*>) to (tensor<[^>]*>)\s*$")
_RESHAPE = re.compile(r"^\s*(%[\w.]+) = tensor\.(expand|collapse)_shape "
                      r"(%[\w.]+) (\[\[.*?\]\]) : "
                      r"(tensor<[^>]*>) (?:into|to) (tensor<[^>]*>)\s*$")


def _dims(ty):
    """["1", "4", "?"] and the element type, from `tensor<1x4x?xf32>`."""
    body = ty[len("tensor<"):-1]
    parts = body.split("x")
    return parts[:-1], parts[-1]


def _static(ty):
    return "?" not in ty


def _groups(reassoc):
    return [[int(i) for i in g.split(",")]
            for g in reassoc.strip("[]").split("], [")]


def _resolve_reshape(kind, src, dst, reassoc):
    """The static result type of a reshape whose source is static.

    torch-mlir writes a grouped convolution through a dynamic shape even when
    the model has none, so the result type arrives as `tensor<?x4x?x?x?xf32>`.
    Every `?` is recoverable: a collapsed dimension is the product of the ones
    it came from, and an expanded group multiplies back to the dimension it
    came from, so a group with a single `?` in it is determined.
    """
    sdims, elem = _dims(src)
    ddims, _ = _dims(dst)
    groups = _groups(reassoc)
    out = list(ddims)
    for g, members in enumerate(groups):
        if kind == "collapse":
            # members index the *source*; the group is one result dimension.
            n = 1
            for i in members:
                n *= int(sdims[i])
            out[g] = str(n)
        else:
            # members index the *result*; together they are one source dim.
            unknown = [i for i in members if out[i] == "?"]
            known = 1
            for i in members:
                if out[i] != "?":
                    known *= int(out[i])
            if not unknown:
                continue
            if len(unknown) > 1:
                return None
            total = int(sdims[g])
            if known == 0 or total % known:
                return None
            out[unknown[0]] = str(total // known)
    if "?" in out:
        return None
    return "tensor<%s>" % "x".join(out + [elem])


def _make_static(mlir):
    """Undo the dynamic shapes torch-mlir introduces around grouped convolutions.

    It casts the operands to `tensor<?x?x?x?xf32>`, reshapes them into the
    5-D form `linalg.conv_2d_ngchw_gfchw` wants, and casts the result back --
    so the shapes are dynamic in the text and static in the model. Nothing
    downstream can work with that, and `tensor.expand_shape` will not even
    parse without an `output_shape`, which needs the very dimensions the `?`
    hides.

    Both are recovered the same way: a cast whose source is static tells us
    what the dynamic type really is, and a reshape of a static source resolves
    its own result.

    **The fix belongs to the value, not to the spelling.** A block with three
    grouped convolutions casts three *different* static types -- `1x16x16x16`,
    `1x8x16x16`, `1x8x16x16` -- to the same `tensor<?x?x?x?xf32>`, so
    substituting that spelling everywhere gives two of them the first one's
    shape and the file stops verifying. Each resolved type is put back only on
    the lines that name the value it belongs to. A wrong substitution still
    fails the verifier rather than passing quietly.
    """
    for _ in range(64):
        lines = mlir.split("\n")
        for n, line in enumerate(lines):
            m = _CAST.match(line)
            if m and _static(m.group(3)) and not _static(m.group(4)):
                result, source, static_ty, dyn_ty = m.groups()
                rest = lines[:n] + lines[n + 1:]
                body = re.sub(re.escape(result) + r"\b", source, "\n".join(rest))
                mlir = "\n".join(
                    l.replace(dyn_ty, static_ty)
                    if re.search(re.escape(source) + r"\b", l) else l
                    for l in body.split("\n"))
                break

            m = _RESHAPE.match(line)
            if m and _static(m.group(5)) and not _static(m.group(6)):
                got = _resolve_reshape(m.group(2), m.group(5), m.group(6),
                                       m.group(4))
                if not got:
                    continue
                result, dyn_ty = m.group(1), m.group(6)
                mlir = "\n".join(
                    l.replace(dyn_ty, got)
                    if re.search(re.escape(result) + r"\b", l) else l
                    for l in lines)
                break
        else:
            if not _propagate_static(mlir):
                return mlir
            mlir = _propagate_static(mlir)
    return mlir


def _rank(ty):
    return len(_dims(ty)[0])


def _propagate_static(mlir):
    """Put a value's known static type back on the lines that use it.

    A cast or a reshape resolves the type where the value is *defined*; the
    operations that read it are still printed with the `?` spelling they were
    given. Only where it is unambiguous: the line has one dynamic type of that
    rank and one used value whose definition says what it is.
    """
    defined = {}
    for line in mlir.split("\n"):
        m = re.match(r"^\s*(%[\w.]+) = .*?(tensor<[^>]*>)\s*$", line)
        if m and _static(m.group(2)):
            defined[m.group(1)] = m.group(2)
    out, changed = [], False
    for line in mlir.split("\n"):
        dyn = {t for t in re.findall(r"tensor<[^>]*>", line) if not _static(t)}
        used = {n for n in re.findall(r"%[\w.]+", line) if n in defined}
        m = re.match(r"^\s*(%[\w.]+) = ", line)
        if m:
            used.discard(m.group(1))
        for t in dyn:
            same = [n for n in used if _rank(defined[n]) == _rank(t)]
            if len(same) == 1:
                line = line.replace(t, defined[same[0]])
                changed = True
        out.append(line)
    return "\n".join(out) if changed else None


def normalize(mlir):
    """Fix the two places torch-mlir's output cannot be used as it stands.

    `tensor.expand_shape` grew a mandatory `output_shape` list and torch-mlir
    still prints the form without it, so its output does not parse at all.
    Every result shape it emits is static once `_make_static` has run, so the
    list is just the result's dimensions.

    And a grouped convolution arrives with its shapes erased to `?` even
    though the model is static; see `_make_static`.
    """
    mlir = _make_static(mlir)

    def fill(m):
        dims = ", ".join(m.group(3).split("x"))
        return "%s output_shape [%s] %s" % (m.group(1), dims, m.group(2))

    return _EXPAND.sub(fill, mlir)


def activation_ranges(model, calib_inputs, layer_types=None):
    """max|input| for each matmul-bearing layer, in execution order."""
    import torch
    import torch.nn as nn

    if layer_types is None:
        layer_types = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d,
                       nn.AdaptiveAvgPool2d, nn.AvgPool2d)

    seen = []   # [(module, max_abs, in_features, out_features, rhs_max, out_max)]
    index = {}

    def hook(mod, args):
        x = args[0]
        m = float(x.detach().abs().max())
        # A convolution's operand pair is (in-channels, out-channels); a linear
        # layer's is its own (in, out). A pooling layer has neither: once
        # `--average-pool-to-matmul` has had it, the contraction is over the
        # pixels and the channels come out untouched.
        if isinstance(mod, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            # `--average-pool-to-contraction` makes the whole image a matmul
            # over the pixels, and any other window a depthwise convolution --
            # whose operand pair, having no output channel of its own, is
            # (C, C).
            channels = int(x.shape[-3])
            if _is_global(mod, x.shape):
                fan_in = int(x.shape[-2]) * int(x.shape[-1])
                fan_out = channels
            else:
                fan_in = fan_out = channels
            i = index.get(id(mod))
            if i is None:
                index[id(mod)] = len(seen)
                # Which operand the measured range belongs to depends on what
                # the pool becomes. A **global** one is `ones(1, P) x image(P,
                # C)`: the activation is on the *right*, and the left is a
                # constant whose range `--force-quantized-matmul` reads for
                # itself. A **windowed** one is a depthwise convolution, where
                # the image is the left operand and the ones are the filter.
                # Recording it only on the left is why `gapb`'s global average
                # pool quantized at the pass's fallback scale.
                rhs = m if _is_global(mod, x.shape) else None
                seen.append([mod, m, fan_in, fan_out, rhs, None])
            else:
                seen[i][1] = max(seen[i][1], m)
                if seen[i][4] is not None:
                    seen[i][4] = seen[i][1]
            return
        fan_in = getattr(mod, "in_features", None)
        if fan_in is None:
            fan_in = mod.in_channels
        fan_out = getattr(mod, "out_features", None)
        if fan_out is None:
            fan_out = mod.out_channels
        # A grouped convolution is G convolutions in the IR --
        # `--split-grouped-conv` writes it that way because Gemmini has no
        # groups -- so it needs G entries here, each over its own share of the
        # channels.
        parts = getattr(mod, "groups", 1)
        ranges = [m]
        if parts > 1 and fan_in % parts == 0 and fan_out % parts == 0:
            # Depthwise keeps its own operation and its own Gemmini call, so it
            # stays one entry.
            if parts != fan_in or fan_out != fan_in:
                # All G entries get the *same* range, and it has to be the whole
                # tensor's. The groups slice one quantized activation --
                # `--split-grouped-conv` takes `tensor.extract_slice` of it --
                # so there is one scale to have, and the layer that produces it
                # requantizes to that one scale. Giving each group its own range
                # asks for something the IR cannot express: three of the four
                # then read a tensor quantized at the fourth's scale. Measured
                # on a ResNeXt block, 0.1172 relative L2 against 0.0055; the
                # weights are a different matter and are per group, because each
                # group really does have its own slice of the filter.
                fan_in //= parts
                fan_out //= parts
                ranges = [m] * parts
            else:
                parts = 1
        else:
            parts = 1
        i = index.get(id(mod))
        if i is None:
            index[id(mod)] = len(seen)
            for k in range(parts):
                # No fifth entry: a layer's right operand is its weight, and
                # the weight's own range is in the constant.
                seen.append([mod, ranges[k], fan_in, fan_out, None, None])
        else:
            for k in range(parts):
                seen[i + k][1] = max(seen[i + k][1], ranges[k])

    # A transformer's two busiest contractions are not layers: `Q @ K.T` and
    # `probs @ V` are the `@` operator on tensors, and a hook on modules never
    # sees them. They reach the IR as `linalg.batch_matmul` all the same, so
    # without these the count would not line up and `annotate` would refuse.
    #
    # Only at the top level: `nn.Linear` reaches `matmul` on its way down, and
    # counting that too would record the same contraction twice. The pre- and
    # post-hooks bracket a module so the mode knows when it is inside one.
    depth = [0]

    def enter(mod, args):
        hook(mod, args)
        depth[0] += 1

    # `--fold-batch-norm` folds a batch norm into the weights above it, so the
    # value the compiler can requantize is the *batch norm's* output, not the
    # convolution's. The two are not close: the norm rescales per channel.
    # Tracked by tensor identity rather than by position, which is exact.
    produced = {}

    def record_output(i, tensor):
        m = float(tensor.detach().abs().max())
        mod = seen[i][0]
        for k in range(i, len(seen)):
            if seen[k][0] is not mod:
                break
            seen[k][5] = m if seen[k][5] is None else max(seen[k][5], m)
        produced[id(tensor)] = i

    def leave(mod, args, output):
        # Only the modules that also have an `enter`. A batch norm registers
        # this hook alone -- it is here for the output range below, not for the
        # bracket -- and decrementing for it drove `depth` steadily negative:
        # one per batch norm, 121 of them in DenseNet. Every functional branch
        # in the mode below is guarded on `depth[0] == 0`, so after the first
        # batch norm none of them could fire again, and a model whose global
        # average pool is written as a function came out one contraction short.
        # It fails loudly in `annotate`, which is the only reason this was not
        # quietly wrong: MobileNetV2 is that shape and still has the `_cal.mlir`
        # it was given before the batch-norm hook existed.
        if isinstance(mod, layer_types):
            depth[0] -= 1
        if not isinstance(output, torch.Tensor):
            return
        # The layer's *own* output, which is a different thing from the next
        # layer's input: an activation the accelerator cannot end in sits
        # between them, and a requantization has to go where the accelerator
        # needs one. See --quantize-unfoldable-tails.
        i = index.get(id(mod))
        if i is not None:
            record_output(i, output)
            return
        # A batch norm reading what a recorded layer wrote *is* that layer once
        # the fold has happened, so its output is the range to quantize at.
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)) and args:
            j = produced.get(id(args[0]))
            if j is not None:
                record_output(j, output)

    # A contraction the mode sees has no module to key on, so it is matched
    # across calibration passes **by the order it occurs in** -- the same order
    # in every forward pass of a fixed graph.
    #
    # Appending on every pass instead made the layer count grow with the number
    # of calibration inputs: torchvision's MobileNetV2 spells its global pool
    # `nn.functional.adaptive_avg_pool2d`, so it read 54 layers for one image
    # and 57 for four, and `annotate` refused. Nothing caught it because every
    # export until now calibrated on a single input.
    first_pass = [True]
    functional = []
    fcursor = [0]

    def record_functional(entry):
        if first_pass[0]:
            functional.append(len(seen))
            seen.append(entry)
            return
        if fcursor[0] >= len(functional):
            raise RuntimeError(
                "a later calibration input records more module-less "
                "contractions than the first; the graph is not fixed")
        row = seen[functional[fcursor[0]]]
        fcursor[0] += 1
        if row[2] != entry[2] or row[3] != entry[3]:
            raise RuntimeError(
                "a module-less contraction changed shape between calibration "
                "inputs: %r against %r" % (row[2:4], entry[2:4]))
        row[1] = max(row[1], entry[1])
        for k in (4, 5):
            if entry[k] is not None:
                row[k] = entry[k] if row[k] is None else max(row[k], entry[k])

    class RecordMatmuls(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = getattr(func, "__name__", "")
            if depth[0] == 0 and name in ("matmul", "__matmul__") and len(args) >= 2:
                a, b = args[0], args[1]
                if (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                        and a.dim() >= 2 and b.dim() >= 2):
                    # (..., M, K) x (..., K, N): the contraction is K, the
                    # operand pair (K, N), and the range is the left one's --
                    # the same convention a Linear layer's hook uses.
                    # Both ranges. The right operand of an `@` is an
                    # activation too, and `--force-quantized-matmul` has no
                    # constant to read it off -- see `rhsActivationScaleOf`.
                    record_functional([None, float(a.detach().abs().max()),
                                       int(b.shape[-2]), int(b.shape[-1]),
                                       float(b.detach().abs().max()), None])
            # `nn.functional.linear` with no module above it. A
            # `nn.MultiheadAttention` keeps its three projections packed in one
            # `in_proj_weight` **Parameter** and applies it as a function, so no
            # forward hook is ever called for it -- twelve of a ViT's seventy-four
            # contractions were missing for that reason alone. The depth guard is
            # what keeps `nn.Linear`'s own descent into `linear` from counting
            # twice.
            if depth[0] == 0 and name == "linear" and len(args) >= 2:
                x, w = args[0], args[1]
                if (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor)
                        and x.dim() >= 2 and w.dim() == 2):
                    # The layer's own output range as well, which is what
                    # `--quantize-unfoldable-tails` quantizes at. A module gets
                    # it from the `leave` hook; this one has no module, so it
                    # comes from the result -- computed once, here, and handed
                    # back rather than letting the call run twice.
                    out = func(*args, **kwargs)
                    # No fifth entry, as for a `nn.Linear`: the right operand is
                    # a weight and its range is in the constant.
                    record_functional([None, float(x.detach().abs().max()),
                                       int(w.shape[1]), int(w.shape[0]), None,
                                       float(out.detach().abs().max())])
                    return out

            # `nn.functional.adaptive_avg_pool2d(x, (1, 1))` is the third
            # spelling of the same pool and the one torchvision's MobileNetV2
            # uses -- a *function*, so neither the module hook nor the `mean`
            # branch below sees it, and the model came out one contraction short.
            # Of the six torchvision models that export here, three spell the
            # global pool as a module, two as `x.mean([2, 3])` and one this way.
            if depth[0] == 0 and name == "adaptive_avg_pool2d" and args:
                out = kwargs.get("output_size")
                if out is None and len(args) >= 2:
                    out = args[1]
                out = tuple(out) if isinstance(out, (tuple, list)) else (out, out)
                x = args[0]
                if (isinstance(x, torch.Tensor) and x.dim() == 4
                        and out == (1, 1)):
                    m = float(x.detach().abs().max())
                    record_functional([None, m,
                                       int(x.shape[2]) * int(x.shape[3]),
                                       int(x.shape[1]), m, None])

            # `x.mean(dim=(2, 3))` is a global average pool written without a
            # module, so no hook sees it -- and
            # `--raise-spatial-sum-to-pool` turns it into a contraction all the
            # same, which `annotate` would then find one too many of. Recorded
            # with the same fields a global `AdaptiveAvgPool2d` gets: the
            # contraction is `ones(1, H*W) x image(H*W, C)`, so the operand pair
            # is (H*W, C) and the measured range belongs to the **right**
            # operand.
            if depth[0] == 0 and name == "mean" and args:
                x = args[0]
                dim = kwargs.get("dim")
                if dim is None and len(args) >= 2:
                    dim = args[1]
                if (isinstance(x, torch.Tensor) and x.dim() == 4
                        and isinstance(dim, (tuple, list)) and len(dim) == 2):
                    axes = sorted(d % 4 for d in dim)
                    if axes == [2, 3]:
                        m = float(x.detach().abs().max())
                        record_functional([None, m,
                                           int(x.shape[2]) * int(x.shape[3]),
                                           int(x.shape[1]), m, None])
            return func(*args, **kwargs)

    handles = []
    for m in model.modules():
        if isinstance(m, layer_types):
            handles.append(m.register_forward_pre_hook(enter))
            handles.append(m.register_forward_hook(leave))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            handles.append(m.register_forward_hook(leave))
    try:
        with torch.no_grad(), RecordMatmuls():
            for n, inp in enumerate(calib_inputs):
                first_pass[0] = n == 0
                fcursor[0] = 0
                model(*(inp if isinstance(inp, (tuple, list)) else (inp,)))
    finally:
        for h in handles:
            h.remove()

    if not seen:
        raise RuntimeError("no calibration hooks fired; is the model made of %s?"
                           % (layer_types,))
    return [(m, k, n, rhs, out) for _, m, k, n, rhs, out in seen]


def annotate(mlir, ranges):
    """Write gemmlir.activation_scale onto each contraction, in order.

    A contraction whose right operand is an activation rather than a weight --
    a transformer's `Q @ K.T` and `probs @ V` -- gets a second scale, because
    there is no constant for the pass to read that one off.
    """
    found = list(_OP.finditer(mlir))
    if len(found) != len(ranges):
        raise RuntimeError(
            "the IR has %d f32 contractions but %d layers were calibrated; "
            "the model does not lower one-to-one, so positional matching "
            "would guess" % (len(found), len(ranges)))

    out, last = [], 0
    for i, (match, entry) in enumerate(zip(found, ranges)):
        absmax, k, n = entry[0], entry[1], entry[2]
        rhs_absmax = entry[3] if len(entry) > 3 else None
        got_k, got_n = _contracted(match)
        if (got_k, got_n) != (k, n):
            raise RuntimeError(
                "operation %d in the IR contracts %dx%d but the layer it lines "
                "up with is %dx%d; refusing to annotate the wrong operation"
                % (i, got_k, got_n, k, n))
        out_absmax = entry[4] if len(entry) > 4 else None
        # A measured range of **exactly zero** is a real measurement, not a
        # failure: an LSTM's hidden state at the first timestep is
        # `torch.zeros`, and half of a recurrent model's contractions read it.
        # The tensor is all zeros, so any positive scale represents it exactly;
        # writing the zero through would put a divide by zero in the
        # quantization loop and an `fptosi` of an infinity after it.
        absmax = absmax or 1.0
        attr = "gemmlir.activation_scale = %.9e : f64" % (absmax / 127.0)
        if rhs_absmax:
            attr += (", gemmlir.rhs_activation_scale = %.9e : f64"
                     % (rhs_absmax / 127.0))
        if out_absmax:
            # What this layer itself writes. Only used where the tail cannot
            # fold and a requantization has to be put in by hand.
            attr += (", gemmlir.output_scale = %.9e : f64"
                     % (out_absmax / 127.0))
        existing = match.group("attrs")
        if existing is None:
            # No dictionary yet: add one right after the operation's name.
            insert = match.start() + len("linalg.") + len(match.group("op"))
            out.append(mlir[last:insert])
            out.append(" {%s}" % attr)
            last = insert
        else:
            # A convolution already carries strides and dilations; merge in.
            insert = match.start("attrs") + existing.index("{") + 1
            out.append(mlir[last:insert])
            out.append("%s, " % attr)
            last = insert
    out.append(mlir[last:])
    return "".join(out)


def calibrate(model, example_input, calib_inputs, layer_types=None):
    """Export `model` to linalg-on-tensors and annotate it with measured scales."""
    import torch_mlir

    ranges = activation_ranges(model, calib_inputs, layer_types)
    mlir = str(torch_mlir.compile(model, example_input,
                                  output_type="linalg-on-tensors"))
    return annotate(mlir, ranges)


def _self_check():
    """What `activation_ranges` has to get right, checked without torch-mlir.

    Every entry is five fields; the fifth is the range of a *right* operand
    that is an activation rather than a weight. A global average pool becomes
    `ones(1, P) x image(P, C)`, so its image is on the right and the fifth
    field carries it -- leaving it off is what made `gapb`'s pool quantize at
    the pass's fallback scale. A windowed pool becomes a depthwise convolution,
    where the image is the left operand and there is no second activation.
    """
    import torch
    import torch.nn as nn

    class Pools(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.window = nn.AvgPool2d(2, 2)
            self.whole = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            return self.whole(self.window(x))

    torch.manual_seed(0)
    ranges = activation_ranges(Pools().eval(), [torch.randn(1, 3, 8, 8)])
    assert len(ranges) == 3, ranges
    for entry in ranges:
        assert len(entry) == 5, \
            "every entry is (max, k, n, rhs_max, out_max): %r" % (entry,)
    # A convolution is a module, so its own output is measured too; the pools
    # are handled by the same hook and get theirs as well.
    assert ranges[0][4] and ranges[0][4] > 0, ranges[0]
    conv, window, whole = ranges
    assert conv[1:3] == (3, 8) and conv[3] is None, conv
    # A window of 2x2 over 8 channels: the depthwise convolution's operand
    # pair has no output channel of its own.
    assert window[1:3] == (8, 8) and window[3] is None, window
    # 4x4 pixels left after the window, 8 channels, and the image on the right.
    assert whole[1:3] == (16, 8), whole
    assert whole[3] == whole[0], "the global pool's image is its right operand"
    print("calibrate.py: %d entries, shapes and operand sides as expected"
          % len(ranges))

    # More calibration inputs must widen the ranges, never lengthen the list.
    # A contraction with a module dedupes by `id(mod)`; one the
    # `TorchFunctionMode` sees has no module and is matched by position. When
    # it was not, torchvision's MobileNetV2 -- whose global pool is
    # `nn.functional.adaptive_avg_pool2d`, a function -- read 54 layers for one
    # calibration image and 57 for four, and `annotate` refused. Every export
    # before the demo calibrated on a single input, so nothing exercised it.
    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            return self.fc(x.flatten(1)) @ torch.eye(4)

    torch.manual_seed(0)
    model = Functional().eval()
    small = [torch.randn(1, 3, 8, 8) * 0.1]
    many = small + [torch.randn(1, 3, 8, 8) * 3.0, torch.randn(1, 3, 8, 8)]
    one = activation_ranges(model, small)
    lots = activation_ranges(model, many)
    # Four: the convolution and the linear are modules, the pool and the `@`
    # are not -- so both kinds are exercised.
    assert len(one) == len(lots) == 4, (len(one), len(lots))
    assert all(b[0] >= a[0] - 1e-6 for a, b in zip(one, lots)), (one, lots)
    assert any(b[0] > a[0] + 1e-6 for a, b in zip(one, lots)), \
        "a wider input should widen at least one range"
    print("calibrate.py: %d entries for one input and for three, ranges widened"
          % len(lots))

    # An attention block, which is where the two things a module hook cannot see
    # both appear: a projection applied as `F.linear` straight off a Parameter
    # (`nn.MultiheadAttention` keeps its packed QKV that way) and the two
    # matmuls between activations. A ViT is 74 contractions and 48 of them are
    # one of these two; without both branches the count does not line up and
    # `annotate` refuses rather than guessing.
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Parameter(torch.randn(3 * 12, 12) * 0.1)
            self.out = nn.Linear(12, 12)

        def forward(self, x):
            q, k, v = torch.nn.functional.linear(x, self.qkv).chunk(3, dim=-1)
            a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
            return self.out(torch.matmul(a, v))

    torch.manual_seed(0)
    att = activation_ranges(Attention().eval(), [torch.randn(1, 5, 12)])
    assert len(att) == 4, att
    proj, qk, av, out = att
    # The packed projection: (in, out) off the weight, and the weight is a
    # constant so there is no second activation.
    assert proj[1:3] == (12, 36) and proj[3] is None, proj
    # `q @ k.T`: the right operand is `k` transposed, so (12, 5), and it is an
    # activation -- `--force-quantized-matmul` has no constant to read it off.
    assert qk[1:3] == (12, 5) and qk[3] is not None, qk
    # `probs @ v`: (5, 12), also an activation on the right.
    assert av[1:3] == (5, 12) and av[3] is not None, av
    # The output projection is an ordinary module, weight on the right.
    assert out[1:3] == (12, 12) and out[3] is None, out
    # A functional linear carries its own output range too -- there is no module
    # for the `leave` hook to read it off, and without it the layer's tail has
    # nothing to quantize at.
    assert proj[4] and proj[4] > 0, proj
    print("calibrate.py: attention gives %d entries, both sides recorded"
          % len(att))


if __name__ == "__main__":
    import sys
    import torch
    import torch.nn as nn

    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 48)
            self.fc2 = nn.Linear(48, 32)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(0)
    model = MLP().eval()
    example = torch.randn(32, 64)
    text = calibrate(model, example, [example])
    sys.stdout.write(text)
