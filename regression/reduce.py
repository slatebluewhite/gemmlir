"""Reduce a gate log to a verdict and a timing table.

A model passes when its new object is 0 of N against the CPU reference.
"vs previous build" says whether the change moved any output byte at all -- a
nonzero there on a model whose object did not change is the board's
intermittent resadd fault, not the change.

Times are the minimum of the two alternated readings on each side.
"""
import re, sys
log = sys.argv[1] if len(sys.argv) > 1 else '/srv/nfs/debian-riscv64/tmp/gate.log'
lines = open(log).read().splitlines()
bad = []
for l in lines:
    m = re.match(r'^(\S+): (\d+) of (\d+) vs cpu \| (\d+) of \d+ vs previous build', l)
    if m:
        name, cpu, runs, prev = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if cpu or prev:
            bad.append(f'  {name}: {cpu} of {runs} vs cpu, {prev} of {runs} vs previous build')
checked = sum(1 for l in lines if ' vs cpu | ' in l)
print(f'byte for byte: {checked - len(bad)} of {checked} clean')
for b in bad: print(b)
new, prev = {}, {}
for l in lines:
    m = re.match(r'^(\S+)\s+(NEW|PREV)\s', l)
    if not m: continue
    ts = [float(x) for x in re.findall(r'([\d.]+) ms/inference', l)]
    if ts: (new if m.group(2) == 'NEW' else prev)[m.group(1)] = min(ts)
if prev:
    tn = tp = 0.0
    print(f'\n{"model":22s} {"prev":>9s} {"new":>9s} {"delta":>8s}')
    for k in prev:
        if k not in new: continue
        tn += new[k]; tp += prev[k]
        print(f'{k:22s} {prev[k]:9.2f} {new[k]:9.2f} {(new[k]-prev[k])/prev[k]*100:+7.2f}%')
    print(f'{"SET":22s} {tp:9.2f} {tn:9.2f} {(tn-tp)/tp*100:+7.2f}%')
if 'DONE' not in lines: print('\n(log incomplete: no DONE line)')
