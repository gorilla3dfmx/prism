#!/usr/bin/env python3
"""Dump the header of a GGUF file: metadata, tensor shapes, type histogram.

Prism only ever mmaps the tensors it knows by name, so when a model refuses to
load the first question is always "what is actually in this file?". This answers
that without loading a single weight -- it reads only the header.

Usage:
    python tools/DumpGguf.py model/some-model.gguf [max_tensors]

`max_tensors` defaults to 60; pass a large number to list them all. The name
histogram at the end collapses digits to `N`, so a 30-layer model shows one row
per distinct tensor role instead of 453 rows.

See docs/z-image-turbo-t2i.md for an example of reading the output.
"""

import collections
import struct
import sys

# ggml_type -> name. Prism supports the subset listed in Prism.Vector.pas
# (TGgmlType); everything else here is informational.
GGML_TYPE = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
    8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
    14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
    19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS',
    24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M',
    30: 'BF16',
}

PRISM_SUPPORTED = {'F32', 'F16', 'Q4_0', 'Q4_1', 'Q8_0', 'Q4_K', 'Q5_K', 'Q6_K'}


class Reader:
    def __init__(self, f):
        self.f = f

    def _u(self, fmt, n):
        return struct.unpack(fmt, self.f.read(n))[0]

    def u32(self):
        return self._u('<I', 4)

    def u64(self):
        return self._u('<Q', 8)

    def string(self):
        return self.f.read(self.u64()).decode('utf-8', errors='replace')


# GGUF metadata value types, in the order defined by the spec.
_SCALARS = {
    0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
    4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<B', 1),
    10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
}


def read_value(r, typ):
    if typ in _SCALARS:
        fmt, size = _SCALARS[typ]
        v = struct.unpack(fmt, r.f.read(size))[0]
        return bool(v) if typ == 7 else v
    if typ == 8:
        return r.string()
    if typ == 9:  # array
        elem = r.u32()
        return [read_value(r, elem) for _ in range(r.u64())]
    raise ValueError('unknown GGUF value type %d' % typ)


def main(path, limit):
    with open(path, 'rb') as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b'GGUF':
            raise SystemExit('not a GGUF file (magic = %r)' % magic)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        print('GGUF v%d  --  %d tensors, %d metadata keys' % (version, n_tensors, n_kv))

        print('\n--- METADATA ---')
        for _ in range(n_kv):
            key = r.string()
            value = read_value(r, r.u32())
            if isinstance(value, list) and len(value) > 8:
                # Vocabularies and charsmaps are huge; show only a sample.
                print('%s = [%d items] %s ...' % (key, len(value), value[:6]))
            else:
                text = str(value)
                print('%s = %s' % (key, text[:300] + '...' if len(text) > 300 else text))

        infos = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            infos.append((name, dims, GGML_TYPE.get(r.u32(), '?'), r.u64()))

    print('\n--- TENSORS ---')
    for name, dims, typ, _ in infos[:limit]:
        print('%-58s %-24s %s' % (name, dims, typ))
    if len(infos) > limit:
        print('... (%d more, pass a larger max_tensors to see them)' % (len(infos) - limit))

    print('\n--- TYPE HISTOGRAM ---')
    types = collections.Counter(t for _, _, t, _ in infos)
    for typ, count in types.most_common():
        mark = '' if typ in PRISM_SUPPORTED else '   <- NOT supported by Prism'
        print('%6d  %s%s' % (count, typ, mark))

    print('\n--- NAME HISTOGRAM (digits collapsed to N) ---')
    roles = collections.Counter(
        '.'.join('N' if p.isdigit() else p for p in name.split('.'))
        for name, _, _, _ in infos)
    for role, count in sorted(roles.items()):
        print('%6d  %s' % (count, role))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    # Vocabularies contain characters cp1252 cannot encode (e.g. U+2581).
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
