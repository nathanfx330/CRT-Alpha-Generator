#!/usr/bin/env python3
# crt_mask.py
#
# CRT phosphor mask / scanline plate generator.
# Bakes a linear coverage field for compositing in Fusion.
# Pure stdlib. No numpy, no PIL, no OpenEXR.
#
# The mask lattice (horizontal, tube property) and the scanline lattice
# (vertical, signal property) are independent. Set them separately.

import argparse
import math
import struct
import sys
import zlib


# ----------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------

def profile(t, duty, soft):
    """Stripe cross-section. t in [0,1) within one period, centered at 0.5."""
    half = duty * 0.5
    d = abs(t - 0.5)
    if soft <= 0.0:
        return 1.0 if d <= half else 0.0
    a = half - soft * 0.5
    b = half + soft * 0.5
    if b <= a:
        return 1.0 if d <= half else 0.0
    if d <= a:
        return 1.0
    if d >= b:
        return 0.0
    u = (d - a) / (b - a)
    return 1.0 - (u * u * (3.0 - 2.0 * u))


def radial(d, r, soft):
    """Dot cross-section. d = distance from site center."""
    if soft <= 0.0:
        return 1.0 if d <= r else 0.0
    a = r - soft * 0.5
    b = r + soft * 0.5
    if d <= a:
        return 1.0
    if d >= b:
        return 0.0
    u = (d - a) / (b - a)
    return 1.0 - (u * u * (3.0 - 2.0 * u))


# ----------------------------------------------------------------------
# axis builders (box-filtered by supersampling, so non-integer pitch is safe)
# ----------------------------------------------------------------------

def build_h(width, pitch, duty, soft, samples, subcell=None, parity=None):
    """Horizontal stripe array.

    subcell: None for all three (monochrome mask), or 0/1/2 to isolate one
             phosphor stripe (used for --rgb).
    parity:  None for every triad column, or 0/1 to isolate even/odd columns
             (used by slot mode to keep the pattern separable).
    """
    out = [0.0] * width
    inv = 1.0 / samples
    sub = pitch / 3.0
    for i in range(width):
        acc = 0.0
        for s in range(samples):
            x = i + (s + 0.5) * inv
            c = math.floor(x / pitch)
            if parity is not None and (int(c) & 1) != parity:
                continue
            local = x - c * pitch
            k = int(local / sub)
            if k > 2:
                k = 2
            if subcell is not None and k != subcell:
                continue
            t = (local - k * sub) / sub
            acc += profile(t, duty, soft)
        out[i] = acc * inv
    return out


def build_v_rect(height, pitch, duty, soft, samples, phase=0.0):
    """Vertical rect train. Slot mask vertical structure."""
    out = [0.0] * height
    inv = 1.0 / samples
    for i in range(height):
        acc = 0.0
        for s in range(samples):
            y = i + (s + 0.5) * inv
            t = ((y - phase) / pitch) % 1.0
            acc += profile(t, duty, soft)
        out[i] = acc * inv
    return out


def build_scan(height, pitch, beam, floor_, samples):
    """Electron beam vertical profile. Raised cosine, not a hard line."""
    out = [0.0] * height
    inv = 1.0 / samples
    w = max(beam * 0.5, 1e-9)
    for i in range(height):
        acc = 0.0
        for s in range(samples):
            y = i + (s + 0.5) * inv
            t = (y / pitch) % 1.0
            d = (t - 0.5) / w
            if abs(d) >= 1.0:
                v = 0.0
            else:
                v = 0.5 * (1.0 + math.cos(math.pi * d))
            acc += floor_ + (1.0 - floor_) * v
        out[i] = acc * inv
    return out


def apply_wires(vy, height, count, px, dark, samples):
    """Trinitron damper wires. Thin horizontal shadows."""
    if count <= 0:
        return vy
    inv = 1.0 / samples
    ys = [height * (k + 1.0) / (count + 1.0) for k in range(count)]
    for i in range(height):
        acc = 0.0
        for s in range(samples):
            y = i + (s + 0.5) * inv
            v = 1.0
            for wy in ys:
                d = abs(y - wy)
                if d < px * 0.5:
                    u = d / (px * 0.5)
                    v *= dark + (1.0 - dark) * (u * u * (3.0 - 2.0 * u))
            acc += v
        vy[i] *= acc * inv
    return vy


def build_dot_tile(pitch, fill, soft, ss):
    """Hex dot lattice. Not separable, so bake a tile at snapped integer pitch."""
    p = max(3, int(round(pitch)))
    rp = max(2, int(round(p * math.sqrt(3.0) * 0.5)))
    th = 2 * rp
    r = p * fill * 0.5
    sw = soft * p
    sites = []
    for m in range(-1, 3):
        for n in range(-1, 3):
            sites.append((n * p + (m & 1) * p * 0.5, m * rp))
    inv = 1.0 / ss
    tile = []
    for y in range(th):
        row = [0.0] * p
        for x in range(p):
            acc = 0.0
            for sy in range(ss):
                yy = y + (sy + 0.5) * inv
                for sx in range(ss):
                    xx = x + (sx + 0.5) * inv
                    best = 1e18
                    for (cx, cy) in sites:
                        dx = xx - cx
                        dy = yy - cy
                        d2 = dx * dx + dy * dy
                        if d2 < best:
                            best = d2
                    acc += radial(math.sqrt(best), r, sw)
            row[x] = acc * inv * inv
        tile.append(row)
    return tile, p, th


# ----------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------

def _png_chunk(tag, data):
    return (struct.pack('>I', len(data)) + tag + data
            + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))


def write_png16(path, w, h, rows, channels):
    """16 bit PNG. channels 1 = grayscale, 3 = RGB."""
    ctype = 0 if channels == 1 else 2
    raw = bytearray()
    packer = struct.Struct('>%dH' % (w * channels))
    for row in rows:
        raw.append(0)
        vals = []
        for v in row:
            q = int(v * 65535.0 + 0.5)
            if q < 0:
                q = 0
            elif q > 65535:
                q = 65535
            vals.append(q)
        raw += packer.pack(*vals)
    out = bytearray(b'\x89PNG\r\n\x1a\n')
    out += _png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, ctype, 0, 0, 0))
    out += _png_chunk(b'IDAT', zlib.compress(bytes(raw), 9))
    out += _png_chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(out)


def _exr_attr(name, atype, payload):
    return (name.encode('ascii') + b'\x00' + atype.encode('ascii') + b'\x00'
            + struct.pack('<i', len(payload)) + payload)


def write_exr32(path, w, h, rows, channels):
    """Uncompressed scanline EXR, 32 bit float, always RGB.

    Monochrome input is replicated to R, G and B so the Loader is unambiguous.
    """
    chlist = b''
    for cname in ('B', 'G', 'R'):  # alphabetical, required
        chlist += cname.encode('ascii') + b'\x00'
        chlist += struct.pack('<i', 2)       # FLOAT
        chlist += struct.pack('<B', 0)       # pLinear
        chlist += b'\x00\x00\x00'            # reserved
        chlist += struct.pack('<ii', 1, 1)   # x/y sampling
    chlist += b'\x00'

    header = bytearray()
    header += struct.pack('<I', 20000630)
    header += struct.pack('<I', 2)
    header += _exr_attr('channels', 'chlist', chlist)
    header += _exr_attr('compression', 'compression', struct.pack('<B', 0))
    header += _exr_attr('dataWindow', 'box2i', struct.pack('<iiii', 0, 0, w - 1, h - 1))
    header += _exr_attr('displayWindow', 'box2i', struct.pack('<iiii', 0, 0, w - 1, h - 1))
    header += _exr_attr('lineOrder', 'lineOrder', struct.pack('<B', 0))
    header += _exr_attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
    header += _exr_attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
    header += _exr_attr('screenWindowWidth', 'float', struct.pack('<f', 1.0))
    header += b'\x00'

    block = 3 * w * 4
    base = len(header) + 8 * h
    offsets = b''.join(struct.pack('<Q', base + i * (8 + block)) for i in range(h))

    packer = struct.Struct('<%df' % w)
    body = bytearray()
    for y, row in enumerate(rows):
        if channels == 1:
            b_ = g_ = r_ = row
        else:
            r_ = row[0::3]
            g_ = row[1::3]
            b_ = row[2::3]
        body += struct.pack('<ii', y, block)
        body += packer.pack(*b_)
        body += packer.pack(*g_)
        body += packer.pack(*r_)

    with open(path, 'wb') as f:
        f.write(bytes(header))
        f.write(offsets)
        f.write(bytes(body))


# ----------------------------------------------------------------------

def parse_res(s):
    a, b = s.lower().split('x')
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(
        description='Bake a CRT phosphor mask plate for Fusion.')
    ap.add_argument('--out', default='1920x1080',
                    help='output plate resolution, WxH')
    ap.add_argument('--cells', default='320x240',
                    help='emulated CRT: triads across x scanlines down')
    ap.add_argument('--mask', default='grille', choices=('grille', 'slot', 'dot'))
    ap.add_argument('--fill', type=float, default=0.70,
                    help='stripe width as fraction of subcell')
    ap.add_argument('--soft', type=float, default=0.15,
                    help='edge softness as fraction of subcell')
    ap.add_argument('--slot-duty', type=float, default=0.72,
                    help='slot mode: vertical fill of each slot')
    ap.add_argument('--slot-aspect', type=float, default=1.0,
                    help='slot mode: slot vertical pitch / triad pitch')
    ap.add_argument('--dot-fill', type=float, default=0.82,
                    help='dot mode: dot diameter / triad pitch')
    ap.add_argument('--scan', type=float, default=0.75,
                    help='beam width as fraction of line pitch, 0 disables')
    ap.add_argument('--scan-floor', type=float, default=0.0,
                    help='floor between scanlines, 0 is fully black')
    ap.add_argument('--wires', type=int, default=0,
                    help='grille mode: damper wire count (0, 1 or 2)')
    ap.add_argument('--wire-px', type=float, default=1.5)
    ap.add_argument('--wire-dark', type=float, default=0.45)
    ap.add_argument('--rgb', action='store_true',
                    help='emit R/G/B stripes in separate channels')
    ap.add_argument('--normalize', action='store_true',
                    help='scale mean to 1.0 so the plate does not darken')
    ap.add_argument('--samples', type=int, default=32,
                    help='1D supersample rate')
    ap.add_argument('--png', default=None)
    ap.add_argument('--exr', default=None)
    args = ap.parse_args()

    W, H = parse_res(args.out)
    NX, NY = parse_res(args.cells)
    S = max(1, args.samples)

    pitch = W / float(NX)
    line = H / float(NY)

    print('plate      %dx%d' % (W, H))
    print('mask       %s' % args.mask)
    print('triads     %d across, pitch %.4f px, subcell %.4f px'
          % (NX, pitch, pitch / 3.0))
    print('scanlines  %d, pitch %.4f px' % (NY, line))

    sub = pitch / 3.0
    if sub < 2.0:
        print('WARNING    subcell < 2 px, mask is below Nyquist, it will vanish',
              file=sys.stderr)
    elif sub < 3.0:
        print('WARNING    subcell < 3 px, mask contrast will be very weak.',
              file=sys.stderr)
        print('           at exactly 2.0 px a centered stripe box-filters to a '
              'flat field.', file=sys.stderr)
        print('           want subcell >= 3 px, so triad pitch >= 9 px, so '
              'width >= %d for %d triads.' % (int(math.ceil(NX * 9)), NX),
              file=sys.stderr)
    if args.scan > 0.0 and line < 3.0:
        print('WARNING    line pitch < 3 px, scanlines will alias. want '
              'height >= %d for %d lines.' % (NY * 3, NY), file=sys.stderr)
    if args.rgb and args.mask == 'dot':
        print('ERROR      --rgb is not supported in dot mode', file=sys.stderr)
        return 1

    ch = 3 if args.rgb else 1

    # vertical
    if args.scan > 0.0:
        vscan = build_scan(H, line, args.scan, args.scan_floor, S)
    else:
        vscan = [1.0] * H

    if args.mask == 'grille':
        vy = list(vscan)
        vy = apply_wires(vy, H, args.wires, args.wire_px, args.wire_dark, S)
        terms = [(None, vy)]
    elif args.mask == 'slot':
        sp = pitch * args.slot_aspect
        ve = build_v_rect(H, sp, args.slot_duty, args.soft, S, 0.0)
        vo = build_v_rect(H, sp, args.slot_duty, args.soft, S, sp * 0.5)
        terms = [(0, [a * b for a, b in zip(ve, vscan)]),
                 (1, [a * b for a, b in zip(vo, vscan)])]
    else:
        terms = None

    rows = []

    if args.mask == 'dot':
        tile, tw, th = build_dot_tile(pitch, args.dot_fill, args.soft, 4)
        print('dot tile   %dx%d px (pitch snapped to %d, %.2f triads across)'
              % (tw, th, tw, W / float(tw)))
        cols = [x % tw for x in range(W)]
        for y in range(H):
            trow = tile[y % th]
            v = vscan[y]
            rows.append([trow[c] * v for c in cols])
    elif not args.rgb:
        hs = [(build_h(W, pitch, args.fill, args.soft, S, None, par), vv)
              for par, vv in terms]
        for y in range(H):
            if len(hs) == 1:
                hx, vv = hs[0]
                v = vv[y]
                rows.append([a * v for a in hx])
            else:
                (h0, v0), (h1, v1) = hs
                a0, a1 = v0[y], v1[y]
                rows.append([p * a0 + q * a1 for p, q in zip(h0, h1)])
    else:
        chans = []
        for k in range(3):
            chans.append([(build_h(W, pitch, args.fill, args.soft, S, k, par), vv)
                          for par, vv in terms])
        for y in range(H):
            planes = []
            for k in range(3):
                hs = chans[k]
                if len(hs) == 1:
                    hx, vv = hs[0]
                    v = vv[y]
                    planes.append([a * v for a in hx])
                else:
                    (h0, v0), (h1, v1) = hs
                    a0, a1 = v0[y], v1[y]
                    planes.append([p * a0 + q * a1 for p, q in zip(h0, h1)])
            r, g, b = planes
            row = [0.0] * (W * 3)
            row[0::3] = r
            row[1::3] = g
            row[2::3] = b
            rows.append(row)

    total = 0.0
    for row in rows:
        total += math.fsum(row)
    mean = total / float(W * H * ch)
    print('mean        %.6f' % mean)

    if args.normalize:
        if mean <= 1e-9:
            print('ERROR      mean is zero, cannot normalize', file=sys.stderr)
            return 1
        g = 1.0 / mean
        print('gain        %.6f (mean scaled to 1.0)' % g)
        rows = [[v * g for v in row] for row in rows]

    if not args.png and not args.exr:
        args.exr = 'crt_%s_%dx%d.exr' % (args.mask, W, H)

    if args.exr:
        write_exr32(args.exr, W, H, rows, ch)
        print('wrote      %s' % args.exr)
    if args.png:
        if args.normalize:
            print('NOTE       PNG clips above 1.0, normalized values will crush',
                  file=sys.stderr)
        write_png16(args.png, W, H, rows, ch)
        print('wrote      %s' % args.png)

    return 0


if __name__ == '__main__':
    sys.exit(main())
