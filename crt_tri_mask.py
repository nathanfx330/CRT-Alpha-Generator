#!/usr/bin/env python3
# crt_mask.py
#
# CRT phosphor mask / scanline plate generator.
# Bakes linear coverage fields for compositing in Fusion.
# Pure stdlib. No numpy, no PIL, no OpenEXR.
#
# The mask lattice (horizontal, tube property) and the scanline lattice
# (vertical, signal property) are independent. Set them separately.
#
#   default   one mono plate, all three stripes lit
#   --rgb     one RGB plate, stripes packed into R/G/B (use Channel Boolean)
#   --split   three mono plates, _r _g _b (separate Loaders, then Merge)

import argparse
import math
import struct
import sys
import zlib

SQRT3 = math.sqrt(3.0)


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

def build_h(width, pitch, duty, soft, samples, subcell=None, parity=None,
            dx=0.0):
    """Horizontal stripe array.

    subcell: None for all three stripes, or 0/1/2 to isolate one phosphor.
    parity:  None for every triad column, or 0/1 for even/odd columns.
             Slot mode uses this to stay separable.
    dx:      convergence offset in output pixels.
    """
    out = [0.0] * width
    inv = 1.0 / samples
    sub = pitch / 3.0
    for i in range(width):
        acc = 0.0
        for s in range(samples):
            x = i + (s + 0.5) * inv + dx
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


def build_v_rect(height, pitch, duty, soft, samples, phase=0.0, dy=0.0):
    """Vertical rect train. Slot mask vertical structure."""
    out = [0.0] * height
    inv = 1.0 / samples
    for i in range(height):
        acc = 0.0
        for s in range(samples):
            y = i + (s + 0.5) * inv + dy
            t = ((y - phase) / pitch) % 1.0
            acc += profile(t, duty, soft)
        out[i] = acc * inv
    return out


def build_scan(height, pitch, beam, floor_, samples, dy=0.0):
    """Electron beam vertical profile. Raised cosine, not a hard line.

    The beam is a signal property, so convergence offsets do NOT apply here
    unless you are simulating vertical misconvergence. dy is passed only when
    the caller wants that.
    """
    out = [0.0] * height
    inv = 1.0 / samples
    w = max(beam * 0.5, 1e-9)
    for i in range(height):
        acc = 0.0
        for s in range(samples):
            y = i + (s + 0.5) * inv + dy
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
    """Trinitron damper wires. Thin horizontal shadows, tube property."""
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


# ----------------------------------------------------------------------
# dot triad (shadow mask)
# ----------------------------------------------------------------------

def dot_tile_dims(pitch):
    """Tile size for a 3-colourable triangular lattice.

    Dot spacing d = pitch/sqrt(3) puts phosphor density in the same range as
    a grille of the same triad pitch. Lattice basis is
        x = n*d + m*d/2 ,  y = m*d*sqrt(3)/2
    and the colouring (n - m) mod 3 is valid: all six neighbours of a site
    differ from it. That colouring repeats over n+3 and m+6, so the tile is
    3d wide by 3d*sqrt(3) tall.
    """
    d = pitch / SQRT3
    tw = max(3, int(round(3.0 * d)))
    th = max(3, int(round(3.0 * d * SQRT3)))
    return tw, th


def build_dot_tile(pitch, fill, soft, ss, subcell=None, dx=0.0, dy=0.0):
    tw, th = dot_tile_dims(pitch)
    d = tw / 3.0
    rp = th / 6.0
    r = d * fill * 0.5
    sw = soft * d

    sites = []
    for m in range(-3, 10):
        for n in range(-3, 7):
            if subcell is not None and ((n - m) % 3) != subcell:
                continue
            sites.append((n * d + m * d * 0.5 - dx, m * rp - dy))

    inv = 1.0 / ss
    tile = []
    for y in range(th):
        row = [0.0] * tw
        for x in range(tw):
            acc = 0.0
            for sy in range(ss):
                yy = y + (sy + 0.5) * inv
                for sx in range(ss):
                    xx = x + (sx + 0.5) * inv
                    best = 1e18
                    for (cx, cy) in sites:
                        ddx = xx - cx
                        ddy = yy - cy
                        d2 = ddx * ddx + ddy * ddy
                        if d2 < best:
                            best = d2
                    acc += radial(math.sqrt(best), r, sw)
            row[x] = acc * inv * inv
        tile.append(row)
    return tile, tw, th


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


def parse_conv(s):
    if s is None:
        return 0.0, 0.0
    p = s.replace(' ', '').split(',')
    if len(p) == 1:
        return float(p[0]), 0.0
    if len(p) == 2:
        return float(p[0]), float(p[1])
    raise ValueError('convergence must be "dx" or "dx,dy"')


def suffix_path(path, tag):
    if path is None:
        return None
    i = path.rfind('.')
    j = max(path.rfind('/'), path.rfind('\\'))
    if i <= j:
        return path + '_' + tag
    return path[:i] + '_' + tag + path[i:]


def render_mono(args, W, H, pitch, line, subcell, conv):
    """Build one monochrome plate. subcell None = all stripes."""
    dx, dy = conv
    S = max(1, args.samples)

    if args.scan > 0.0:
        vscan = build_scan(H, line, args.scan, args.scan_floor, S,
                           dy if args.conv_vertical else 0.0)
    else:
        vscan = [1.0] * H

    if args.mask == 'dot':
        tile, tw, th = build_dot_tile(pitch, args.dot_fill, args.soft,
                                      args.dot_ss, subcell, dx, dy)
        cols = [x % tw for x in range(W)]
        rows = []
        for y in range(H):
            trow = tile[y % th]
            v = vscan[y]
            rows.append([trow[c] * v for c in cols])
        return rows, (tw, th)

    if args.mask == 'grille':
        vy = list(vscan)
        vy = apply_wires(vy, H, args.wires, args.wire_px, args.wire_dark, S)
        terms = [(None, vy)]
    else:  # slot
        sp = pitch * args.slot_aspect
        ve = build_v_rect(H, sp, args.slot_duty, args.soft, S, 0.0, dy)
        vo = build_v_rect(H, sp, args.slot_duty, args.soft, S, sp * 0.5, dy)
        terms = [(0, [a * b for a, b in zip(ve, vscan)]),
                 (1, [a * b for a, b in zip(vo, vscan)])]

    hs = [(build_h(W, pitch, args.fill, args.soft, S, subcell, par, dx), vv)
          for par, vv in terms]

    rows = []
    for y in range(H):
        if len(hs) == 1:
            hx, vv = hs[0]
            v = vv[y]
            rows.append([a * v for a in hx])
        else:
            (h0, v0), (h1, v1) = hs
            a0, a1 = v0[y], v1[y]
            rows.append([p * a0 + q * a1 for p, q in zip(h0, h1)])
    return rows, None


def mean_of(rows, n):
    total = 0.0
    for row in rows:
        total += math.fsum(row)
    return total / float(n)


def main():
    ap = argparse.ArgumentParser(
        description='Bake CRT phosphor mask plates for Fusion.')
    ap.add_argument('--out', default='3840x2160',
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
                    help='dot mode: dot diameter / dot spacing')
    ap.add_argument('--dot-ss', type=int, default=4,
                    help='dot mode: 2D supersample rate per axis')
    ap.add_argument('--scan', type=float, default=0.75,
                    help='beam width as fraction of line pitch, 0 disables')
    ap.add_argument('--scan-floor', type=float, default=0.0,
                    help='floor between scanlines, 0 is fully black')
    ap.add_argument('--wires', type=int, default=0,
                    help='grille mode: damper wire count (0, 1 or 2)')
    ap.add_argument('--wire-px', type=float, default=1.5)
    ap.add_argument('--wire-dark', type=float, default=0.45)

    ap.add_argument('--rgb', action='store_true',
                    help='one RGB plate, stripes packed into R/G/B')
    ap.add_argument('--split', action='store_true',
                    help='three mono plates written as _r _g _b')

    ap.add_argument('--conv-r', default=None, metavar='DX[,DY]',
                    help='red convergence offset in output px')
    ap.add_argument('--conv-g', default=None, metavar='DX[,DY]')
    ap.add_argument('--conv-b', default=None, metavar='DX[,DY]')
    ap.add_argument('--conv-vertical', action='store_true',
                    help='also apply DY to the scanline beam, not just the mask')

    ap.add_argument('--normalize', action='store_true',
                    help='scale mean to 1.0 so the plate does not darken')
    ap.add_argument('--samples', type=int, default=32,
                    help='1D supersample rate')
    ap.add_argument('--png', default=None)
    ap.add_argument('--exr', default=None)
    args = ap.parse_args()

    if args.rgb and args.split:
        print('ERROR      --rgb and --split are mutually exclusive',
              file=sys.stderr)
        return 1

    W, H = parse_res(args.out)
    NX, NY = parse_res(args.cells)
    pitch = W / float(NX)
    line = H / float(NY)

    try:
        conv = [parse_conv(args.conv_r), parse_conv(args.conv_g),
                parse_conv(args.conv_b)]
    except ValueError as e:
        print('ERROR      %s' % e, file=sys.stderr)
        return 1

    print('plate      %dx%d' % (W, H))
    print('mask       %s' % args.mask)
    print('triads     %d across, pitch %.4f px, subcell %.4f px'
          % (NX, pitch, pitch / 3.0))
    print('scanlines  %d, pitch %.4f px' % (NY, line))
    if args.mask == 'dot':
        tw, th = dot_tile_dims(pitch)
        print('dot tile   %dx%d px, spacing %.4f px' % (tw, th, tw / 3.0))

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

    if not args.png and not args.exr:
        args.exr = 'crt_%s_%dx%d.exr' % (args.mask, W, H)

    tags = ('r', 'g', 'b')

    # ---------------- split: three mono plates ----------------
    if args.split:
        for k in range(3):
            rows, _ = render_mono(args, W, H, pitch, line, k, conv[k])
            m = mean_of(rows, W * H)
            note = ''
            if conv[k] != (0.0, 0.0):
                note = '  conv %+.3f,%+.3f' % conv[k]
            print('%s          mean %.6f%s' % (tags[k].upper(), m, note))
            if args.normalize:
                if m <= 1e-9:
                    print('ERROR      %s mean is zero, cannot normalize'
                          % tags[k].upper(), file=sys.stderr)
                    return 1
                g = 1.0 / m
                print('           gain %.6f' % g)
                rows = [[v * g for v in row] for row in rows]
            if args.exr:
                p = suffix_path(args.exr, tags[k])
                write_exr32(p, W, H, rows, 1)
                print('wrote      %s' % p)
            if args.png:
                p = suffix_path(args.png, tags[k])
                write_png16(p, W, H, rows, 1)
                print('wrote      %s' % p)
        if args.normalize and args.png:
            print('NOTE       PNG clips above 1.0, normalized values will crush',
                  file=sys.stderr)
        return 0

    # ---------------- rgb: one packed plate ----------------
    if args.rgb:
        planes = []
        for k in range(3):
            rows, _ = render_mono(args, W, H, pitch, line, k, conv[k])
            m = mean_of(rows, W * H)
            note = ''
            if conv[k] != (0.0, 0.0):
                note = '  conv %+.3f,%+.3f' % conv[k]
            print('%s          mean %.6f%s' % (tags[k].upper(), m, note))
            if args.normalize:
                if m <= 1e-9:
                    print('ERROR      %s mean is zero, cannot normalize'
                          % tags[k].upper(), file=sys.stderr)
                    return 1
                rows = [[v / m for v in row] for row in rows]
            planes.append(rows)
        rows = []
        for y in range(H):
            r, g, b = planes[0][y], planes[1][y], planes[2][y]
            row = [0.0] * (W * 3)
            row[0::3] = r
            row[1::3] = g
            row[2::3] = b
            rows.append(row)
        ch = 3
    # ---------------- default: one mono plate ----------------
    else:
        rows, _ = render_mono(args, W, H, pitch, line, None, (0.0, 0.0))
        m = mean_of(rows, W * H)
        print('mean        %.6f' % m)
        if args.normalize:
            if m <= 1e-9:
                print('ERROR      mean is zero, cannot normalize', file=sys.stderr)
                return 1
            g = 1.0 / m
            print('gain        %.6f (mean scaled to 1.0)' % g)
            rows = [[v * g for v in row] for row in rows]
        ch = 1

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
