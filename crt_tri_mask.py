#!/usr/bin/env python3
# crt_mask.py
#
# CRT phosphor mask / scanline plate generator.
# Bakes linear coverage fields for compositing in Fusion.
# Pure stdlib. No numpy, no PIL, no OpenEXR.
#
# Run with no arguments for an interactive prompt.
# Run with flags for scripted / batch use. The interactive mode prints the
# equivalent command line when it finishes, so you can script it afterwards.
#
# The mask lattice (horizontal, tube property) and the scanline lattice
# (vertical, signal property) are independent. Set them separately.
#
#   mono    one plate, all three stripes lit
#   rgb     one RGB plate, stripes packed into R/G/B (use Channel Boolean)
#   split   three mono plates, _r _g _b (separate Loaders, then Merge/Add)

import argparse
import math
import os
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
    """Electron beam vertical profile. Raised cosine, not a hard line."""
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
    differ from it. Verified by exhaustive nearest-neighbour check.
    (The obvious-looking row-stagger rule (n + 2m) mod 3 is NOT valid, it
    collides on 17% of neighbour pairs.)
    Colours repeat over n+3 and m+6, so the tile is 3d by 3d*sqrt(3).
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


def exr_size_bytes(w, h):
    return 331 + 8 * h + h * (8 + 3 * w * 4)


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024.0:
            return '%.0f %s' % (n, u)
        n /= 1024.0
    return '%.1f TB' % n


# ----------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------

def parse_res(s):
    a, b = s.lower().replace(' ', '').split('x')
    a, b = int(a), int(b)
    if a < 8 or b < 8:
        raise ValueError('resolution too small')
    return a, b


def parse_conv(s):
    if s is None or s == '':
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


# ----------------------------------------------------------------------
# interactive prompts
# ----------------------------------------------------------------------

class Abort(Exception):
    pass


def _raw(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise Abort()


def ask_choice(label, options, default=0, note=None):
    """options: list of (value, text). Returns value."""
    print()
    print(label)
    if note:
        print('  ' + note)
    for i, (_, text) in enumerate(options):
        mark = '*' if i == default else ' '
        print('  %s %d) %s' % (mark, i + 1, text))
    while True:
        s = _raw('  choice [%d]: ' % (default + 1))
        if s == '':
            return options[default][0]
        try:
            k = int(s)
            if 1 <= k <= len(options):
                return options[k - 1][0]
        except ValueError:
            pass
        print('  enter 1..%d' % len(options))


def ask_yesno(label, default=True):
    d = 'Y/n' if default else 'y/N'
    while True:
        s = _raw('  %s [%s]: ' % (label, d)).lower()
        if s == '':
            return default
        if s in ('y', 'yes'):
            return True
        if s in ('n', 'no'):
            return False
        print('  enter y or n')


def ask_float(label, default, lo=None, hi=None):
    while True:
        s = _raw('  %s [%g]: ' % (label, default))
        if s == '':
            return default
        try:
            v = float(s)
        except ValueError:
            print('  enter a number')
            continue
        if lo is not None and v < lo:
            print('  must be >= %g' % lo)
            continue
        if hi is not None and v > hi:
            print('  must be <= %g' % hi)
            continue
        return v


def ask_int(label, default, lo=None, hi=None):
    while True:
        s = _raw('  %s [%d]: ' % (label, default))
        if s == '':
            return default
        try:
            v = int(s)
        except ValueError:
            print('  enter an integer')
            continue
        if lo is not None and v < lo:
            print('  must be >= %d' % lo)
            continue
        if hi is not None and v > hi:
            print('  must be <= %d' % hi)
            continue
        return v


def ask_text(label, default):
    s = _raw('  %s [%s]: ' % (label, default))
    return s if s else default


def ask_res(label, presets, default_idx):
    opts = [(p, p) for p in presets] + [(None, 'custom')]
    v = ask_choice(label, opts, default_idx)
    if v is not None:
        return parse_res(v)
    while True:
        s = _raw('  WxH: ')
        try:
            return parse_res(s)
        except Exception:
            print('  format is WIDTHxHEIGHT, e.g. 3840x2160')


def ask_conv(tag):
    while True:
        s = _raw('  %s offset "dx" or "dx,dy" [0]: ' % tag)
        if s == '':
            return 0.0, 0.0
        try:
            return parse_conv(s)
        except Exception:
            print('  format is dx or dx,dy')


def interactive(args):
    print()
    print('  CRT phosphor mask generator')
    print('  ' + '-' * 44)
    print('  enter for the marked default, ctrl-c to quit')

    W, H = ask_res(
        'Output plate resolution',
        ['1920x1080', '2560x1440', '3840x2160', '4096x2160', '7680x4320'],
        2)

    while True:
        NX, NY = ask_res(
            'Emulated CRT (triads across x scanlines down)',
            ['256x224', '320x240', '320x200', '640x480', '160x120'],
            1)
        pitch = W / float(NX)
        line = H / float(NY)
        sub = pitch / 3.0
        print()
        print('  triad pitch %.3f px, subcell %.3f px, line pitch %.3f px'
              % (pitch, sub, line))
        need_w = int(math.ceil(NX * 9))
        need_h = NY * 3
        bad_w = sub < 3.0
        bad_h = line < 3.0
        if not bad_w and not bad_h:
            break
        if bad_w:
            print('  WARNING subcell is %.2f px. below 3 px the mask washes out,'
                  % sub)
            print('          and at exactly 2.0 px it box-filters to flat grey.')
            print('          %d triads wants a plate at least %d px wide.'
                  % (NX, need_w))
        if bad_h:
            print('  WARNING line pitch is %.2f px, scanlines will alias.' % line)
            print('          %d lines wants a plate at least %d px tall.'
                  % (NY, need_h))
        fix = ask_choice('How do you want to handle that?', [
            ('bump', 'enlarge the plate to %dx%d'
             % (max(W, need_w) if bad_w else W, max(H, need_h) if bad_h else H)),
            ('cells', 'pick a different CRT resolution'),
            ('ignore', 'keep my settings anyway'),
        ], 0)
        if fix == 'bump':
            if bad_w:
                W = max(W, need_w)
            if bad_h:
                H = max(H, need_h)
            print('  plate is now %dx%d' % (W, H))
            break
        if fix == 'ignore':
            break

    mask = ask_choice('Shadow mask type', [
        ('slot', 'slot     vertical slots, staggered. most consumer sets'),
        ('grille', 'grille   continuous stripes. Trinitron / aperture grille'),
        ('dot', 'dot      triangular dot triads. older tubes'),
    ], 0)

    layout = ask_choice('Channel layout', [
        ('mono', 'mono     one plate, all three stripes lit'),
        ('split', 'split    three mono plates _r _g _b, Merge with Add'),
        ('rgb', 'rgb      one RGB plate, pull with Channel Boolean'),
    ], 0)

    fmt = ask_choice('File format', [
        ('exr', 'exr      32f linear, uncompressed. use this for Fusion'),
        ('png', 'png      16 bit. preview / reference only, clips at 1.0'),
        ('both', 'both'),
    ], 0)

    print()
    norm = False
    if fmt in ('exr', 'both'):
        print('  Normalizing scales the mean to 1.0 so the plate does not')
        print('  darken the shot. Recommended when you multiply against footage.')
        norm = ask_yesno('Normalize', True)
        if norm and fmt == 'both':
            print('  note: PNG clips at 1.0, so the PNG copy will be crushed.')

    print()
    scan = 0.75
    if ask_yesno('Include scanlines', True):
        scan = ask_float('beam width, fraction of line pitch', 0.75, 0.05, 1.0)
    else:
        scan = 0.0

    conv = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    if layout in ('split', 'rgb'):
        print()
        print('  Convergence error offsets each phosphor. Baked in at sample')
        print('  time, so no resampling softness. Positive moves the plate left.')
        if ask_yesno('Add convergence error', False):
            conv = [ask_conv('red'), ask_conv('green'), ask_conv('blue')]

    print()
    if ask_yesno('Adjust mask geometry (fill, softness, wires, quality)', False):
        args.fill = ask_float('stripe fill, fraction of subcell', args.fill,
                              0.05, 1.0)
        args.soft = ask_float('edge softness', args.soft, 0.0, 1.0)
        if mask == 'slot':
            args.slot_duty = ask_float('slot vertical fill', args.slot_duty,
                                       0.05, 1.0)
            args.slot_aspect = ask_float('slot pitch / triad pitch',
                                         args.slot_aspect, 0.1, 4.0)
        if mask == 'dot':
            args.dot_fill = ask_float('dot diameter / spacing', args.dot_fill,
                                      0.05, 1.2)
            args.dot_ss = ask_int('dot supersample per axis', args.dot_ss, 1, 16)
        if mask == 'grille':
            args.wires = ask_int('damper wires (0, 1 or 2)', args.wires, 0, 4)
        if scan > 0.0:
            args.scan_floor = ask_float('floor between scanlines',
                                        args.scan_floor, 0.0, 1.0)
        args.samples = ask_int('supersample rate', args.samples, 1, 128)

    print()
    nplate = 3 if layout == 'split' else 1
    stem = ask_text('Output name (no extension)',
                    'crt_%s_%dx%d' % (mask, W, H))

    args.out = '%dx%d' % (W, H)
    args.cells = '%dx%d' % (NX, NY)
    args.mask = mask
    args.rgb = (layout == 'rgb')
    args.split = (layout == 'split')
    args.normalize = norm
    args.scan = scan
    args.exr = (stem + '.exr') if fmt in ('exr', 'both') else None
    args.png = (stem + '.png') if fmt in ('png', 'both') else None
    args.conv_r = '%g,%g' % conv[0]
    args.conv_g = '%g,%g' % conv[1]
    args.conv_b = '%g,%g' % conv[2]

    # summary
    print()
    print('  ' + '-' * 44)
    print('  plate      %dx%d' % (W, H))
    print('  crt        %d triads x %d lines' % (NX, NY))
    print('  mask       %s' % mask)
    print('  layout     %s (%d file%s per format)'
          % (layout, nplate, '' if nplate == 1 else 's'))
    print('  format     %s%s' % (fmt, ', normalized' if norm else ''))
    if args.exr:
        tot = exr_size_bytes(W, H) * nplate
        print('  exr size   %s total' % human(tot))
        if tot > 512 * 1024 * 1024:
            print('             (uncompressed by design, Fusion caches it once)')
    est = (W * H * args.samples) / 6.0e7 * nplate
    if mask == 'dot':
        est = (W * H) / 8.0e6 * nplate
    print('  est time   ~%s' % ('%.0f s' % est if est < 90 else
                                '%.1f min' % (est / 60.0)))
    print('  ' + '-' * 44)

    cmd = ['./crt_mask.py', '--out', args.out, '--cells', args.cells,
           '--mask', mask]
    if args.split:
        cmd.append('--split')
    if args.rgb:
        cmd.append('--rgb')
    if args.normalize:
        cmd.append('--normalize')
    if scan == 0.0:
        cmd += ['--scan', '0']
    elif abs(scan - 0.75) > 1e-9:
        cmd += ['--scan', '%g' % scan]
    for tag, c in zip(('r', 'g', 'b'), conv):
        if c != (0.0, 0.0):
            cmd += ['--conv-%s' % tag, '%g,%g' % c]
    if args.exr:
        cmd += ['--exr', args.exr]
    if args.png:
        cmd += ['--png', args.png]
    print()
    print('  same thing as a command:')
    print('    ' + ' '.join(cmd))
    print()

    if not ask_yesno('Bake it', True):
        raise Abort()
    print()
    return args


# ----------------------------------------------------------------------
# render
# ----------------------------------------------------------------------

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
        return rows

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
    return rows


def mean_of(rows, n):
    total = 0.0
    for row in rows:
        total += math.fsum(row)
    return total / float(n)


# ----------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description='Bake CRT phosphor mask plates for Fusion. '
                    'Run with no arguments for interactive mode.')
    ap.add_argument('-i', '--interactive', action='store_true',
                    help='force the interactive prompt')
    ap.add_argument('--out', default='3840x2160',
                    help='output plate resolution, WxH')
    ap.add_argument('--cells', default='320x240',
                    help='emulated CRT: triads across x scanlines down')
    ap.add_argument('--mask', default='slot', choices=('grille', 'slot', 'dot'))
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
                    help='red convergence offset in output px, positive = left')
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
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    want_ui = args.interactive or len(sys.argv) == 1
    if want_ui:
        if not sys.stdin.isatty():
            print('ERROR      interactive mode needs a terminal. '
                  'pass flags instead, see --help', file=sys.stderr)
            return 2
        try:
            args = interactive(args)
        except Abort:
            print('\n  cancelled')
            return 130

    if args.rgb and args.split:
        print('ERROR      --rgb and --split are mutually exclusive',
              file=sys.stderr)
        return 1

    try:
        W, H = parse_res(args.out)
        NX, NY = parse_res(args.cells)
    except Exception:
        print('ERROR      resolution format is WIDTHxHEIGHT', file=sys.stderr)
        return 1

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

    if not want_ui:
        sub = pitch / 3.0
        if sub < 2.0:
            print('WARNING    subcell < 2 px, mask is below Nyquist, it will '
                  'vanish', file=sys.stderr)
        elif sub < 3.0:
            print('WARNING    subcell < 3 px, mask contrast will be very weak.',
                  file=sys.stderr)
            print('           at exactly 2.0 px a centered stripe box-filters '
                  'to a flat field.', file=sys.stderr)
            print('           want subcell >= 3 px, so width >= %d for %d '
                  'triads.' % (int(math.ceil(NX * 9)), NX), file=sys.stderr)
        if args.scan > 0.0 and line < 3.0:
            print('WARNING    line pitch < 3 px, scanlines will alias. want '
                  'height >= %d for %d lines.' % (NY * 3, NY), file=sys.stderr)

    if not args.png and not args.exr:
        args.exr = 'crt_%s_%dx%d.exr' % (args.mask, W, H)

    for p in (args.png, args.exr):
        if p:
            d = os.path.dirname(os.path.abspath(p))
            if not os.path.isdir(d):
                print('ERROR      no such directory: %s' % d, file=sys.stderr)
                return 1

    tags = ('r', 'g', 'b')

    # ---------------- split: three mono plates ----------------
    if args.split:
        for k in range(3):
            rows = render_mono(args, W, H, pitch, line, k, conv[k])
            m = mean_of(rows, W * H)
            note = '  conv %+.3f,%+.3f' % conv[k] if conv[k] != (0.0, 0.0) else ''
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
            rows = render_mono(args, W, H, pitch, line, k, conv[k])
            m = mean_of(rows, W * H)
            note = '  conv %+.3f,%+.3f' % conv[k] if conv[k] != (0.0, 0.0) else ''
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
        rows = render_mono(args, W, H, pitch, line, None, (0.0, 0.0))
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
