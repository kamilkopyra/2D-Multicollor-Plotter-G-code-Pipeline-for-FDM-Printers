#!/usr/bin/env python3
"""
image_processor.py  —  efekty graficzne dla obrazów
=============================================
Wymagania: pip install Pillow numpy scipy opencv-python-headless

Użycie:
    python image_processor.py obraz.jpg --mode sketch
    python image_processor.py obraz.jpg --mode stippling --mode_size 3000
    python image_processor.py obraz.jpg --mode voronoi --mode_size 5000
    python image_processor.py obraz.jpg --mode ascii --mode_size 6
    python image_processor.py obraz.jpg --mode ascii_color --mode_size 6
    python image_processor.py obraz.jpg --mode voronoi_color --mode_size 5000
    python image_processor.py obraz.jpg --mode stippling_color --mode_size 3000
    python image_processor.py obraz.jpg --mode sketch_color
"""

import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


# ── tryby B&W ────────────────────────────────────────────────────────────────

def apply_ascii(arr, chars=' .:-=+*#@$%^&?', cell=4):
    """ASCII art — jasność kratki → znak."""
    H, W = arr.shape
    img = Image.fromarray(np.full(arr.shape, 255, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    for y in range(0, H, cell):
        for x in range(0, W, cell):
            patch = arr[y:y+cell, x:x+cell]
            brightness = patch.mean() / 255.0
            ch = chars[int(brightness * (len(chars) - 1))]
            if ch != ' ':
                draw.text((x, y), ch, fill=0)
    return np.array(img)


def apply_voronoi(arr, n_points=5000):
    """Diagram Voronoi z punktów maski."""
    from scipy.spatial import Voronoi
    H, W = arr.shape
    black = np.argwhere(arr < 128)
    if len(black) < 4:
        return arr
    idx = np.random.choice(len(black), min(n_points, len(black)), replace=False)
    pts = black[idx][:, ::-1].astype(float)
    corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=float)
    pts = np.vstack([pts, corners])
    vor = Voronoi(pts)
    out = Image.fromarray(np.full((H, W), 255, dtype=np.uint8))
    draw = ImageDraw.Draw(out)
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=0, width=1)
    return np.array(out)


def apply_stippling(arr, n_points=2000, dot_radius=2):
    """Kropkowanie — gęstość zależy od ciemności."""
    H, W = arr.shape
    prob = (255 - arr).astype(float)
    prob /= prob.sum()
    flat_idx = np.random.choice(H * W, size=n_points, replace=False, p=prob.ravel())
    ys, xs = np.unravel_index(flat_idx, (H, W))
    out = Image.fromarray(np.full((H, W), 255, dtype=np.uint8))
    draw = ImageDraw.Draw(out)
    for x, y in zip(xs, ys):
        draw.ellipse(
            [(x - dot_radius, y - dot_radius),
             (x + dot_radius, y + dot_radius)],
            fill=0
        )
    return np.array(out)


def apply_sketch(arr, blur=21, threshold1=30, threshold2=100):
    """Szkic ołówkowy przez detekcję krawędzi Canny."""
    import cv2
    blurred = cv2.GaussianBlur(arr, (blur, blur), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return cv2.bitwise_not(edges)


# ── tryby kolorowe ────────────────────────────────────────────────────────────

def apply_ascii_color(img_rgb, chars=' .:-=+*#@$%^&?', cell=4):
    """ASCII art — znak ma kolor oryginalnego piksela, czarne tło."""
    arr_gray = np.array(img_rgb.convert('L'))
    arr_rgb  = np.array(img_rgb)
    H, W = arr_gray.shape
    out = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    for y in range(0, H, cell):
        for x in range(0, W, cell):
            patch = arr_gray[y:y+cell, x:x+cell]
            brightness = patch.mean() / 255.0
            ch = chars[int(brightness * (len(chars) - 1))]
            if ch != ' ':
                cy = min(y + cell//2, H-1)
                cx = min(x + cell//2, W-1)
                color = tuple(arr_rgb[cy, cx])
                draw.text((x, y), ch, fill=color)
    return np.array(out)


def apply_voronoi_color(img_rgb, n_points=5000):
    """Diagram Voronoi — krawędzie w kolorze najbliższego punktu."""
    from scipy.spatial import Voronoi
    arr_gray = np.array(img_rgb.convert('L'))
    arr_rgb  = np.array(img_rgb)
    H, W = arr_gray.shape
    black = np.argwhere(arr_gray < 128)
    if len(black) < 4:
        return arr_rgb
    idx = np.random.choice(len(black), min(n_points, len(black)), replace=False)
    pts = black[idx][:, ::-1].astype(float)  # (x, y)
    colors = [tuple(arr_rgb[int(p[1]), int(p[0])]) for p in pts]
    corners = np.array([[0,0],[W,0],[0,H],[W,H]], dtype=float)
    corner_colors = [(128,128,128)] * 4
    pts = np.vstack([pts, corners])
    colors = colors + corner_colors
    vor = Voronoi(pts)
    out = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    for ridge_idx, ridge in enumerate(vor.ridge_vertices):
        if -1 in ridge:
            continue
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        # kolor z pierwszego punktu ridge
        pt_idx = vor.ridge_points[ridge_idx][0]
        color = colors[pt_idx] if pt_idx < len(colors) else (128,128,128)
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=color, width=1)
    return np.array(out)


def apply_stippling_color(img_rgb, n_points=2000, dot_radius=2):
    """Kropkowanie — kropka ma kolor oryginalnego piksela."""
    arr_gray = np.array(img_rgb.convert('L'))
    arr_rgb  = np.array(img_rgb)
    H, W = arr_gray.shape
    prob = (255 - arr_gray).astype(float)
    prob /= prob.sum()
    flat_idx = np.random.choice(H * W, size=n_points, replace=False, p=prob.ravel())
    ys, xs = np.unravel_index(flat_idx, (H, W))
    out = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    for x, y in zip(xs, ys):
        color = tuple(arr_rgb[y, x])
        draw.ellipse(
            [(x - dot_radius, y - dot_radius),
             (x + dot_radius, y + dot_radius)],
            fill=color
        )
    return np.array(out)


def apply_sketch_color(img_rgb, blur=21, threshold1=30, threshold2=100):
    """Szkic Canny — krawędzie w kolorze oryginalnego piksela, czarne tło."""
    import cv2
    arr_gray = np.array(img_rgb.convert('L'))
    arr_rgb  = np.array(img_rgb)
    blurred = cv2.GaussianBlur(arr_gray, (blur, blur), 0)
    edges = cv2.Canny(blurred, threshold1, threshold2)
    # maska krawędzi nałożona na kolor
    out = np.zeros((*arr_gray.shape, 3), dtype=np.uint8)
    mask = edges > 0
    out[mask] = arr_rgb[mask]
    return out


# ── main ─────────────────────────────────────────────────────────────────────

MODES = ['ascii', 'ascii_color', 'voronoi', 'voronoi_color',
         'stippling', 'stippling_color', 'sketch', 'sketch_color']

def main():
    p = argparse.ArgumentParser(description="Efekty graficzne dla obrazów")
    p.add_argument('input',              type=Path)
    p.add_argument('-o', '--output',     type=Path, default=None)
    p.add_argument('--mode',             choices=MODES, default='sketch')
    p.add_argument('--mode_size',        type=int, default=8,
                   help="Rozmiar kratki (ascii) / liczba punktów (voronoi, stippling)")
    p.add_argument('--stippling_radius', type=int, default=2)
    p.add_argument('--sketch_blur',      type=int, default=21)
    p.add_argument('--sketch_t1',        type=int, default=30)
    p.add_argument('--sketch_t2',        type=int, default=100)
    args = p.parse_args()

    if not args.input.exists():
        print(f"Błąd: nie znaleziono {args.input}", file=sys.stderr)
        sys.exit(1)

    img_rgb  = Image.open(args.input).convert('RGB')
    img_gray = img_rgb.convert('L')
    arr_gray = np.array(img_gray)

    print(f"  Wczytano: {args.input}  ({img_rgb.width}x{img_rgb.height}px)")
    print(f"  Tryb: {args.mode}")

    if   args.mode == 'ascii':
        result = apply_ascii(arr_gray, cell=args.mode_size)
    elif args.mode == 'ascii_color':
        result = apply_ascii_color(img_rgb, cell=args.mode_size)
    elif args.mode == 'voronoi':
        result = apply_voronoi(arr_gray, n_points=args.mode_size)
    elif args.mode == 'voronoi_color':
        result = apply_voronoi_color(img_rgb, n_points=args.mode_size)
    elif args.mode == 'stippling':
        result = apply_stippling(arr_gray, n_points=args.mode_size, dot_radius=args.stippling_radius)
    elif args.mode == 'stippling_color':
        result = apply_stippling_color(img_rgb, n_points=args.mode_size, dot_radius=args.stippling_radius)
    elif args.mode == 'sketch':
        result = apply_sketch(arr_gray, args.sketch_blur, args.sketch_t1, args.sketch_t2)
    elif args.mode == 'sketch_color':
        result = apply_sketch_color(img_rgb, args.sketch_blur, args.sketch_t1, args.sketch_t2)

    out = args.output or args.input.with_name(f"{args.input.stem}_{args.mode}.png")
    Image.fromarray(result).save(out)
    print(f"  Zapisano: {out}\n")


if __name__ == '__main__':
    main()