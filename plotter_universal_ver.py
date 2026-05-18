#!/usr/bin/env python3
"""
plotter_pipeline.py  —  obraz → jeden G-code dla plotera 2D
Wymagania: pip install Pillow scikit-learn numpy vtracer vpype vpype-gcode
"""

import argparse, re, sys, tempfile
from pathlib import Path
import numpy as np
import vtracer
from PIL import Image
from sklearn.cluster import KMeans

Z_UP_DEFAULT    = 5.0
# defaultowe ustawienia prędkości
FEEDRATE_DRAW   = 3000
FEEDRATE_TRAVEL = 8000
FEEDRATE_Z      = 1000


# ustawienia położenia fotoklatki
PHOTO_X_DEFAULT = 10.0
PHOTO_Y_DEFAULT = 10.0
PHOTO_WAIT_MS   = 5000


# krok 1: rozdzielanie kolorów 
# rozdzielam obraz na n części, dla każdej z nich
# generowany będzie osobny gcode - niezbędne do obsługi
# kilku pisaków

# Biblioteki:
# - Pillow - otwieranie i zapisywanie obrazów
# - NumPy - operacje na tablicach pikseli
# scikit-leart - algorytm K-means odpowiedzialny za rozdzielenie kolorów

def split_colors(img_path, n_colors, tmp_dir, colors_dir):
    print(f"\n[1/5] Rozdzielam kolory ({n_colors} klastrów)...")

    img = Image.open(img_path).convert('RGB')
    W, H = img.size

    pixels = np.array(img).reshape(-1, 3).astype(float)

    km = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(int)

    results = []

    for i, color in enumerate(centers):
        r, g, b = color
        hex_col = f"#{r:02X}{g:02X}{b:02X}"

        pct = (labels == i).sum() / len(labels) * 100

        mask = (labels == i).reshape(H, W)

        arr = np.full((H, W), 255, dtype=np.uint8)
        arr[mask] = 0

        name = f"kolor_{i+1:02d}_{hex_col}.png"

        img_out = Image.fromarray(arr, mode='L')
        img_out.save(tmp_dir / name)
        img_out.save(colors_dir / name)

        print(f"    kolor {i+1}: {hex_col}  ({pct:.1f}%)  → {colors_dir/name}")

        results.append((tmp_dir / name, hex_col))

    return results, W, H


#  krok 2: wektoryzacja
#  przekonwertowanie obrazu na plik svg

# Biblioteka vtracker  

def vectorize(png_path, svg_path):
    vtracer.convert_image_to_svg_py(
        str(png_path), str(svg_path),
        colormode='binary',
        filter_speckle=4,
        corner_threshold=60,
        length_threshold=4.0,
        path_precision=3,
    )


# krok 3: SVG -> raw G-code
# biblioteka vpype
def svg_to_raw_gcode(svg_path, gcode_path):
    from vpype_cli import execute

    execute(
        f"read '{svg_path}' "
        f"linemerge --tolerance 0.5mm "
        f"linesort "    # optymalizacja ścieżek
        f"gwrite --profile gcodemm '{gcode_path}'" # w mm
    )


# mierzy rozmiar obiektu reprezentowanego przez gcode
def measure_bounds(gcode_path):
    text = gcode_path.read_text(encoding='utf-8', errors='replace')

    xs = [float(m.group(1)) for m in re.finditer(r'X(-?[\d.]+)', text)]
    ys = [float(m.group(1)) for m in re.finditer(r'Y(-?[\d.]+)', text)]

    if not xs or not ys:
        return None

    return min(xs), max(xs), min(ys), max(ys)


# krok 4: wspólna skala i przesunięcie 

def transform_gcode(gcode_path, scale, shift_x, shift_y):
    text = gcode_path.read_text(encoding='utf-8', errors='replace')

    text = re.sub(
        r'X(-?[\d.]+)',
        lambda m: f"X{float(m.group(1))*scale + shift_x:.4f}",
        text
    )

    text = re.sub(
        r'Y(-?[\d.]+)',
        lambda m: f"Y{float(m.group(1))*scale + shift_y:.4f}",
        text
    )

    gcode_path.write_text(text, encoding='utf-8')


#  krok 5: filtrowanie i składanie jednego G-code 

# filtrowanie poszczególnych komend 
RE_HEAT   = re.compile(r'^\s*M(104|109|140|190|141|116)\b', re.I)
RE_FAN    = re.compile(r'^\s*M(106|107)\b', re.I)
RE_G28    = re.compile(r'^\s*G28\b', re.I)
RE_E_ONLY = re.compile(r'^\s*[GT]\d+\s+E-?[\d.]+\s*(;.*)?$', re.I)
RE_E_AXIS = re.compile(r'\bE-?[\d.]+\s*', re.I)
RE_Z_MOVE = re.compile(r'^\s*(G[01])\b(.*?)\bZ\s*(-?[\d.]+)(.*?)$', re.I)
RE_F      = re.compile(r'\bF[\d.]+', re.I)


def filter_gcode(lines, z_up):
    out = []

    for line in lines:
        s = line.strip()

        if not s or s.startswith(';'):
            out.append(line)
            continue

        if RE_HEAT.match(s) or RE_FAN.match(s) or RE_G28.match(s) or RE_E_ONLY.match(s):
            continue

        zm = RE_Z_MOVE.match(s)

        if zm:
            cmd  = zm.group(1).upper()
            pre  = RE_F.sub('', zm.group(2)).strip()
            zval = float(zm.group(3))
            post = RE_F.sub('', zm.group(4)).strip()

            if zval <= z_up * 0.5:
                nz, feed, cmt = 0.0, FEEDRATE_DRAW, "; piszak W DOL"
            else:
                nz, feed, cmt = z_up, FEEDRATE_TRAVEL, "; piszak w gore"

            parts = [cmd, f"Z{nz:.2f}", f"F{feed}"]

            if pre:
                parts.append(pre)

            if post:
                parts.append(post)

            parts.append(cmt)
            out.append(' '.join(parts))
            continue

        if RE_E_AXIS.search(s):
            out.append(RE_E_AXIS.sub('', s).strip())
            continue

        out.append(line)

    return out


# dla bambu ponoć M0 nie działa
def pause_cmd(printer, msg):
    if printer == 'bambu':
        return f"M400 U1 ; {msg}"
    return f"M0 {msg}"

# dodałem sekwencję startową żeby oszukać Bambu, ale nie działa w slicerze i tak
def start_seq(z_up, first_color, printer):
    if printer == 'bambu':
        header = [
            "; generated by BambuStudio",
            "; HEADER_BLOCK_START",
            "; printer_model = Bambu Lab",
            "; nozzle_diameter = 0.4",
            "; bed_type = Textured PEI Plate",
            "; HEADER_BLOCK_END",
            "",
            "M104 S0",
            "M140 S0",
            "M107",
            "",
            "G21 ; mm",
            "G90 ; absolute positioning",
            "M83 ; relative extrusion",
            "G17 ; XY plane",
            "G94 ; feed per minute",
            "M220 S100",
            "M221 S100",
            "",
        ]
    else:
        header = [
            "; ╔══════════════════════════════════╗",
            "; ║  PLOTTER 2D — plotter_pipeline   ║",
            "; ╚══════════════════════════════════╝",
            "",
            "M104 S0",
            "M140 S0",
            "M107",
            "",
            "G21 ; mm",
            "G90 ; absolute positioning",
            "G17 ; XY plane",
            "G94 ; feed per minute",
            "",
        ]

    return header + [
        "G28",
        "M420 S0",
        f"G0 Z{z_up:.2f} F{FEEDRATE_Z}",
        f"G0 X50 Y50 F{FEEDRATE_TRAVEL}",
        "",
        f"; PAUZA — wloz pisak {first_color}, opusc Z az dotknie papieru",
        pause_cmd(printer, "Wloz pisak i ustaw Z, wznow z ekranu"),
        "",
        "G92 Z0",
        f"G0 Z{z_up:.2f} F{FEEDRATE_Z}",
        "",
        "; ─── START ──────────────────────────",
        "",
    ]


def photo_seq(photo_x, photo_y, z_up, wait_ms):
    return [
        "",
        "; ─── FOTO-KLATKA / ZMIANA PISAKA ───",
        f"G0 Z{z_up:.2f} F{FEEDRATE_Z}",
        f"G0 X{photo_x:.1f} Y{photo_y:.1f} F{FEEDRATE_TRAVEL}",
        f"G4 P{wait_ms}",   # czeka na serwo i zmianę pisaka
        f"G0 Z{z_up:.2f} F{FEEDRATE_Z}",
        "",
    ]


def end_seq(z_up):
    return [
        "",
        "; ─── KONIEC ─────────────────────────",
        f"G0 Z{z_up:.2f} F{FEEDRATE_Z}",
        f"G0 X0 Y0 F{FEEDRATE_TRAVEL}",
        "M84",
    ]


def save_combined(gcode_files, colors, stem, out_dir, z_up, photo_x, photo_y, photo_wait, printer):
    print(f"\n[5/5] Zapisuję jeden plik G-code...")

    out_path = out_dir / f"{stem}_plotter.gcode"

    lines = start_seq(z_up, colors[0], printer)

    for idx, (gcode_path, color_hex) in enumerate(zip(gcode_files, colors)):
        raw = gcode_path.read_text(encoding='utf-8', errors='replace').splitlines()
        filtered = filter_gcode(raw, z_up)

        lines.append("")
        lines.append(f"; ═══ KOLOR {idx+1} → {color_hex} ═══")
        lines.extend(filtered)

        lines.extend(photo_seq(photo_x, photo_y, z_up, photo_wait))

        

    lines.extend(end_seq(z_up))

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f"    {out_path.name}")

    return out_path
 

def main():
    p = argparse.ArgumentParser(description="Pipeline: obraz → jeden G-code plotera 2D")

    p.add_argument('input',          type=Path)
    p.add_argument('-n', '--colors', type=int,   default=4)
    p.add_argument('--z-up',         type=float, default=Z_UP_DEFAULT)
    p.add_argument('--margin',       type=float, default=5.0)
    p.add_argument('--paper',        choices=['a4', 'a5', 'a3'], default='a4')

    p.add_argument('--photo-x',      type=float, default=PHOTO_X_DEFAULT)
    p.add_argument('--photo-y',      type=float, default=PHOTO_Y_DEFAULT)
    p.add_argument('--photo-wait',   type=int,   default=PHOTO_WAIT_MS)

    p.add_argument(
        '--printer',
        choices=['marlin', 'bambu'],
        default='marlin',
        help="Typ drukarki: marlin używa M0, bambu używa M400 U1 i header Bambu-like"
    )

    args = p.parse_args()

    paper = {'a4': (210, 297), 'a5': (148, 210), 'a3': (297, 420)}[args.paper]

    area_w = paper[0] - 2 * args.margin
    area_h = paper[1] - 2 * args.margin

    if not args.input.exists():
        print(f"Błąd: nie znaleziono {args.input}", file=sys.stderr)
        sys.exit(1)

    colors_dir = args.input.parent / f"{args.input.stem}_kolory"
    colors_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        color_data, img_w, img_h = split_colors(args.input, args.colors, tmp_dir, colors_dir)

        print(f"\n[2/5] Wektoryzuję PNG → SVG...")

        svg_files = []

        for png_path, hex_col in color_data:
            svg_path = tmp_dir / png_path.with_suffix('.svg').name
            vectorize(png_path, svg_path)
            svg_files.append(svg_path)
            print(f"    {png_path.name}")

        print(f"\n[3/5] Generuję raw G-code...")

        raw_gcodes = []

        for svg_path, (_, _) in zip(svg_files, color_data):
            gc_path = tmp_dir / svg_path.with_suffix('.gcode').name
            svg_to_raw_gcode(svg_path, gc_path)
            raw_gcodes.append(gc_path)

        print(f"\n[4/5] Mierzę i skaluję...")

        all_bounds = [measure_bounds(gc) for gc in raw_gcodes]
        all_bounds = [b for b in all_bounds if b]

        global_min_x = min(b[0] for b in all_bounds)
        global_max_x = max(b[1] for b in all_bounds)
        global_min_y = min(b[2] for b in all_bounds)
        global_max_y = max(b[3] for b in all_bounds)

        raw_w = global_max_x - global_min_x
        raw_h = global_max_y - global_min_y

        scale = min(area_w / raw_w, area_h / raw_h)

        shift_x = args.margin - global_min_x * scale
        shift_y = args.margin - global_min_y * scale

        print(f"    raw bbox: {raw_w:.1f} x {raw_h:.1f} mm")
        print(f"    scale: {scale:.4f}")
        print(f"    wynik: {raw_w*scale:.1f} x {raw_h*scale:.1f} mm na papierze {args.paper}")

        for gc_path in raw_gcodes:
            transform_gcode(gc_path, scale, shift_x, shift_y)

        colors = [hc for _, hc in color_data]

        saved = save_combined(
            raw_gcodes,
            colors,
            args.input.stem,
            args.input.parent,
            args.z_up,
            args.photo_x,
            args.photo_y,
            args.photo_wait,
            args.printer,
        )

    print(f"\n  Zapisany plik:")
    print(f"    {saved}")
    print(f"  Podgląd PNG : {colors_dir.resolve()}\n")


if __name__ == '__main__':
    main()

# przykład wywołania:
# wszystkie argumenty
#python plotter_bambu_ver.py obraz.webp --printer bambu -n 4 --paper a4 --photo-x 200 --photo-y 10 --photo-wait 5000

# minimalna wersja:
#python plotter_bambu_ver.py obraz.webp --printer bambu/marlin
