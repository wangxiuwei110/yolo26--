import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_dir: Path, exclude_glob: str = ""):
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if exclude_glob:
        files = [p for p in files if not p.match(exclude_glob)]
    return files


def fit_with_padding(img: np.ndarray, cell_w: int, cell_h: int, bg=(20, 20, 20)):
    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    canvas = np.full((cell_h, cell_w, 3), bg, dtype=np.uint8)
    x = (cell_w - new_w) // 2
    y = (cell_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="result", help="directory with result images")
    parser.add_argument("--output", type=str, default="result/grid.jpg", help="output grid image path")
    parser.add_argument("--cols", type=int, default=4, help="number of columns in the grid")
    parser.add_argument("--cell_w", type=int, default=480, help="cell width")
    parser.add_argument("--cell_h", type=int, default=320, help="cell height")
    parser.add_argument("--gap", type=int, default=12, help="gap between cells")
    parser.add_argument("--title_h", type=int, default=30, help="title bar height per cell")
    parser.add_argument("--show_name", action="store_true", help="show file name on each cell")
    parser.add_argument("--exclude_glob", type=str, default="grid*.jpg", help="exclude files matching this glob")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = collect_images(input_dir, exclude_glob=args.exclude_glob)
    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    cols = max(1, args.cols)
    rows = math.ceil(len(files) / cols)
    title_h = args.title_h if args.show_name else 0

    grid_w = cols * args.cell_w + (cols + 1) * args.gap
    grid_h = rows * (args.cell_h + title_h) + (rows + 1) * args.gap
    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)

    for i, img_path in enumerate(files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        r = i // cols
        c = i % cols
        x = args.gap + c * (args.cell_w + args.gap)
        y = args.gap + r * (args.cell_h + title_h + args.gap)

        cell = fit_with_padding(img, args.cell_w, args.cell_h)
        grid[y : y + args.cell_h, x : x + args.cell_w] = cell

        if args.show_name:
            text = img_path.name
            text_y = y + args.cell_h + 21
            cv2.putText(grid, text, (x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"saved grid: {out_path} | images={len(files)} | layout={rows}x{cols}")


if __name__ == "__main__":
    main()
