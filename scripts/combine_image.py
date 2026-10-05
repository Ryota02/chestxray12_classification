import math
from pathlib import Path
import sys

import yaml
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.utils.config import load_config

def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def collect_runs(input_root, image_pattern):
    input_root = Path(input_root)

    runs = []

    for backbone_dir in sorted(input_root.iterdir()):
        if not backbone_dir.is_dir():
            continue

        for experiment_dir in sorted(backbone_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue

            for seed_dir in sorted(experiment_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                seed_name = seed_dir.name
                if not seed_name.startswith("seed"):
                    continue

                image_paths = sorted(seed_dir.glob(image_pattern))

                if not image_paths:
                    continue

                runs.append(
                    {
                        "backbone": backbone_dir.name,
                        "experiment": experiment_dir.name,
                        "seed": seed_name,
                        "directory": seed_dir,
                        "images": image_paths,
                    }
                )

    return runs


def get_page_size(cfg):
    use_landscape = (
        cfg.get("page", {})
        .get("landscape", True)
    )

    if use_landscape:
        return landscape(A4)

    return portrait(A4)


def draw_header(pdf, text, page_width, page_height, margin_top):
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        20,
        page_height - margin_top + 5,
        text,
    )


def draw_footer(pdf, text, page_width, margin_bottom):
    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(
        page_width - 20,
        margin_bottom - 10,
        text,
    )


def fit_image_to_box(img_width, img_height, box_width, box_height):
    scale = min(box_width / img_width, box_height / img_height)
    draw_width = img_width * scale
    draw_height = img_height * scale
    return draw_width, draw_height


def draw_image_grid_page(
    pdf,
    image_paths,
    backbone,
    experiment,
    seed,
    page_number,
    total_pages,
    page_size,
    cfg,
):
    page_width, page_height = page_size

    page_cfg = cfg.get("page", {})
    grid_cfg = cfg.get("grid", {})

    margin_left = page_cfg.get("margin_left", 20)
    margin_right = page_cfg.get("margin_right", 20)
    margin_top = page_cfg.get("margin_top", 30)
    margin_bottom = page_cfg.get("margin_bottom", 30)

    header_enabled = cfg.get("header", {}).get("enabled", True)
    footer_enabled = cfg.get("footer", {}).get("enabled", True)

    columns = grid_cfg.get("columns", 2)
    rows = grid_cfg.get("rows", 2)
    h_spacing = grid_cfg.get("horizontal_spacing", 10)
    v_spacing = grid_cfg.get("vertical_spacing", 18)
    caption_height = grid_cfg.get("caption_height", 14)

    if header_enabled:
        header_text = (
            f"Backbone: {backbone} | "
            f"Experiment: {experiment} | "
            f"Seed: {seed}"
        )
        draw_header(
            pdf,
            header_text,
            page_width,
            page_height,
            margin_top,
        )

    if footer_enabled:
        footer_text = f"Page {page_number}/{total_pages}"
        draw_footer(
            pdf,
            footer_text,
            page_width,
            margin_bottom,
        )

    content_top = page_height - margin_top - 20
    content_bottom = margin_bottom + 10

    if header_enabled:
        content_top -= 15

    if footer_enabled:
        content_bottom += 10

    available_width = page_width - margin_left - margin_right
    available_height = content_top - content_bottom

    cell_width = (
        available_width - (columns - 1) * h_spacing
    ) / columns

    cell_height = (
        available_height - (rows - 1) * v_spacing
    ) / rows

    image_box_height = cell_height - caption_height

    for idx, image_path in enumerate(image_paths):
        row = idx // columns
        col = idx % columns

        if row >= rows:
            break

        x0 = margin_left + col * (cell_width + h_spacing)
        y_top = content_top - row * (cell_height + v_spacing)
        y0 = y_top - cell_height

        # Caption
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size

            draw_width, draw_height = fit_image_to_box(
                img_width,
                img_height,
                cell_width,
                image_box_height,
            )

            img_x = x0 + (cell_width - draw_width) / 2
            img_y = y0 + (image_box_height - draw_height) / 2

            pdf.drawImage(
                ImageReader(img),
                img_x,
                img_y,
                width=draw_width,
                height=draw_height,
                preserveAspectRatio=True,
                mask="auto",
            )

        except Exception as e:
            pdf.setFont("Helvetica", 8)
            pdf.drawString(
                x0,
                y0 + 10,
                f"[ERROR] {image_path.name}",
            )
            pdf.drawString(
                x0,
                y0,
                str(e),
            )

    pdf.showPage()


def create_run_pdf(run, page_size, cfg):
    backbone = run["backbone"]
    experiment = run["experiment"]
    seed = run["seed"]
    image_paths = run["images"]
    seed_directory = run["directory"]

    experiment_directory = seed_directory.parent
    output_path = experiment_directory / f"{seed}.pdf"

    grid_cfg = cfg.get("grid", {})
    columns = grid_cfg.get("columns", 2)
    rows = grid_cfg.get("rows", 2)
    images_per_page = columns * rows

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=page_size,
    )

    pages = list(chunk_list(image_paths, images_per_page))
    total_pages = len(pages)

    for page_number, page_images in enumerate(pages, start=1):
        print(
            f"    [Page {page_number}/{total_pages}] "
            f"{len(page_images)} images"
        )

        draw_image_grid_page(
            pdf=pdf,
            image_paths=page_images,
            backbone=backbone,
            experiment=experiment,
            seed=seed,
            page_number=page_number,
            total_pages=total_pages,
            page_size=page_size,
            cfg=cfg,
        )

    pdf.save()
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    report_cfg = cfg.get("report", {})

    input_root = report_cfg.get("input_root", "outputs/xai")
    image_pattern = report_cfg.get("image_pattern", "*_xai.png")

    runs = collect_runs(
        input_root=input_root,
        image_pattern=image_pattern,
    )

    if not runs:
        print("[INFO] No XAI image runs found.")
        return

    page_size = get_page_size(report_cfg)

    print(f"[INFO] Found {len(runs)} runs.")

    for run in runs:
        print(
            f"[INFO] Creating PDF: "
            f"{run['backbone']} / {run['experiment']} / {run['seed']}"
        )

        output_path = create_run_pdf(
            run=run,
            page_size=page_size,
            cfg=report_cfg,
        )

        print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()