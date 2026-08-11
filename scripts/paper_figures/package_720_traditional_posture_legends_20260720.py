#!/usr/bin/env python3
"""Package the 7.20 manuscript figures after the legend terminology update.

The script copies the exact figure sources used by the current manuscript into
one delivery directory and replaces the matching embedded PNG files in a Word
copy.  It never modifies the source manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path("/Users/saintyoung/Desktop/FOWT-CONTROLLER-main")
SOURCE_DOCX = Path(
    "/Users/saintyoung/Desktop/小论文/"
    "7.20基于短时风况预测的浮式风机主动压载调节算法.docx"
)
OUT = ROOT / "outputs/paper_figures/20260720_all_legends_traditional_posture"
FIG_OUT = OUT / "figures"
WORD_OUT = OUT / "word"
QA_OUT = OUT / "qa"
OUTPUT_DOCX = WORD_OUT / (
    "7.20基于短时风况预测的浮式风机主动压载调节算法_图例更新.docx"
)


# media file -> (delivery stem, regenerated figure stem)
FIGURES: dict[str, tuple[str, Path]] = {
    "image1.png": (
        "fig1_algorithm_framework",
        ROOT
        / "outputs/paper_figures/20260711_chapter3_redraw/"
        "algorithm_framework_compact_v33/figures/"
        "fig1_algorithm_framework_compact_v33",
    ),
    "image3.png": (
        "fig2_prediction_performance",
        ROOT
        / "outputs/paper_figures/20260720_ocean_engineering_boxed_legends/"
        "figures/fig2_prediction_performance_boxed_legend",
    ),
    "image4.png": (
        "fig3a_cumulative_pump_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_a_cumulative_pump_volume",
    ),
    "image5.png": (
        "fig3b_pump_runtime_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_b_pump_runtime_ratio",
    ),
    "image6.png": (
        "fig3c_attitude_over_2deg_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_c_attitude_over_2deg_ratio",
    ),
    "image7.png": (
        "fig3d_attitude_over_3deg_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_d_attitude_over_3deg_ratio",
    ),
    "image8.png": (
        "fig3e_attitude_over_4deg_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_e_attitude_over_4deg_ratio",
    ),
    "image9.png": (
        "fig3f_attitude_over_5deg_density",
        ROOT
        / "outputs/paper_figures/20260712_ocean_engineering_chapter3/"
        "density6_ocean_engineering_grid72_20260713/figures/"
        "panel_f_attitude_over_5deg_ratio",
    ),
    "image10.png": (
        "fig4a_casewise_cumulative_pump",
        ROOT
        / "outputs/paper_figures/20260711_chapter3_redraw/"
        "population_pump_panels_v38_equal_frames_compact/figures/"
        "fig3a_casewise_cumulative_pump_named",
    ),
    "image11.png": (
        "fig4b_total_cumulative_pump",
        ROOT
        / "outputs/paper_figures/20260720_ocean_engineering_boxed_legends/"
        "figures/fig3b_total_cumulative_pump_boxed_legend",
    ),
    "image12.png": (
        "fig5a_ablation_casewise_pump_ecdf",
        ROOT
        / "outputs/paper_figures/20260720_ocean_engineering_boxed_legends/"
        "figures/fig4a_ablation_casewise_pump_ecdf_boxed_legend",
    ),
    "image13.png": (
        "fig5b_ablation_paired_pump_scatter",
        ROOT
        / "outputs/paper_figures/20260720_ocean_engineering_boxed_legends/"
        "figures/fig4b_ablation_paired_pump_scatter_boxed_legend",
    ),
    "image14.png": (
        "fig6a_rolling_strategy_target",
        ROOT
        / "outputs/paper_figures/20260720_chapter3_3_38_rolling_process_final/"
        "standalone_figures/06_sel6h_38_panel_a_strategy_target",
    ),
    "image15.png": (
        "fig6b_measured_wind",
        ROOT
        / "outputs/paper_figures/20260720_chapter3_3_38_rolling_process_final/"
        "standalone_figures/06_sel6h_38_panel_b_measured_wind",
    ),
    "image16.png": (
        "fig6c_cumulative_pump",
        ROOT
        / "outputs/paper_figures/20260720_chapter3_3_38_rolling_process_final/"
        "standalone_figures/06_sel6h_38_panel_c_cumulative_pump",
    ),
    "image17.png": (
        "fig6d_attitude_response",
        ROOT
        / "outputs/paper_figures/20260720_chapter3_3_38_rolling_process_final/"
        "standalone_figures/06_sel6h_38_panel_d_attitude",
    ),
    "image18.png": (
        "fig7a_rolling_strategy_target",
        ROOT
        / "outputs/paper_figures/20260720_07_posadd6h_099_rolling_process_final/"
        "standalone_figures/07_posadd6h_099_panel_a_strategy_target",
    ),
    "image19.png": (
        "fig7b_measured_wind",
        ROOT
        / "outputs/paper_figures/20260720_07_posadd6h_099_rolling_process_final/"
        "standalone_figures/07_posadd6h_099_panel_b_measured_wind",
    ),
    "image20.png": (
        "fig7c_cumulative_pump",
        ROOT
        / "outputs/paper_figures/20260720_07_posadd6h_099_rolling_process_final/"
        "standalone_figures/07_posadd6h_099_panel_c_cumulative_pump",
    ),
    "image21.png": (
        "fig7d_attitude_response",
        ROOT
        / "outputs/paper_figures/20260720_07_posadd6h_099_rolling_process_final/"
        "standalone_figures/07_posadd6h_099_panel_d_attitude",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_media_dimensions() -> dict[str, tuple[int, int]]:
    dims: dict[str, tuple[int, int]] = {}
    with zipfile.ZipFile(SOURCE_DOCX) as archive:
        for media_name in FIGURES:
            member = f"word/media/{media_name}"
            with archive.open(member) as handle:
                with Image.open(handle) as image:
                    dims[media_name] = image.size
    return dims


def package_figures() -> list[dict[str, object]]:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    old_dims = source_media_dimensions()
    rows: list[dict[str, object]] = []

    for media_name, (delivery_stem, source_stem) in FIGURES.items():
        copied: dict[str, str] = {}
        for suffix in (".png", ".pdf", ".svg"):
            source = source_stem.with_suffix(suffix)
            if not source.exists():
                raise FileNotFoundError(source)
            destination = FIG_OUT / f"{delivery_stem}{suffix}"
            shutil.copy2(source, destination)
            copied[suffix[1:]] = str(destination)

        with Image.open(source_stem.with_suffix(".png")) as image:
            new_dims = image.size
        if new_dims != old_dims[media_name]:
            raise ValueError(
                f"{media_name}: regenerated PNG dimensions {new_dims} "
                f"do not match embedded image dimensions {old_dims[media_name]}"
            )

        rows.append(
            {
                "word_media": media_name,
                "delivery_stem": delivery_stem,
                "source_stem": str(source_stem),
                "width_px": new_dims[0],
                "height_px": new_dims[1],
                "png_sha256": sha256(source_stem.with_suffix(".png")),
                **copied,
            }
        )
    return rows


def build_word_copy() -> None:
    WORD_OUT.mkdir(parents=True, exist_ok=True)
    temp_docx = OUTPUT_DOCX.with_suffix(".tmp.docx")

    with zipfile.ZipFile(SOURCE_DOCX, "r") as source_archive:
        with zipfile.ZipFile(
            temp_docx, "w", compression=zipfile.ZIP_DEFLATED
        ) as target_archive:
            for item in source_archive.infolist():
                data = source_archive.read(item.filename)
                if item.filename.startswith("word/media/"):
                    media_name = Path(item.filename).name
                    if media_name in FIGURES:
                        _, source_stem = FIGURES[media_name]
                        data = source_stem.with_suffix(".png").read_bytes()
                target_archive.writestr(item, data)

    temp_docx.replace(OUTPUT_DOCX)


def write_manifest(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)

    manifest_json = {
        "source_docx": str(SOURCE_DOCX),
        "output_docx": str(OUTPUT_DOCX),
        "scope": "current manuscript figures only",
        "legend_change": "姿态反馈算法 -> 传统姿态反馈算法",
        "figure_count": len(rows),
        "figures": rows,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(SOURCE_DOCX)
    rows = package_figures()
    build_word_copy()
    write_manifest(rows)
    print(f"Packaged {len(rows)} figures in {FIG_OUT}")
    print(f"Created Word copy: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
