"""
Qualitative viewer for ASMK retrieval CSV exports (top-k per query, default 5).

**Changing how many results exist in the CSV (max k you can pick here):** in
``main.py``, set ``PipelineConfig.results_topk`` before running the pipeline.

**Changing full ranking depth for mAP / internal search:** ``ASMKConfig.topk``
(None = full DB; int = truncated search). ``export_retrieval_results`` always
clips to ``results_topk`` rows per query.

Run from the project directory (after syncing deps)::

    uv sync
    uv run python vis.py

Optional arguments::

    uv run python vis.py --csv content/shi_asmk_outputs/retrieval_results.csv \\
        --top-k 10 \\
        --metadata content/shi_asmk_outputs/valid_metadata.csv \\
        --oxford-gt-dir content/oxford_gt_files --paris-gt-dir content/paris_gt_files \\
        --host 0.0.0.0 --server_port 7860

PDF exports are written under ``./results/`` (override with ``--results-dir``).

Add ``--share`` to create a temporary public Gradio link.
"""

from __future__ import annotations

import argparse
import importlib
import math
import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio viewer for retrieval_results.csv (top-k).")
    parser.add_argument(
        "--csv",
        type=str,
        default="content/shi_asmk_outputs/retrieval_results.csv",
        help="Path to retrieval_results.csv (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="content/shi_asmk_outputs/valid_metadata.csv",
        help="Dataset metadata CSV with columns dataset, img_name, img_path (matching the retrieval run).",
    )
    parser.add_argument(
        "--oxford-gt-dir",
        type=str,
        default="content/oxford_gt_files",
        help="Oxford landmark ground-truth directory (*_query.txt with bbox lines).",
    )
    parser.add_argument(
        "--paris-gt-dir",
        type=str,
        default="content/paris_gt_files",
        help="Paris ground-truth directory (*_query.txt with bbox lines).",
    )
    parser.add_argument(
        "--query-max-side",
        type=int,
        default=900,
        help="Downscale query preview so max(width,height) does not exceed this (keeps bbox correct before resize).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio bind host.")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a temporary Gradio share URL.",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=12,
        help="Border width in pixels around each thumbnail.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Initial number of retrieved images to show by rank (≤ max rank in CSV); UI can change this.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory under the project root for PDF figure exports.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_arg: str) -> Path:
    p = Path(csv_arg)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def resolve_project_path(path_arg: str) -> Path:
    p = Path(path_arg)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def import_main():
    """Load pipeline module so we reuse GroundTruth loaders (same semantics as retrieval)."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("main")


def load_ground_truth_list(metadata_df: pd.DataFrame, oxford_gt: Path, paris_gt: Path) -> tuple[list[object], str | None]:
    try:
        m = import_main()
        cfg = m.GroundTruthConfig(oxford_gt_dir=oxford_gt, paris_gt_dir=paris_gt)
        queries = m.load_ground_truth_queries(metadata_df, cfg)
        return queries, None
    except Exception as exc:
        return [], f"*Could not load ground truth (query ROI unavailable): `{exc}`*"


def basename_from_path(img_name: str) -> str:
    return Path(img_name).stem


def query_placeholder_image(message: str) -> Image.Image:
    im = Image.new("RGB", (720, 400), (40, 40, 42))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    wrapped = "\n".join([message[i : i + 80] for i in range(0, min(len(message), 400), 80)])
    draw.multiline_text((24, 150), wrapped or "(no message)", fill=(210, 210, 215), font=font)
    return im


def render_query_roi(
    query_id: int,
    queries: list[object],
    metadata_df: pd.DataFrame,
    max_side: int | None = 900,
    border_px: int = 3,
) -> tuple[Image.Image, str]:
    """Draw GT bounding box on the query image."""
    if not queries:
        return query_placeholder_image("Ground-truth query list not loaded."), "*Query ROI unavailable.*"
    if query_id < 0 or query_id >= len(queries):
        return query_placeholder_image("Invalid query index."), "*Invalid query id.*"

    query = queries[query_id]

    basename_to_path: dict[tuple[str, str], str] = {}
    required_cols = {"dataset", "img_name", "img_path"}
    if metadata_df.empty or not required_cols <= set(metadata_df.columns):
        return (
            query_placeholder_image("Metadata missing or empty."),
            "*Need metadata CSV columns: dataset, img_name, img_path.*",
        )

    for row in metadata_df.itertuples(index=False):
        dataset = str(getattr(row, "dataset"))
        img_name = str(getattr(row, "img_name"))
        img_path = str(getattr(row, "img_path"))
        basename_to_path[(dataset, basename_from_path(img_name))] = img_path

    dataset = getattr(query, "dataset")
    qbase = getattr(query, "query_basename")
    key = (dataset, qbase)
    if key not in basename_to_path:
        return query_placeholder_image(
            f"No metadata match for {dataset}/{qbase}"
        ), f"*No image in metadata for query `{qbase}` (dataset `{dataset}`).*"

    rel = Path(basename_to_path[key])
    path = rel if rel.is_absolute() else PROJECT_ROOT / rel
    if not path.is_file():
        return query_placeholder_image(f"Missing file:\n{path.name}"), f"*Query image missing on disk:* `{path}`"

    bbox = getattr(query, "bbox")
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    img = Image.open(path).convert("RGB")
    w, h = img.size

    xl = max(0, min(int(round(x1)), w - 1))
    yt = max(0, min(int(round(y1)), h - 1))
    xr = max(0, min(int(round(x2)), w - 1))
    yb = max(0, min(int(round(y2)), h - 1))

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(xl, yt), (xr, yb)], outline=(0, 255, 255), width=max(3, border_px))

    qh, qw = overlay.size
    if max_side is None:
        scale = 1.0
    else:
        scale = min(1.0, float(max_side) / max(qh, qw))
    if scale < 1.0:
        nw, nh = int(qw * scale), int(qh * scale)
        overlay = overlay.resize((nw, nh), Image.Resampling.LANCZOS)

    caption = (
        f"**Query** · **{dataset}** · `{getattr(query, 'query_name')}` · `{qbase}`\n\n"
        f"GT ROI (pixels, full-res image): `{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}`"
    )
    return overlay, caption


def relevance_style(is_positive: bool, is_junk: bool) -> tuple[tuple[int, int, int], str]:
    if is_positive:
        return (34, 197, 94), "positive"
    if is_junk:
        return (251, 191, 36), "junk"
    return (239, 68, 68), "not relevant"


def pad_with_border(image: Image.Image, border: int, rgb: tuple[int, int, int]) -> Image.Image:
    w, h = image.size
    out = Image.new("RGB", (w + 2 * border, h + 2 * border), rgb)
    out.paste(image, (border, border))
    return out


def rounded_outer_frame(
    padded: Image.Image,
    rgb: tuple[int, int, int],
    border: int,
) -> Image.Image:
    """Rounded stroke on outer edge (plan: rounded rectangular border cue)."""
    draw = ImageDraw.Draw(padded)
    w, h = padded.size
    radius = max(12, border)
    inset = max(1, border // 4)
    stroke = max(2, min(8, border // 2))
    draw.rounded_rectangle(
        (inset, inset, w - 1 - inset, h - 1 - inset),
        radius=radius,
        outline=rgb,
        width=stroke,
    )
    return padded


def placeholder_missing(path: Path, border: int) -> Image.Image:
    w, h = 320, 240
    im = Image.new("RGB", (w, h), (60, 60, 60))
    draw = ImageDraw.Draw(im)
    text = f"Missing:\n{path.name}"
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.multiline_text((10, 10), text, fill=(220, 220, 220), font=font)
    red = (239, 68, 68)
    return pad_with_border(im, border, red)


def load_thumbnail(path: Path, border: int, rgb: tuple[int, int, int], max_side: int = 512) -> Image.Image:
    if not path.is_file():
        return rounded_outer_frame(placeholder_missing(path, border), rgb, border)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return rounded_outer_frame(pad_with_border(im, border, rgb), rgb, border)


def query_label(row: pd.Series) -> str:
    return f"[{int(row['query_id'])}] {row['landmark']} — {row['query_name']}"


def parse_query_id_from_choice(choice: str) -> int:
    m = re.match(r"^\[(\d+)\]", choice.strip())
    if not m:
        raise ValueError(f"Could not parse query_id from dropdown value: {choice!r}")
    return int(m.group(1))


def build_query_choices(df: pd.DataFrame) -> list[str]:
    meta = (
        df[["query_id", "query_name", "landmark"]]
        .drop_duplicates(subset=["query_id"])
        .sort_values("query_id")
    )
    return [query_label(row) for _, row in meta.iterrows()]


def top_k_for_query(df: pd.DataFrame, query_id: int, k: int) -> pd.DataFrame:
    if k < 1:
        k = 1
    sub = df.loc[df["query_id"] == query_id].copy()
    sub = sub.loc[sub["rank"] <= k]
    sub = sub.sort_values("rank").drop_duplicates(subset=["rank"], keep="first").head(k)
    return sub.reset_index(drop=True)


def summarize_top_k(slice_df: pd.DataFrame) -> str:
    n_pos = int(slice_df["is_positive"].sum())
    n_junk = int((~slice_df["is_positive"] & slice_df["is_junk"]).sum())
    n_fp = int((~slice_df["is_positive"] & ~slice_df["is_junk"]).sum())
    n = len(slice_df)
    return (
        f"**Top-{n} summary:** {n} images — "
        f"{n_pos} positive, {n_junk} junk, {n_fp} not relevant (false positives)."
    )


def gallery_for_choice(
    df: pd.DataFrame, choice: str, border: int, k: int
) -> tuple[list[tuple[Image.Image, str]], str]:
    query_id = parse_query_id_from_choice(choice)
    rows = top_k_for_query(df, query_id, k)
    if rows.empty:
        return [], "**No rows** for this query (check k vs CSV ranks)."

    pairs: list[tuple[Image.Image, str]] = []
    for _, row in rows.iterrows():
        rel_path = Path(str(row["retrieved_path"]))
        abs_path = rel_path if rel_path.is_absolute() else PROJECT_ROOT / rel_path
        color, tag = relevance_style(bool(row["is_positive"]), bool(row["is_junk"]))
        thumb = load_thumbnail(abs_path, border, color)
        missing_note = "" if abs_path.is_file() else " | **MISSING FILE**"
        cap = (
            f"#{int(row['rank'])} | score={float(row['score']):.4f} | "
            f"{row['landmark']} | {row['retrieved_basename']} | **{tag}**{missing_note}"
        )
        pairs.append((thumb, cap))

    return pairs, summarize_top_k(rows)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_")[:100] or "query"


def _configure_matplotlib_paper() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )


def export_ranking_pdf(
    *,
    df: pd.DataFrame,
    choice: str,
    top_k: int,
    gt_queries: list[object],
    metadata_df: pd.DataFrame,
    results_dir: Path,
    thumbnail_max_side_pdf: int = 420,
) -> Path:
    """Single-page PDF: query with GT ROI on top, retrieved images in a grid below."""
    _configure_matplotlib_paper()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    q_id = parse_query_id_from_choice(choice)
    if top_k < 1:
        top_k = 1
    rows = top_k_for_query(df, q_id, top_k)
    if rows.empty:
        raise ValueError("No retrieval rows for this query and top-k.")

    query_row_meta = df.loc[df["query_id"] == q_id].iloc[0]
    query_display_name = str(query_row_meta["query_name"])

    q_pil, _ = render_query_roi(
        q_id,
        gt_queries,
        metadata_df,
        max_side=1600,
    )

    nc = min(5, len(rows))
    nr = math.ceil(len(rows) / nc)
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
    fig.suptitle(
        f"{query_display_name} — top-{len(rows)} retrievals",
        fontsize=12,
        weight="medium",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.42, 0.56],
        hspace=0.18,
        top=0.93,
        bottom=0.04,
        left=0.06,
        right=0.94,
    )

    ax_q = fig.add_subplot(gs[0])
    ax_q.imshow(np.asarray(q_pil))
    ax_q.set_axis_off()
    ax_q.set_title("Query (GT region of interest)", loc="center", pad=6)

    inner = gridspec.GridSpecFromSubplotSpec(
        nr,
        nc,
        subplot_spec=gs[1],
        wspace=0.12,
        hspace=0.35,
    )

    for i, (_, row) in enumerate(rows.iterrows()):
        ri, ci = divmod(i, nc)
        ax = fig.add_subplot(inner[ri, ci])
        rel_path = Path(str(row["retrieved_path"]))
        abs_path = rel_path if rel_path.is_absolute() else PROJECT_ROOT / rel_path
        rgb, tag = relevance_style(bool(row["is_positive"]), bool(row["is_junk"]))
        pil_r = load_thumbnail(abs_path, border=6, rgb=rgb, max_side=thumbnail_max_side_pdf)
        ax.imshow(np.asarray(pil_r))
        ax.set_axis_off()
        ax.set_title(
            rf"#{int(row['rank'])} ({tag})\n{row['retrieved_basename']}",
            fontsize=8,
            color="0.05",
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = (
        results_dir / f"retrieval_q{_slug(q_id)}_{_slug(query_display_name)}_top{len(rows)}.pdf"
    )
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return outfile.resolve()


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["query_id"] = pd.to_numeric(df["query_id"], errors="raise").astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise")

    def _csv_bool(series: pd.Series) -> pd.Series:
        """Robust booleans whether CSV inferred bool or strings."""
        if series.dtype == bool:
            return series
        lowered = series.astype(str).str.lower().str.strip()
        return lowered.isin(("true", "1", "yes", "y"))

    df["is_positive"] = _csv_bool(df["is_positive"])
    df["is_junk"] = _csv_bool(df["is_junk"])
    required = {
        "query_id",
        "query_name",
        "landmark",
        "rank",
        "retrieved_path",
        "retrieved_basename",
        "score",
        "is_positive",
        "is_junk",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    choices = build_query_choices(df)
    if not choices:
        raise ValueError(f"No queries found in {csv_path}")

    default_choice = choices[0]
    max_rank_in_csv = max(1, int(df["rank"].max()))
    initial_k = max(1, min(int(args.top_k), max_rank_in_csv))
    results_dir = resolve_project_path(args.results_dir)

    metadata_path = resolve_project_path(args.metadata)
    oxford_gt = resolve_project_path(args.oxford_gt_dir)
    paris_gt = resolve_project_path(args.paris_gt_dir)

    warnings_md: list[str] = []
    if not metadata_path.is_file():
        metadata_df = pd.DataFrame()
        gt_queries: list[object] = []
        warnings_md.append(
            f"*Metadata CSV not found at `{metadata_path}` — query ROI visualization is disabled.*"
        )
    else:
        metadata_df = pd.read_csv(metadata_path)
        meta_need = {"dataset", "img_name", "img_path"}
        if not meta_need <= set(metadata_df.columns):
            raise ValueError(
                f"Metadata CSV missing columns {sorted(meta_need - set(metadata_df.columns))}"
            )
        gt_queries, gt_err = load_ground_truth_list(metadata_df, oxford_gt, paris_gt)
        if gt_err:
            warnings_md.append(gt_err)

    banner = "\n\n".join(warnings_md) if warnings_md else ""

    intro_lines = [
        "## Retrieval qualitative view",
        f"CSV: `{csv_path}` ({len(choices)} queries; ranks **1-{max_rank_in_csv}** per query in this file). "
        "**Green** = positive, **amber** = junk, **red** = not relevant.",
        f"*PDF exports go to `{results_dir}` (see button below).*",
        f"Metadata: `{metadata_path}` · Oxford GT: `{oxford_gt}` · Paris GT: `{paris_gt}`",
    ]
    if banner:
        intro_lines.append(banner)
    intro_text = "\n\n".join(intro_lines)

    with gr.Blocks(title="Retrieval viewer") as demo:

        gr.Markdown(intro_text)

        with gr.Row():
            with gr.Column(scale=3):
                dropdown = gr.Dropdown(
                    choices=choices,
                    value=default_choice,
                    label="Query",
                )
            with gr.Column(scale=2):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=max_rank_in_csv,
                    value=initial_k,
                    step=1,
                    label=(
                        f"Top-k (CSV has up to rank {max_rank_in_csv}; "
                        "raise `PipelineConfig.results_topk` in main.py to export more rows)"
                    ),
                )

        with gr.Row():
            with gr.Column(scale=1, min_width=24):
                pass
            with gr.Column(scale=6):
                query_caption = gr.Markdown()
                query_image = gr.Image(
                    type="pil",
                    label="Query with GT bounding box",
                    show_label=True,
                )
            with gr.Column(scale=1, min_width=24):
                pass

        summary = gr.Markdown()
        gallery = gr.Gallery(
            label="Top-k retrieved images",
            columns=5,
            height="auto",
            object_fit="contain",
            show_label=True,
        )

        with gr.Row():
            export_btn = gr.Button("Export current query + top-k as PDF")
            export_status = gr.Markdown("")
            export_file = gr.File(label="Download PDF")

        def _clamp_k(raw: float | int) -> int:
            return max(1, min(max_rank_in_csv, int(round(float(raw)))))

        def _update(choice: str, k_raw: float):
            k = _clamp_k(k_raw)
            qid = parse_query_id_from_choice(choice)
            q_pil, q_md = render_query_roi(qid, gt_queries, metadata_df, args.query_max_side)
            imgs, md5 = gallery_for_choice(df, choice, args.border, k)
            return q_md, q_pil, md5, imgs

        def _export(choice: str, k_raw: float):
            k = _clamp_k(k_raw)
            try:
                path = export_ranking_pdf(
                    df=df,
                    choice=choice,
                    top_k=k,
                    gt_queries=gt_queries,
                    metadata_df=metadata_df,
                    results_dir=results_dir,
                )
                return (
                    f"Saved **{path.name}** to `{results_dir.resolve()}`",
                    str(path),
                )
            except Exception as exc:
                return f"**Export failed:** `{exc}`", None

        dropdown.change(
            _update,
            inputs=[dropdown, top_k_slider],
            outputs=[query_caption, query_image, summary, gallery],
        )
        top_k_slider.change(
            _update,
            inputs=[dropdown, top_k_slider],
            outputs=[query_caption, query_image, summary, gallery],
        )
        demo.load(
            _update,
            inputs=[dropdown, top_k_slider],
            outputs=[query_caption, query_image, summary, gallery],
        )

        export_btn.click(
            _export,
            inputs=[dropdown, top_k_slider],
            outputs=[export_status, export_file],
        )

    demo.launch(server_name=args.host, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
