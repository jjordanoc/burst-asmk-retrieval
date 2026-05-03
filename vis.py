"""
Qualitative viewer for ASMK retrieval CSV exports (top-5 per query).

Run from the project directory (after syncing deps):

    uv sync
    uv run python vis.py

Optional arguments::

    uv run python vis.py --csv content/shi_asmk_outputs/retrieval_results.csv \\
        --host 0.0.0.0 --server_port 7860

Add ``--share`` to create a temporary public Gradio link.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio viewer for retrieval_results.csv (top-5).")
    parser.add_argument(
        "--csv",
        type=str,
        default="content/shi_asmk_outputs/retrieval_results.csv",
        help="Path to retrieval_results.csv (relative to project root unless absolute).",
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
    return parser.parse_args()


def resolve_csv_path(csv_arg: str) -> Path:
    p = Path(csv_arg)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


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


def top5_for_query(df: pd.DataFrame, query_id: int) -> pd.DataFrame:
    sub = df.loc[df["query_id"] == query_id].copy()
    sub = sub.loc[sub["rank"] <= 5]
    sub = sub.sort_values("rank").drop_duplicates(subset=["rank"], keep="first").head(5)
    return sub.reset_index(drop=True)


def summarize_top5(slice_df: pd.DataFrame) -> str:
    n_pos = int(slice_df["is_positive"].sum())
    n_junk = int((~slice_df["is_positive"] & slice_df["is_junk"]).sum())
    n_fp = int((~slice_df["is_positive"] & ~slice_df["is_junk"]).sum())
    return (
        f"**Top-5 summary:** {len(slice_df)} images — "
        f"{n_pos} positive, {n_junk} junk, {n_fp} not relevant (false positives)."
    )


def gallery_for_choice(df: pd.DataFrame, choice: str, border: int) -> tuple[list[tuple[Image.Image, str]], str]:
    query_id = parse_query_id_from_choice(choice)
    rows = top5_for_query(df, query_id)
    if rows.empty:
        return [], "**No rows** for this query."

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

    return pairs, summarize_top5(rows)


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

    with gr.Blocks(title="Retrieval top-5") as demo:

        gr.Markdown(
            "## Retrieval qualitative view\n"
            f"CSV: `{csv_path}` ({len(choices)} queries). "
            "**Green** = positive, **amber** = junk, **red** = not relevant."
        )

        dropdown = gr.Dropdown(
            choices=choices,
            value=default_choice,
            label="Query",
        )
        summary = gr.Markdown()
        gallery = gr.Gallery(
            label="Top-5 retrievals",
            columns=5,
            rows=1,
            height="auto",
            object_fit="contain",
        )

        def _update(choice: str):
            imgs, md = gallery_for_choice(df, choice, args.border)
            return md, imgs

        dropdown.change(_update, inputs=[dropdown], outputs=[summary, gallery])
        demo.load(_update, inputs=[dropdown], outputs=[summary, gallery])

    demo.launch(server_name=args.host, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
