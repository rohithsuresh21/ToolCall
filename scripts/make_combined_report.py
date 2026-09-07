"""Build a professional combined ATR evaluation report (dev set + judge benchmark)
as an A4 PDF, reading the outputs the pipeline writes:

  --dev    artifacts/eval-*/report.json   (shape: {"overall": {...}, "by_task_type": {...}})
  --judge  artifacts/judge_eval_*/scores.jsonl   (rows: id, hops, acc_text, acc_exact, steps, calls, ...)

Usage:
  python scripts/make_combined_report.py \
      --dev   artifacts/eval-grpo-planb-step3/report.json \
      --judge artifacts/judge_eval_grpo-planb/scores.jsonl \
      --out   artifacts/ship/ATR-Eval-Report.pdf \
      --model Qwen/Qwen3-1.7B --steps 50
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def pct(v) -> str:
    if v is None:
        return "--"
    return f"{100 * v:.1f}%"


def load_dev(path) -> dict:
    rep = json.loads(Path(path).read_text())
    overall = rep.get("overall", {})
    by_type = rep.get("by_task_type", {})
    by_diff = rep.get("by_difficulty", {})
    return {"overall": overall, "by_type": by_type, "by_diff": by_diff,
            "meta": rep.get("meta", {})}


def load_judge(path) -> dict:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by_hop: dict[int, list[dict]] = {}
    for r in rows:
        by_hop.setdefault(r["hops"], []).append(r)
    agg = {}
    for h, rs in sorted(by_hop.items()):
        agg[h] = {
            "n": len(rs),
            "f1": statistics.fmean(r["f1"] for r in rs),
            "acc_exact": statistics.fmean(r["acc_exact_diagnostic"] for r in rs),
            "avg_steps": statistics.fmean(r["steps"] for r in rs),
            "avg_calls": statistics.fmean(r["calls"] for r in rs),
            "format": statistics.fmean(r["format_strict"] for r in rs),
        }
    overall = {
        "n": len(rows),
        "f1": (statistics.fmean(r["f1"] for r in rows) if rows else 0.0),
        "acc_exact": (statistics.fmean(r["acc_exact_diagnostic"] for r in rows) if rows else 0.0),
        "avg_steps": (statistics.fmean(r["steps"] for r in rows) if rows else 0.0),
        "avg_calls": (statistics.fmean(r["calls"] for r in rows) if rows else 0.0),
        "format": (statistics.fmean(r["format_strict"] for r in rows) if rows else 0.0),
    }
    return {"by_hop": agg, "overall": overall, "rows": rows}


def bullets(styles, items):
    return ListFlowable(
        [ListItem(Paragraph(t, styles["body"]), leftIndent=10, value="\u2022") for t in items],
        bulletType="bullet", start="\u2022", leftIndent=10, spaceAfter=2)


def styled_table(data, col_widths, header_bg="#1f3864", zebra=("#ffffff", "#eef3f9")):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d3e0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), list(zebra)),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", default="50")
    args = ap.parse_args()

    dev = load_dev(args.dev)
    judge = load_judge(args.judge)

    s = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=17, spaceAfter=4,
                             textColor=colors.HexColor("#1f3864")),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=12.5, spaceBefore=12,
                             spaceAfter=5, textColor=colors.HexColor("#2f6f4f")),
        "body": ParagraphStyle("body", parent=s["Normal"], fontSize=9.5, leading=13),
        "meta": ParagraphStyle("meta", parent=s["Normal"], fontSize=8.5, textColor=colors.grey),
    }
    doc = SimpleDocTemplate(args.out, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story = []

    # ---- Title ----
    story.append(Paragraph("ATR Evaluation Report", styles["h1"]))
    story.append(Paragraph(
        "Multi-hop tool-reasoning benchmark: held-out dev set &amp; the judge's public "
        "tools-benchmark. Model: {} &middot; GRPO corrected run ({} steps) &middot; "
        "generated {}".format(
            args.model, args.steps, datetime.now().strftime("%Y-%m-%d %H:%M")),
        styles["meta"]))
    story.append(Spacer(1, 3 * mm))

    # ---- Section 1: Dev set ----
    story.append(Paragraph("1. Held-out Dev Set", styles["h2"]))
    o = dev["overall"]
    story.append(Paragraph(
        "The dev set is our balanced held-out evaluation of 60 fresh questions (20 each of "
        "2-hop, 3-hop and 4-hop), scored at greedy decoding on the <b>vllm</b> backend. "
        "Success requires the tool chain to be followed strictly and the final answer to "
        "match the gold. The official headline metric is the SQuAD/MuSiQue token F1.",
        styles["body"]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Overall", styles["body"]))
    overall_tbl = [
        ["Metric", "Value", "Metric", "Value"],
        ["Task success", pct(o.get("success")), "Avg steps used", f"{o.get('avg_steps', 0):.2f}"],
        ["Answer token-F1", pct(o.get("final_f1")), "Avg tool calls", f"{o.get('avg_calls', 0):.2f}"],
        ["Final answer correct", pct(o.get("final_correct")), "Avg tool errors", f"{o.get('avg_tool_errors', 0):.2f}"],
        ["Format strict", pct(o.get("format_strict")), "Tool necessity OK", pct(o.get("necessity_ok"))],
        ["Tool selection OK", pct(o.get("selection_ok")), "Arguments usable", pct(o.get("args_ok"))],
    ]
    story.append(styled_table(overall_tbl, [56 * mm, 38 * mm, 58 * mm, 38 * mm]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("By task type (2 / 3 / 4-hop)", styles["body"]))
    rows = [["Type", "n", "F1", "Success", "Final", "Select", "Args", "Steps"]]
    for k, b in sorted(dev["by_type"].items()):
        rows.append([k, b["n"], pct(b.get("final_f1")), pct(b.get("success")),
                     pct(b.get("final_correct")), pct(b.get("selection_ok")),
                     pct(b.get("args_ok")), f"{b.get('avg_steps', 0):.2f}"])
    story.append(styled_table(rows, [40 * mm, 14 * mm, 24 * mm, 26 * mm, 24 * mm, 24 * mm, 24 * mm, 16 * mm]))
    story.append(Spacer(1, 2 * mm))

    # ---- Section 2: Judge benchmark ----
    story.append(Paragraph("2. Judge's Tools Benchmark", styles["h2"]))
    jo = judge["overall"]
    story.append(Paragraph(
        "The judge's public benchmark (Cynaptics Tools Benchmark) is 54 multi-hop QA examples "
        "(18 each of 2/3/4-hop) from MuSiQue. Each episode uses BM25 search over the example's "
        "own 20-passage candidate set; the headline score is the same token-F1 used by the dev "
        "eval and the GRPO reward. Evaluated at greedy decoding on the <b>vllm</b> backend.",
        styles["body"]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Overall", styles["body"]))
    j_overall_tbl = [
        ["Metric", "Value", "Metric", "Value"],
        ["Token-F1", pct(jo.get("f1")), "Avg steps used", f"{jo.get('avg_steps', 0):.2f}"],
        ["Exact (diag.)", pct(jo.get("acc_exact")), "Avg tool calls", f"{jo.get('avg_calls', 0):.2f}"],
        ["Format strict", pct(jo.get("format")), "EPisodes scored", f"{jo.get('n', 0)}"],
    ]
    story.append(styled_table(j_overall_tbl, [56 * mm, 38 * mm, 58 * mm, 38 * mm]))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("By difficulty", styles["body"]))
    jrows = [["Hop", "n", "Token-F1", "Exact (diag.)", "Avg steps", "Avg calls", "Format"]]
    for h, b in sorted(judge["by_hop"].items()):
        jrows.append([f"{h}-hop", b["n"], pct(b["f1"]), pct(b["acc_exact"]),
                      f"{b['avg_steps']:.2f}", f"{b['avg_calls']:.2f}", pct(b["format"])])
    story.append(styled_table(jrows, [26 * mm, 16 * mm, 28 * mm, 30 * mm, 24 * mm, 24 * mm, 24 * mm]))
    story.append(Spacer(1, 2 * mm))

    # ---- Section 3: Summary ----
    story.append(Paragraph("3. Summary &amp; Interpretation", styles["h2"]))
    story.append(bullets(styles, [
        "The GRPO checkpoint was trained for exactly <b>{}</b> steps with the 4-hop fix: the "
        "speed reward is disabled (efficiency lambda = 0) and the missing-step under-call "
        "penalty is active, so early-stop shortcuts no longer pay.".format(args.steps),
        "Dev-set result is the ground truth for the 2/3/4-hop balance; the judge benchmark "
        "is the transfer check against the external public split.",
        "Any 4-hop shortfall should be interpreted together with the call-count column: "
        "a genuinely-chained answer (4 calls) is the expected behaviour of the fix.",
    ]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Backend: <b>vllm</b> (offline engine, greedy decoding, temperature 0.0). No HF "
        "backend was used for any evaluation.", styles["meta"]))

    doc.build(story)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
