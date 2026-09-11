#!/usr/bin/env python3
"""Generate a shareable PDF comparison report from evaluation result files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "Kimi K2.5": ("results/kimi-k2.5_d2b77b49.json", "Bedrock (moonshotai.kimi-k2.5)"),
    "Claude Opus 5": ("results/claude-opus-5_6dfacb94.json", "Bedrock (us.anthropic.claude-opus-5)"),
    "Claude Sonnet 5": ("results/claude-sonnet-5_67dca2c5.json", "Bedrock (us.anthropic.claude-sonnet-5)"),
    "GPT-4o-mini": ("results/gpt-4o-mini_61892f2b.json", "OpenAI API"),
}

NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#EAF0F6")


def load(path):
    return json.load(open(_ROOT / path))


def pct(n, d):
    return f"{100*n/d:.0f}%" if d else "-"


def main():
    stats = {}
    percat = {}
    perdiff = {}
    for name, (path, _) in MODELS.items():
        d = load(path)
        tot = len(d)
        corr = sum(1 for r in d if r["final_score"] >= 0.5)
        exact = sum(1 for r in d if r["exact_match"])
        lat = sum(r["latency_ms"] for r in d) / tot
        stats[name] = {"acc": 100 * corr / tot, "corr": corr, "tot": tot,
                       "exact": 100 * exact / tot, "lat": lat}
        c = defaultdict(lambda: [0, 0]); dd = defaultdict(lambda: [0, 0])
        for r in d:
            ok = r["final_score"] >= 0.5
            c[r["category"]][0] += ok; c[r["category"]][1] += 1
            dd[r["difficulty"]][0] += ok; dd[r["difficulty"]][1] += 1
        percat[name] = c; perdiff[name] = dd

    model_names = list(MODELS.keys())

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=NAVY, fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(
        str(_ROOT / "results" / "ELE_Model_Comparison.pdf"),
        pagesize=letter, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    story = []

    story.append(Paragraph("Enterprise's Last Exam (ELE)", h1))
    story.append(Paragraph("Model Comparison Report", styles["Heading2"]))
    story.append(Paragraph(
        "Benchmark of organizational-reasoning ability across 161 enterprise scenarios "
        "(156 community submissions + 5 reference scenarios). Each scenario tests whether a "
        "model can reason over fragmented, contradictory enterprise context (CRM, Slack, "
        "email, policy documents, billing systems) and reach the decision an experienced "
        "operator would make.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Judge: GPT-4o-mini (reference-based, fixed across all models). "
                           "Date: July 2026.", small))

    # Overall leaderboard
    story.append(Paragraph("Overall Results", h2))
    rows = [["Rank", "Model", "Access", "Accuracy", "Correct", "Avg Latency"]]
    for i, name in enumerate(sorted(model_names, key=lambda m: -stats[m]["acc"]), 1):
        s = stats[name]
        rows.append([str(i), name, MODELS[name][1].split("(")[0].strip(),
                     f"{s['acc']:.1f}%", f"{s['corr']}/{s['tot']}", f"{s['lat']:.0f} ms"])
    t = Table(rows, colWidths=[0.5*inch, 1.4*inch, 1.5*inch, 0.9*inch, 0.9*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # By difficulty
    story.append(Paragraph("Accuracy by Difficulty", h2))
    diffs = ["hard", "expert"]
    rows = [["Difficulty"] + model_names]
    for diff in diffs:
        row = [diff.capitalize()]
        for m in model_names:
            c, tt = perdiff[m][diff]
            row.append(pct(c, tt))
        rows.append(row)
    t = Table(rows, colWidths=[1.3*inch] + [1.35*inch]*len(model_names))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # By category
    story.append(Paragraph("Accuracy by Reasoning Category", h2))
    cats = sorted({c for m in percat for c in percat[m]})
    rows = [["Category"] + model_names]
    for cat in cats:
        row = [cat.replace("_", " ").title()]
        for m in model_names:
            c, tt = percat[m][cat]
            row.append(pct(c, tt))
        rows.append(row)
    t = Table(rows, colWidths=[1.9*inch] + [1.15*inch]*len(model_names))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # Key findings
    story.append(Paragraph("Key Findings", h2))
    findings = [
        "<b>Kimi K2.5 leads overall (87.0%)</b>, narrowly ahead of Claude Opus 5 (85.7%) "
        "and Claude Sonnet 5 (84.5%). GPT-4o-mini trails at 77.0%.",
        "<b>Kimi is strongest on the hardest tier</b> — 93% on expert scenarios, well above "
        "the Claude models (82–85%). It also leads cross-system synthesis (90%).",
        "<b>Claude Opus 5 leads on approval-chain and entity-resolution</b> reasoning "
        "(96% / 93%), reflecting strong structured-authorization logic.",
        "<b>Precedent-based exception handling is the universal weak spot</b> (76% for the top "
        "three, 65% for GPT-4o-mini) — the category requiring judgment about when to bend a rule.",
        "<b>Latency vs accuracy:</b> Kimi delivered top accuracy at ~1.4s/scenario, while the "
        "Claude models were 3–4x slower. GPT-4o-mini was fastest but least accurate.",
    ]
    for f in findings:
        story.append(Paragraph("• " + f, body))
        story.append(Spacer(1, 3))

    # Methodology
    story.append(Paragraph("Methodology", h2))
    story.append(Paragraph(
        "Scoring pipeline: exact match &rarr; LLM-as-a-judge (for non-exact answers). "
        "Multiple-choice answers are scored by exact letter match (markdown formatting such as "
        "<b>**A**</b> is normalized before extraction). The judge (GPT-4o-mini) grades free-text "
        "answers against a held-out correct answer and rationale that are never shown to the model "
        "under test. All models received identical prompts with no hints. A scenario counts as "
        "correct when its final score is &ge; 0.5.", body))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Note: an earlier scoring pass under-counted Claude Opus 5 because it wrapped "
        "multiple-choice answers in markdown bold; the extractor has been corrected and all "
        "results re-scored uniformly.", small))

    doc.build(story)
    print(f"PDF written to: {_ROOT / 'results' / 'ELE_Model_Comparison.pdf'}")


if __name__ == "__main__":
    main()
