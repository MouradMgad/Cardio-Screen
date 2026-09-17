from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "CardioScreen_Project_Summary.pdf"
CHARTS = ROOT / "work" / "report_charts_bee"

GREEN = colors.HexColor("#0B5D3B")
DARK_GREEN = colors.HexColor("#073D29")
BLACK = colors.HexColor("#111714")
YELLOW = colors.HexColor("#F5C63D")
PALE_YELLOW = colors.HexColor("#FFF4C7")
PALE_GREEN = colors.HexColor("#EAF5EF")
MID_GREEN = colors.HexColor("#8FC5A8")
WHITE = colors.white
GRAY = colors.HexColor("#58615C")
LIGHT_GRAY = colors.HexColor("#EDF0EE")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SummaryTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=28, leading=32, textColor=WHITE, alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SummarySubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=12.5, leading=17, textColor=colors.HexColor("#DDEDE5"),
            spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.4, leading=13, textColor=WHITE,
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.2, leading=10, textColor=YELLOW,
        ),
        "cover_name": ParagraphStyle(
            "CoverName", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10.2, leading=12.5, textColor=WHITE,
        ),
        "h1": ParagraphStyle(
            "SummaryH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=BLACK, spaceBefore=0,
            spaceAfter=9, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "SummaryH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13.2, leading=16, textColor=GREEN, spaceBefore=7,
            spaceAfter=5, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "SummaryBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.6, leading=13.6, textColor=BLACK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SummarySmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=11.2, textColor=GRAY, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "SummaryCaption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=7.7, leading=10, textColor=GRAY, alignment=TA_CENTER,
            spaceBefore=3, spaceAfter=7,
        ),
        "callout": ParagraphStyle(
            "SummaryCallout", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=BLACK, spaceAfter=0,
        ),
        "card_label": ParagraphStyle(
            "CardLabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.7, leading=9, textColor=GREEN, alignment=TA_CENTER,
        ),
        "card_value": ParagraphStyle(
            "CardValue", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=19, textColor=BLACK, alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.4, leading=10.2, textColor=WHITE,
        ),
        "table_body": ParagraphStyle(
            "TableBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.2, leading=10.5, textColor=BLACK,
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLACK)
    canvas.rect(0, height - 0.34 * cm, width, 0.34 * cm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(0, height - 0.45 * cm, width, 0.11 * cm, fill=1, stroke=0)
    canvas.setStrokeColor(YELLOW)
    canvas.setLineWidth(1.2)
    canvas.line(1.55 * cm, 1.05 * cm, width - 1.55 * cm, 1.05 * cm)
    canvas.setFont("Helvetica", 7.7)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.55 * cm, 0.66 * cm, "CardioScreen | Heart Sound Abnormality Screening")
    canvas.drawRightString(width - 1.55 * cm, 0.66 * cm, f"{doc.page}")
    canvas.restoreState()


def section_banner(text, s):
    table = Table([[Paragraph(text, s["h1"])]], colWidths=[16.7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
        ("BOX", (0, 0), (-1, -1), 1.0, GREEN),
        ("LINEBEFORE", (0, 0), (0, 0), 7, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def metric_cards(s, cards):
    cells = []
    for value, label in cards:
        cells.append([
            Paragraph(value, s["card_value"]),
            Paragraph(label, s["card_label"]),
        ])
    row = []
    for value, label in cards:
        inner = Table(
            [[Paragraph(value, s["card_value"])], [Paragraph(label, s["card_label"])]],
            colWidths=[3.85 * cm],
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.9, MID_GREEN),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ]))
        row.append(inner)
    outer = Table([row], colWidths=[4.12 * cm] * len(row))
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return outer


def styled_table(s, rows, widths, header=True):
    converted = []
    for ridx, row in enumerate(rows):
        style = s["table_header"] if header and ridx == 0 else s["table_body"]
        converted.append([Paragraph(str(value), style) for value in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.45, MID_GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), GREEN))
    start = 1 if header else 0
    for idx in range(start, len(rows)):
        commands.append(("BACKGROUND", (0, idx), (-1, idx), PALE_YELLOW if idx % 2 else WHITE))
    table.setStyle(TableStyle(commands))
    return table


def callout(text, s, color=PALE_YELLOW):
    table = Table([[Paragraph(text, s["callout"])]], colWidths=[16.2 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 1.0, YELLOW if color == PALE_YELLOW else GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    s = styles()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        rightMargin=1.65 * cm, leftMargin=1.65 * cm,
        topMargin=1.35 * cm, bottomMargin=1.35 * cm,
        title="CardioScreen Project Summary",
        author="Mourad Mohamed Gad",
        subject="Heart sound abnormality screening project summary",
    )

    story = []

    cover_box = Table([
        [Paragraph("CARDIOSCREEN", ParagraphStyle(
            "Brand", fontName="Helvetica-Bold", fontSize=10, leading=12,
            textColor=YELLOW, tracking=2,
        ))],
        [Paragraph("Heart Sound<br/>Abnormality Screening", s["title"])],
        [Paragraph(
            "A fast, CPU-based machine-learning system that converts recorded stethoscope audio into a probability of general cardiac abnormality.",
            s["subtitle"],
        )],
    ], colWidths=[16.7 * cm])
    cover_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("LINEBELOW", (0, 0), (-1, 0), 1.3, YELLOW),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("TOPPADDING", (0, 0), (-1, 0), 14),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
        ("TOPPADDING", (0, 1), (-1, 1), 24),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 24),
    ]))
    story.extend([Spacer(1, 0.55 * cm), cover_box, Spacer(1, 0.45 * cm)])
    story.append(metric_cards(s, [
        ("3,240", "OPEN WAV RECORDINGS"),
        ("301", "FEATURES PER RECORDING"),
        ("93.6%", "BINARY ACCURACY"),
        ("109.62 s", "MEASURED CPU PIPELINE"),
    ]))
    story.append(Spacer(1, 0.38 * cm))

    abstract = Table([
        [Paragraph("PROJECT SUMMARY", ParagraphStyle(
            "MiniTitle", fontName="Helvetica-Bold", fontSize=9, leading=11,
            textColor=GREEN,
        ))],
        [Paragraph(
            "CardioScreen was developed as a research-oriented screening project using the open PhysioNet/Computing in Cardiology 2016 heart-sound dataset. Each recording is standardized and represented by 301 audio features. A weighted ensemble of Extra Trees and LightGBM then estimates the probability that the recording belongs to the dataset's abnormal class. The system supports both a conventional binary result and a safer three-way result that can return uncertain when the probability is not decisive.",
            s["body"],
        )],
    ], colWidths=[16.7 * cm])
    abstract.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
        ("BOX", (0, 0), (-1, -1), 0.8, MID_GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
    ]))
    story.append(abstract)
    story.append(Spacer(1, 0.42 * cm))

    identity = Table([
        [Paragraph("PREPARED BY", s["cover_label"]), Paragraph("SUPERVISED BY", s["cover_label"])],
        [Paragraph("Mourad Mohamed Gad", s["cover_name"]), Paragraph("Dr. Ahmed Elsayed", s["cover_name"])],
        [Paragraph("Heart Sound Classification Project", s["cover_meta"]), Paragraph(
            "Menoufia National University<br/>Faculty of Engineering, Menoufia University, Egypt<br/>ahmed.elsayed.melegy.nassar@gmail.com",
            s["cover_meta"],
        )],
    ], colWidths=[8.2 * cm, 8.5 * cm])
    identity.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLACK),
        ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 1.2, YELLOW),
        ("LINEAFTER", (0, 0), (0, -1), 0.8, YELLOW),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(identity)

    story.append(PageBreak())
    story.extend([
        section_banner("1. Data and project workflow", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The project uses the open PhysioNet/Computing in Cardiology Challenge 2016 collection. It contains 3,240 labeled phonocardiogram recordings gathered from six source databases. The target is binary: 2,575 recordings are normal and 665 are abnormal. Because the sources use different devices and acquisition conditions, the split was stratified by both source and class.",
            s["body"],
        ),
        metric_cards(s, [
            ("2,268", "TRAIN RECORDINGS"),
            ("486", "VALIDATION RECORDINGS"),
            ("486", "TEST RECORDINGS"),
            ("20.5%", "ABNORMAL SHARE"),
        ]),
        Spacer(1, 0.25 * cm),
        Image(str(CHARTS / "class_distribution.png"), width=11.8 * cm, height=5.2 * cm),
        Paragraph(
            "Figure 1. Class distribution in the complete labeled dataset. The imbalance is handled through class weighting rather than by generating synthetic heart sounds.",
            s["caption"],
        ),
        Paragraph("Data journey", s["h2"]),
        Image(str(CHARTS / "pipeline_diagram.png"), width=15.8 * cm, height=10.6 * cm),
        Paragraph(
            "Figure 2. End-to-end journey from open WAV recordings to the saved application's three possible outputs.",
            s["caption"],
        ),
        Paragraph(
            "<b>Primary source:</b> Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart sound algorithms. <i>Physiological Measurement</i>. 2016;37(12):2181-2213. Dataset: PhysioNet/Computing in Cardiology Challenge 2016, version 1.0.0.",
            s["small"],
        ),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("2. Signal processing and model", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Every recording follows the same deterministic preparation steps: conversion to mono, resampling to 2 kHz, padding recordings shorter than one second, median centering, robust amplitude normalization, clipping to the range -1 to 1, and a fourth-order Butterworth band-pass filter from 20 to 800 Hz.",
            s["body"],
        ),
        Image(str(CHARTS / "recording_examples.png"), width=16.0 * cm, height=7.5 * cm),
        Paragraph(
            "Figure 3. Real examples from the dataset shown as waveforms and time-frequency spectrograms. Spectrograms explain the audio visually; the model itself receives engineered numerical features.",
            s["caption"],
        ),
        Paragraph("The 301-feature representation", s["h2"]),
        styled_table(s, [
            ["Feature family", "Count", "What it represents"],
            ["MFCC and delta-MFCC summaries", "200", "Compact descriptions of sound texture and how it changes over time"],
            ["Time and spectral statistics", "30", "Amplitude, energy, zero crossings, spectral shape, and related summaries"],
            ["Spectral contrast", "25", "Differences between spectral peaks and valleys"],
            ["Recording and amplitude descriptors", "10", "Duration and robust signal-level properties"],
            ["Frequency, entropy, periodicity, heart-rate proxy", "8", "Band energy, complexity, and repeating cardiac timing"],
            ["Wavelet statistics", "28", "Multi-resolution transient patterns using a six-level db4 decomposition"],
        ], [6.0 * cm, 1.5 * cm, 9.0 * cm]),
        Paragraph("Final ensemble", s["h2"]),
        callout(
            "<b>Final abnormality probability = 75% Extra Trees + 25% LightGBM.</b> Extra Trees uses 900 randomized trees. LightGBM uses 700 boosting iterations. This is traditional machine learning, so the system has trees and boosting iterations rather than neural-network epochs.",
            s,
            PALE_GREEN,
        ),
        Spacer(1, 0.18 * cm),
        Paragraph(
            "Five candidate models were compared on the validation set: two SVM settings, one Extra Trees model, and two LightGBM settings. The ensemble weights and probability limits were selected using validation predictions. The final test set was then used for the reported evaluation.",
            s["body"],
        ),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("3. Predictions and measured performance", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The model first produces a probability between 0 and 1. A larger value means that the recording resembles the dataset's abnormal examples more strongly. The project reports this probability in two ways.",
            s["body"],
        ),
        styled_table(s, [
            ["Output mode", "Decision rule", "Purpose"],
            ["Binary", "Abnormal when probability is at least 0.295", "Forces every test recording into normal or abnormal"],
            ["Three-way", "Likely normal at or below 0.215; abnormal at or above 0.825; otherwise uncertain", "Avoids forcing a confident answer for borderline recordings"],
        ], [3.0 * cm, 7.0 * cm, 6.5 * cm]),
        Spacer(1, 0.25 * cm),
        Image(str(CHARTS / "detection_examples.png"), width=14.5 * cm, height=5.6 * cm),
        Paragraph(
            "Figure 4. Three real test examples: a clear normal recording, a borderline recording marked uncertain, and a clear abnormal recording.",
            s["caption"],
        ),
        Paragraph("Binary test results", s["h2"]),
        metric_cards(s, [
            ("93.6%", "ACCURACY"),
            ("79.3%", "PRECISION"),
            ("92.9%", "RECALL"),
            ("0.975", "AUROC"),
        ]),
        Spacer(1, 0.23 * cm),
        Paragraph(
            "At the binary threshold, the model correctly classified 363 normal and 92 abnormal test recordings. It produced 24 false positives and missed 7 abnormal recordings. Precision is lower than accuracy because abnormal recordings form only about one fifth of the dataset, so false-positive abnormal predictions have a strong effect on precision.",
            s["body"],
        ),
        Image(str(CHARTS / "roc_pr.png"), width=15.4 * cm, height=5.9 * cm),
        Paragraph(
            "Figure 5. ROC and precision-recall curves for the final ensemble on the held-out test set.",
            s["caption"],
        ),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("4. Selective result and research direction", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The three-way version is the preferred presentation because it separates decisive predictions from borderline recordings. Among 486 test recordings, 382 received a clear normal or abnormal result and 104 were marked uncertain.",
            s["body"],
        ),
        metric_cards(s, [
            ("78.6%", "CLEAR-RESULT COVERAGE"),
            ("92.1%", "PRECISION"),
            ("97.2%", "RECALL"),
            ("99.1%", "SPECIFICITY"),
        ]),
        Spacer(1, 0.25 * cm),
        Image(str(CHARTS / "confusion.png"), width=7.3 * cm, height=5.25 * cm),
        Paragraph(
            "Figure 6. Confusion matrix for clear three-way decisions only: 343 true normal, 35 true abnormal, 3 false abnormal, and 1 missed abnormal. The 104 uncertain recordings are excluded from this matrix but included in the 78.6% coverage value.",
            s["caption"],
        ),
        Paragraph("What the system detects", s["h2"]),
        callout(
            "The model detects <b>general abnormal heart-sound patterns</b> represented by the dataset's abnormal label. It does not identify a specific disease and should not be described as diagnosing valve disease, arrhythmia, or another named condition.",
            s,
        ),
        Spacer(1, 0.18 * cm),
        Paragraph("Use on new recordings", s["h2"]),
        Paragraph(
            "The saved application can process an unseen, unlabeled WAV recording because labels are needed only for training and evaluation. However, the reported accuracy belongs to this PhysioNet test split. Performance may change with a different stethoscope, hospital, patient population, noise level, or recording protocol.",
            s["body"],
        ),
        Paragraph("Recommended next research steps", s["h2"]),
        styled_table(s, [
            ["Priority", "Next step", "Why it matters"],
            ["1", "Patient-level grouped cross-validation with confidence intervals", "Provides a stronger estimate and reduces leakage risk"],
            ["2", "External evaluation on a second open heart-sound dataset", "Measures generalization beyond one benchmark"],
            ["3", "Ablation study of feature families and ensemble components", "Shows which parts of the method create the improvement"],
            ["4", "Error analysis by source, noise, and recording quality", "Explains where the model succeeds or fails"],
            ["5", "Clinical review and, if possible, disease-specific labels", "Supports medically meaningful interpretation"],
        ], [1.9 * cm, 6.9 * cm, 7.7 * cm]),
    ])

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
