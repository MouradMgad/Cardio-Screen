from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import split_indices
from heart_sound.features import bandpass, load_audio

MODEL_DIR = ROOT / "outputs" / "model_final"
REPORT = ROOT / "outputs" / "CardioScreen_Project_Report.pdf"
CHARTS = ROOT / "work" / "report_charts_bee"
CHARTS.mkdir(parents=True, exist_ok=True)

BLACK = colors.HexColor("#171717")
YELLOW = colors.HexColor("#F4C430")
PALE = colors.HexColor("#FFF6CF")
GRAY = colors.HexColor("#5C5C5C")
WHITE = colors.white


def probability(artifact, frame):
    weighted = sum(artifact["weights"][name] * model.predict_proba(frame)[:, 1]
                   for name, model in artifact["models"].items())
    return weighted / sum(artifact["weights"].values())


def make_charts(df, y_test, probabilities, selective, predictions):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    counts = df["label"].value_counts().sort_index()
    ax.bar(["Normal", "Abnormal"], counts.values, color=["#171717", "#F4C430"])
    for i, value in enumerate(counts.values):
        ax.text(i, value + 35, f"{value:,}", ha="center", weight="bold")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Dataset class distribution")
    ax.set_ylim(0, max(counts.values) * 1.14)
    fig.tight_layout()
    fig.savefig(CHARTS / "class_distribution.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].plot(fpr, tpr, color="#171717", lw=2.4)
    axes[0].plot([0, 1], [0, 1], "--", color="#A0A0A0")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1].plot(recall, precision, color="#E0AA00", lw=2.4)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(CHARTS / "roc_pr.png", dpi=180)
    plt.close(fig)

    low = selective["validation"]["low_threshold"]
    high = selective["validation"]["high_threshold"]
    accepted = (probabilities <= low) | (probabilities >= high)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ConfusionMatrixDisplay.from_predictions(
        y_test[accepted], probabilities[accepted] >= high,
        display_labels=["Normal", "Abnormal"], colorbar=False, cmap="YlOrBr", ax=ax
    )
    ax.set_title("Final test confusion matrix")
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion.png", dpi=180)
    plt.close(fig)

    examples = predictions.set_index("record").loc[["e01432", "e01623"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6))
    for row_index, (record, row) in enumerate(examples.iterrows()):
        audio, sr = load_audio(row["path"])
        shown = audio[:5 * sr]
        times = np.arange(len(shown)) / sr
        axes[row_index, 0].plot(times, shown, color="#171717" if row_index == 0 else "#D8A500", lw=0.65)
        label = "Normal" if not bool(row["label"]) else "Abnormal"
        axes[row_index, 0].set(title=f"Record {record} - reference: {label}",
                               xlabel="Time (seconds)", ylabel="Amplitude", xlim=(0, 5))
        filtered = bandpass(audio, sr)[:5 * sr]
        mel = librosa.feature.melspectrogram(y=filtered, sr=sr, n_fft=256, hop_length=64,
                                             n_mels=48, fmin=20, fmax=800)
        db = librosa.power_to_db(mel, ref=np.max)
        image = axes[row_index, 1].imshow(db, origin="lower", aspect="auto", cmap="inferno",
                                         extent=[0, 5, 20, 800])
        axes[row_index, 1].set(title="Log-mel spectrogram", xlabel="Time (seconds)",
                               ylabel="Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(CHARTS / "recording_examples.png", dpi=180)
    plt.close(fig)

    detection_records = predictions.set_index("record").loc[["e01432", "a0287", "e01623"]]
    fig, ax = plt.subplots(figsize=(9.4, 3.0))
    ax.axvspan(0, 0.215, color="#D9D9D9", alpha=0.9)
    ax.axvspan(0.215, 0.825, color="#FFF2B3", alpha=0.95)
    ax.axvspan(0.825, 1.0, color="#F4C430", alpha=0.95)
    positions = [2, 1, 0]
    colors_list = ["#171717", "#A87C00", "#D39E00"]
    for position, color, (record, row) in zip(positions, colors_list, detection_records.iterrows()):
        p = float(row["probability_abnormal"])
        ax.scatter([p], [position], s=100, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(p, position + 0.18, f"{record}: {p:.3f}", ha="center", fontsize=9, weight="bold")
    ax.axvline(0.215, color="#444444", ls="--", lw=1.2)
    ax.axvline(0.825, color="#8A6500", ls="--", lw=1.2)
    ax.text(0.105, 2.55, "Likely normal", ha="center", weight="bold")
    ax.text(0.52, 2.55, "Uncertain", ha="center", weight="bold")
    ax.text(0.91, 2.55, "Abnormal", ha="center", weight="bold")
    ax.set(xlim=(0, 1), ylim=(-0.45, 2.8), xlabel="Predicted probability of an abnormal recording")
    ax.set_yticks(positions, ["Normal example", "Uncertain example", "Abnormal example"])
    ax.grid(axis="x", alpha=0.35); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CHARTS / "detection_examples.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, w, h, title, detail, fill="#FFF6CF", edge="#171717"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                               facecolor=fill, edgecolor=edge, linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=8.2)
        return (x, y, w, h)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.4, color="#6D5200"))

    box(0.05, 0.84, 0.22, 0.10, "1. Open dataset", "3,240 labeled WAV files", "#F4C430")
    box(0.39, 0.84, 0.22, 0.10, "2. Audit", "labels, files, six sources")
    box(0.73, 0.84, 0.22, 0.10, "3. Stratified split", "source + class, seed 42")
    arrow((0.27, 0.89), (0.39, 0.89)); arrow((0.61, 0.89), (0.73, 0.89))

    box(0.05, 0.63, 0.24, 0.11, "Training set", "2,268 records\n1,802 N / 466 A", "#EFEFEF")
    box(0.38, 0.63, 0.24, 0.11, "Validation set", "486 records\n386 N / 100 A", "#FFF2B3")
    box(0.71, 0.63, 0.24, 0.11, "Test set", "486 records\n387 N / 99 A", "#F4C430")
    # Route the split through one horizontal branch so the three arrows remain
    # easy to follow instead of crossing diagonally over the boxes.
    ax.plot([0.84, 0.84], [0.84, 0.79], color="#6D5200", lw=1.4)
    ax.plot([0.17, 0.84], [0.79, 0.79], color="#6D5200", lw=1.4)
    arrow((0.17, 0.79), (0.17, 0.74))
    arrow((0.50, 0.79), (0.50, 0.74))
    arrow((0.83, 0.79), (0.83, 0.74))

    box(0.05, 0.41, 0.24, 0.11, "4. Preprocess", "mono, 2 kHz, normalize,\n20-800 Hz filter")
    box(0.38, 0.41, 0.24, 0.11, "5. Extract features", "301 numeric features\nper recording")
    box(0.71, 0.41, 0.24, 0.11, "6. Train candidates", "SVM, Extra Trees,\nLightGBM")
    arrow((0.17, 0.63), (0.17, 0.52)); arrow((0.29, 0.465), (0.38, 0.465)); arrow((0.62, 0.465), (0.71, 0.465))

    box(0.18, 0.20, 0.27, 0.11, "7. Validation selection", "choose 3:1 ensemble,\nbinary and uncertainty limits")
    box(0.55, 0.20, 0.27, 0.11, "8. Locked test evaluation", "metrics, curves,\nconfusion matrices")
    # Validation controls model selection. Candidate training joins that route
    # from the right, while the untouched test set travels directly to the
    # locked evaluation box.
    ax.plot([0.50, 0.50], [0.63, 0.57], color="#6D5200", lw=1.4)
    ax.plot([0.32, 0.50], [0.57, 0.57], color="#6D5200", lw=1.4)
    arrow((0.32, 0.57), (0.32, 0.31))
    ax.plot([0.83, 0.83], [0.41, 0.36], color="#6D5200", lw=1.4)
    ax.plot([0.32, 0.83], [0.36, 0.36], color="#6D5200", lw=1.4)
    arrow((0.32, 0.36), (0.32, 0.31))
    arrow((0.45, 0.255), (0.55, 0.255))
    ax.plot([0.83, 0.83], [0.63, 0.35], color="#6D5200", lw=1.4)
    arrow((0.83, 0.35), (0.69, 0.31))

    box(0.28, 0.02, 0.44, 0.10, "9. Application output",
        "new WAV -> same preprocessing -> probability -> likely normal / uncertain / abnormal", "#F4C430")
    arrow((0.69, 0.20), (0.50, 0.12))
    fig.tight_layout()
    fig.savefig(CHARTS / "pipeline_diagram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def page_decoration(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLACK)
    canvas.rect(0, height - 0.42 * cm, width, 0.42 * cm, fill=1, stroke=0)
    canvas.setFillColor(YELLOW)
    canvas.rect(0, height - 0.50 * cm, width, 0.08 * cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.7 * cm, 0.72 * cm, "Heart Sound Classification Project")
    # Keep the page label comfortably inside the printable area in PDF
    # renderers that clip glyphs close to the right edge.
    canvas.drawRightString(width - 2.2 * cm, 0.72 * cm, f"Page {doc.page}")
    canvas.restoreState()


def metric_table(rows, widths):
    header_style = ParagraphStyle("TableHeader", fontName="Helvetica-Bold", fontSize=8.5,
                                  leading=10.5, textColor=YELLOW)
    body_style = ParagraphStyle("TableBody", fontName="Helvetica", fontSize=8.4,
                                leading=11, textColor=BLACK)
    wrapped = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else body_style
        wrapped.append([cell if isinstance(cell, Paragraph) else Paragraph(str(cell), style)
                        for cell in row])
    table = Table(wrapped, colWidths=widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLACK),
        ("TEXTCOLOR", (0, 0), (-1, 0), YELLOW),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8A45A")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    model_metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    selective = json.loads((MODEL_DIR / "selective_metrics.json").read_text())
    artifact = joblib.load(MODEL_DIR / "heart_sound_model.joblib")
    df = pd.read_csv(ROOT / "work" / "features_v2.csv")
    metadata = ["record", "path", "label", "source"]
    feature_names = [column for column in df.columns if column not in metadata]
    X = df[feature_names].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).to_numpy()
    _, _, test_idx = split_indices(df)
    test_probability = probability(artifact, X.iloc[test_idx])
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    make_charts(df, y[test_idx], test_probability, selective, predictions)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BeeTitle", parent=styles["Title"], textColor=BLACK,
                              fontName="Helvetica-Bold", fontSize=24, leading=29, spaceAfter=10))
    styles.add(ParagraphStyle(name="BeeSubtitle", parent=styles["BodyText"], textColor=GRAY,
                              fontSize=11.5, leading=16, spaceAfter=7))
    styles.add(ParagraphStyle(name="BeeName", parent=styles["BodyText"], textColor=BLACK,
                              fontName="Helvetica-Bold", fontSize=12, leading=16, spaceAfter=5))
    styles.add(ParagraphStyle(name="BeeH1", parent=styles["Heading1"], textColor=BLACK,
                              backColor=YELLOW, borderPadding=7, fontSize=16, leading=20,
                              spaceBefore=10, spaceAfter=9))
    styles.add(ParagraphStyle(name="BeeH2", parent=styles["Heading2"], textColor=BLACK,
                              fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="BeeBody", parent=styles["BodyText"], textColor=BLACK,
                              fontSize=9.7, leading=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="BeeNote", parent=styles["BodyText"], textColor=BLACK,
                              backColor=PALE, borderColor=YELLOW, borderWidth=1, borderPadding=9,
                              fontSize=9.5, leading=14, spaceBefore=6, spaceAfter=10))
    styles.add(ParagraphStyle(name="BeeSmall", parent=styles["BodyText"], textColor=GRAY,
                              fontSize=9.2, leading=14, spaceAfter=9))

    doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=1.65 * cm, leftMargin=1.65 * cm,
                            topMargin=1.35 * cm, bottomMargin=1.35 * cm,
                            title="Heart Sound Classification Project - Mourad Mohamed Gad",
                            author="Mourad Mohamed Gad")

    base = model_metrics["test"]
    final = selective["test"]
    candidates = model_metrics["candidate_validation"]
    story = [
        Spacer(1, 0.8 * cm),
        Paragraph("Heart Sound Classification from Digital Stethoscope Recordings", styles["BeeTitle"]),
        Paragraph("Prepared by Mourad Mohamed Gad", styles["BeeName"]),
        Paragraph("Project report - August 2026", styles["BeeSubtitle"]),
        Spacer(1, 0.35 * cm),
        Paragraph("Project overview", styles["BeeH1"]),
        Paragraph("This project uses recorded heart sounds to classify a recording as normal or abnormal. When the model is not confident enough, it returns an uncertain result instead of forcing a prediction.", styles["BeeBody"]),
        Paragraph("Main test results", styles["BeeH2"]),
        metric_table([
            ["Result type", "Precision", "Recall", "Specificity", "Coverage"],
            ["Binary result", pct(base["precision"]), pct(base["recall_sensitivity"]), pct(base["specificity"]), "100.0%"],
            ["Final three-way result", pct(final["precision"]), pct(final["recall_sensitivity"]), pct(final["specificity"]), pct(final["coverage"])],
        ], [4.2 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm, 2.5 * cm]),
        Paragraph("Why two result types are shown", styles["BeeH2"]),
        metric_table([
            ["Result type", "What it means"],
            [Paragraph("<b>Binary result</b>", styles["BeeSmall"]),
             Paragraph("The model must choose normal or abnormal for every recording. This is useful for comparing the model with a standard two-class test, but it can force a choice on difficult recordings.", styles["BeeSmall"])],
            [Paragraph("<b>Final three-way result</b>", styles["BeeSmall"]),
             Paragraph("The model can answer likely normal, abnormal, or uncertain. The uncertain result is used when the probability is between the two confidence limits. This is the version used by the application.", styles["BeeSmall"])],
        ], [4.0 * cm, 11.5 * cm]),
        Spacer(1, 0.3 * cm),
        Paragraph(f"The final result is calculated only for recordings that received a clear normal or abnormal prediction. The model gave a clear answer for {pct(final['coverage'])} of the test recordings. The other {pct(1 - final['coverage'])} were marked uncertain.", styles["BeeNote"]),
        Image(str(CHARTS / "class_distribution.png"), width=15.5 * cm, height=7.9 * cm),
        PageBreak(),
        Paragraph("Complete data journey", styles["BeeH1"]),
        Paragraph("The diagram follows the data from the downloaded WAV files to the final application result. Training, validation, and testing have different jobs and are kept as separate parts of the workflow.", styles["BeeBody"]),
        Image(str(CHARTS / "pipeline_diagram.png"), width=16.2 * cm, height=11.3 * cm),
        Paragraph("Training path", styles["BeeH2"]),
        Paragraph("The training recordings are preprocessed, converted into 301 numeric features, and used to fit five candidate models. The training labels are visible to those models.", styles["BeeBody"]),
        Paragraph("Validation path", styles["BeeH2"]),
        Paragraph("The validation recordings are not used to fit the candidate model parameters. They are used to compare candidates, choose the ensemble weights, and choose the probability limits.", styles["BeeBody"]),
        Paragraph("Test and application paths", styles["BeeH2"]),
        Paragraph("The test set measures the selected system. In the application, a new WAV file follows the same preprocessing and feature extraction before the saved models calculate its abnormality probability.", styles["BeeBody"]),
        PageBreak(),
        Paragraph("The dataset", styles["BeeH1"]),
        Paragraph("The project uses the public PhysioNet/Computing in Cardiology Challenge 2016 heart sound dataset. The recordings were collected by several research groups in clinical and non-clinical environments. They include both children and adults and contain real recording noise such as breathing, speech, movement, and stethoscope contact noise.", styles["BeeBody"]),
        Paragraph("Where the data came from", styles["BeeH2"]),
        Paragraph("Dataset page: https://physionet.org/content/challenge-2016/1.0.0/<br/>Dataset license: Open Data Commons Attribution License v1.0<br/>Downloaded file used in this project: training.zip", styles["BeeNote"]),
        Paragraph("Data used", styles["BeeH2"]),
        metric_table([
            ["Item", "Value"], ["Total labeled recordings", "3,240"],
            ["Normal recordings", "2,575 (79.5%)"], ["Abnormal recordings", "665 (20.5%)"],
            ["Source databases", "6"], ["Audio format", "WAV"],
            ["Standard sample rate", "2,000 Hz"], ["Training / validation / test", "70% / 15% / 15%"],
        ], [7.0 * cm, 8.5 * cm]),
        Spacer(1, 0.25 * cm),
        Paragraph("What the labels mean", styles["BeeH2"]),
        Paragraph("Normal means that the recording was labeled as a normal heart sound. Abnormal means that the patient had a confirmed cardiac abnormality. The dataset does not provide a separate disease name for every recording, so this project predicts normal or abnormal rather than a specific disease.", styles["BeeBody"]),
        Paragraph("How the recordings were prepared", styles["BeeH2"]),
        Paragraph("Each WAV file was converted to one audio channel and resampled to 2,000 Hz. A 20-800 Hz filter was applied to keep the main heart-sound frequency range. The audio level was normalized. The program then calculated MFCC, spectral, energy, timing, frequency-band, and wavelet features from each recording.", styles["BeeBody"]),
        Paragraph("All 3,240 recordings were processed successfully. The abnormal class was smaller, so class balancing was used during training.", styles["BeeBody"]),
        PageBreak(),
        Paragraph("Data split and preprocessing", styles["BeeH1"]),
        Paragraph("The split was produced with random seed 42. Stratification used both the source database and the class label so that each split retained a similar source and class mixture.", styles["BeeBody"]),
        metric_table([
            ["Split", "Total", "Normal", "Abnormal", "Purpose"],
            ["Training", "2,268", "1,802", "466", "Fit model parameters"],
            ["Validation", "486", "386", "100", "Choose models and thresholds"],
            ["Test", "486", "387", "99", "Measure selected system"],
        ], [3.0 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 5.8 * cm]),
        Paragraph("Preprocessing steps", styles["BeeH2"]),
        metric_table([
            ["Step", "Exact operation", "Reason"],
            ["1. Load", "Read WAV and convert to floating-point audio", "Create a consistent numeric signal"],
            ["2. Mono", "Combine channels into one channel", "The model expects one PCG signal"],
            ["3. Resample", "Convert every file to 2,000 samples per second", "Give every recording the same time scale"],
            ["4. Minimum length", "Pad recordings shorter than one second with zeros", "Avoid invalid short inputs"],
            ["5. Center", "Replace invalid values and subtract the median", "Remove the DC offset"],
            ["6. Normalize", "Divide by the 99.5th percentile absolute amplitude and clip to -1 to 1", "Reduce recording-level gain differences"],
            ["7. Filter", "Fourth-order Butterworth band-pass from 20 to 800 Hz", "Keep the main PCG range and reduce unrelated frequencies"],
            ["8. Frame analysis", "256-sample FFT window with 64-sample hop", "Measure how the signal changes over time"],
        ], [2.5 * cm, 7.2 * cm, 5.8 * cm]),
        Paragraph("No audio augmentation, synthetic samples, or deep-learning image input was used. The spectrograms in this report are visual explanations; the model itself receives the 301 numeric features described next.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("Feature extraction", styles["BeeH1"]),
        Paragraph("Feature extraction converts a variable-length audio recording into one fixed row of 301 numbers. This allows fast CPU models to process recordings of different durations.", styles["BeeBody"]),
        metric_table([
            ["Feature family", "Calculation", "Number"],
            ["MFCC and delta MFCC", "20 MFCCs and 20 first differences, each summarized five ways", "200"],
            ["Basic time/spectral", "RMS, zero-crossing, centroid, bandwidth, roll-off, flatness; five summaries each", "30"],
            ["Spectral contrast", "Five contrast bands with five temporal summaries", "25"],
            ["Recording/amplitude", "Duration, clipping, silence, crest factor, and six amplitude percentiles", "10"],
            ["Physiological frequency", "Five band powers, spectral entropy, envelope periodicity, heart-rate proxy", "8"],
            ["Wavelet", "Seven db4 coefficient groups; energy, standard deviation, skew, and kurtosis", "28"],
            ["Total", "One fixed feature vector for each recording", "301"],
        ], [4.1 * cm, 9.6 * cm, 1.8 * cm]),
        Paragraph("Temporal summaries", styles["BeeH2"]),
        Paragraph("For each frame-based feature, the code calculates the mean, standard deviation, median, 10th percentile, and 90th percentile across the complete recording. These summaries keep information about the typical value, spread, and low/high behavior without requiring every recording to have the same duration.", styles["BeeBody"]),
        Paragraph("MFCC and spectral features", styles["BeeH2"]),
        Paragraph("MFCCs describe the short-term spectral shape of the sound. Delta MFCCs describe how that shape changes. Spectral centroid, bandwidth, roll-off, flatness, and contrast describe where frequency energy is located and how concentrated it is.", styles["BeeBody"]),
        Paragraph("Periodicity and wavelets", styles["BeeH2"]),
        Paragraph("Welch power spectral density is used for five bands: 20-60, 60-120, 120-200, 200-400, and 400-800 Hz. A Hilbert-envelope autocorrelation between 0.30 and 1.50 seconds provides periodicity and heart-rate proxy values. A six-level Daubechies-4 wavelet decomposition provides multi-scale energy and shape measurements.", styles["BeeBody"]),
        Paragraph("The feature table was created for all 3,240 recordings with zero extraction failures. The measured extraction time was 79.33 seconds using six CPU workers on this computer.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("What the heart-sound data looks like", styles["BeeH1"]),
        Paragraph("The figure below shows two real recordings from the held-out test set. Only the first five seconds are displayed so the sound pattern is easy to see.", styles["BeeBody"]),
        Image(str(CHARTS / "recording_examples.png"), width=16.2 * cm, height=9.5 * cm),
        Paragraph("Waveform", styles["BeeH2"]),
        Paragraph("The waveform shows how the recorded sound pressure changes with time. Repeating peaks usually follow the heart cycles. The waveform can also contain noise from breathing, movement, or stethoscope contact.", styles["BeeBody"]),
        Paragraph("Spectrogram", styles["BeeH2"]),
        Paragraph("The spectrogram shows the same recording using time on the horizontal axis and frequency on the vertical axis. Brighter areas contain more sound energy. This view helps show when sound energy appears and which frequencies contain it.", styles["BeeBody"]),
        Paragraph("These plots are examples only. A waveform or spectrogram should not be diagnosed by visual appearance alone.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("Models that were tested", styles["BeeH1"]),
        Paragraph("Five CPU model candidates were trained on the same 2,268-record training split. Missing numeric values were filled with the median learned from the training data.", styles["BeeBody"]),
        metric_table([
            ["Candidate", "Main settings", "Validation F1", "Validation AUROC"],
            ["RBF SVM C=3", "Robust scaling; class-balanced", f"{candidates['svm_c3']['f1']:.3f}", f"{candidates['svm_c3']['roc_auc']:.3f}"],
            ["RBF SVM C=10", "Robust scaling; class-balanced", f"{candidates['svm_c10']['f1']:.3f}", f"{candidates['svm_c10']['roc_auc']:.3f}"],
            ["Extra Trees", "900 trees; minimum leaf 1; 45% features; class-balanced", f"{candidates['extra_deep']['f1']:.3f}", f"{candidates['extra_deep']['roc_auc']:.3f}"],
            ["LightGBM balanced", "700 boosting iterations; balanced classes", f"{candidates['lgb_balanced']['f1']:.3f}", f"{candidates['lgb_balanced']['roc_auc']:.3f}"],
            ["LightGBM moderate", "700 iterations; abnormal weight 2.5", f"{candidates['lgb_moderate']['f1']:.3f}", f"{candidates['lgb_moderate']['roc_auc']:.3f}"],
        ], [3.4 * cm, 6.7 * cm, 2.7 * cm, 2.7 * cm]),
        Paragraph("Why these models", styles["BeeH2"]),
        Paragraph("SVM provides a nonlinear decision boundary after robust feature scaling. Extra Trees builds many randomized decision trees and averages them, which handles nonlinear feature interactions well. LightGBM builds trees sequentially so later trees correct errors made by earlier trees.", styles["BeeBody"]),
        Paragraph("The final ensemble", styles["BeeH2"]),
        Paragraph("Validation prediction blends were searched using small integer weights from 0 to 3. The selected blend was three parts Extra Trees and one part moderate LightGBM. Therefore, the final probability is 75% Extra Trees probability plus 25% LightGBM probability.", styles["BeeNote"]),
        metric_table([
            ["Final component", "Exact settings"],
            ["Extra Trees", "900 trees; minimum samples per leaf 1; max_features 0.45; balanced class weights; all CPU cores; random seed 42"],
            ["LightGBM", "700 boosting iterations; learning rate 0.025; 15 leaves; minimum child samples 20; row fraction 0.85; feature fraction 0.75; L2 regularization 2.0; abnormal weight 2.5; six CPU workers; seed 42"],
            ["Probability blend", "(3 x Extra Trees probability + 1 x LightGBM probability) / 4"],
        ], [4.0 * cm, 11.5 * cm]),
        PageBreak(),
        Paragraph("Training process and timing", styles["BeeH1"]),
        Paragraph("This project does not use a neural network, so it does not have epochs. Epochs normally mean repeated full passes through training data in neural-network training. The tree and SVM methods use different stopping units.", styles["BeeNote"]),
        metric_table([
            ["Method", "Equivalent training unit"],
            ["Extra Trees", "900 independent randomized trees are fitted once"],
            ["LightGBM", "700 boosting iterations; one new correction tree per iteration"],
            ["SVM candidates", "The numerical optimizer runs until its convergence condition"],
        ], [4.5 * cm, 11.0 * cm]),
        Paragraph("Training sequence", styles["BeeH2"]),
        Paragraph("1. Fit the median imputer and each candidate using the 2,268 training recordings.<br/>2. Calculate candidate probabilities for the 486 validation recordings.<br/>3. Compare individual models and search integer probability-blend weights.<br/>4. Choose a binary threshold from 0.05 to 0.95 in steps of 0.005 using validation F1 and balanced accuracy.<br/>5. Save the selected models, feature order, blend weights, and threshold.<br/>6. Search lower and upper validation limits for the three-way output.<br/>7. Apply the saved system to the 486 test recordings.", styles["BeeBody"]),
        Paragraph("Measured runtime", styles["BeeH2"]),
        metric_table([
            ["Stage", "Measured time", "Included work"],
            ["Feature extraction", "79.33 seconds", "All 3,240 WAV files, 301 features, six CPU workers"],
            ["Model training and selection", "30.29 seconds", "Five candidate fits, validation predictions, blend search, test prediction, and model save"],
            ["Combined compute stage", "109.62 seconds", "Feature extraction plus model-selection training"],
        ], [5.1 * cm, 3.2 * cm, 7.2 * cm]),
        Paragraph("The timing was measured on the local Windows environment used for this project with eight logical CPU threads and no accessible NVIDIA GPU. Download time, dependency installation, report creation, and human review are not included. Runtime will change on another computer.", styles["BeeBody"]),
        Paragraph("Class imbalance", styles["BeeH2"]),
        Paragraph("Only 20.5% of recordings were abnormal. Class-balanced weights were used for SVM and Extra Trees. The final LightGBM candidate assigned abnormal training examples 2.5 times the weight of normal examples. This helps the model avoid learning to predict the larger normal class too often.", styles["BeeBody"]),
        PageBreak(),
        Paragraph("How the saved application works", styles["BeeH1"]),
        Paragraph("The Flask application runs locally and uses the saved model file. It does not retrain when a user uploads a recording.", styles["BeeBody"]),
        metric_table([
            ["Step", "Application action"],
            ["1. Upload", "Accept a WAV file through the browser form"],
            ["2. Temporary file", "Save the upload temporarily so the audio library can read it"],
            ["3. Preprocess", "Apply the same mono conversion, 2 kHz resampling, normalization, and 20-800 Hz filter used in training"],
            ["4. Features", "Calculate the same 301 features and put them in the saved training-column order"],
            ["5. Component probabilities", "Extra Trees and LightGBM each calculate an abnormality probability"],
            ["6. Ensemble", "Combine probabilities as (3 x Extra Trees + 1 x LightGBM) / 4"],
            ["7. Decision", "Probability at or below 0.215: likely normal; 0.215-0.825: uncertain; at or above 0.825: abnormal"],
            ["8. Display", "Show the final label, abnormality probability, and confidence"],
            ["9. Cleanup", "Delete the temporary uploaded file"],
        ], [3.0 * cm, 12.5 * cm]),
        Paragraph("What is saved", styles["BeeH2"]),
        Paragraph("The model file stores the fitted Extra Trees model, fitted LightGBM model, their 3:1 weights, the binary threshold, the exact 301-feature column order, and a model version. The selective thresholds are stored in selective_metrics.json.", styles["BeeBody"]),
        Paragraph("What the application does not do", styles["BeeH2"]),
        Paragraph("It does not identify a named disease, replace an echocardiogram, provide treatment advice, or prove clinical safety. It produces a normal/uncertain/abnormal research output for a WAV heart-sound recording.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("How the detection looked", styles["BeeH1"]),
        Paragraph("The model converts the features from each recording into an abnormality probability. The two fixed limits then convert that probability into likely normal, uncertain, or abnormal.", styles["BeeBody"]),
        Image(str(CHARTS / "detection_examples.png"), width=16.2 * cm, height=5.2 * cm),
        metric_table([
            ["Record", "Reference label", "Abnormality probability", "Final output"],
            ["e01432", "Normal", "0.003", "Likely normal"],
            ["a0287", "Normal", "0.252", "Uncertain"],
            ["e01623", "Abnormal", "0.977", "Abnormal"],
        ], [3.2 * cm, 3.8 * cm, 4.5 * cm, 4.0 * cm]),
        Spacer(1, 0.3 * cm),
        Paragraph("Example 1: clear normal", styles["BeeH2"]),
        Paragraph("Record e01432 received probability 0.003, which is below the normal limit of 0.215. The final output was likely normal.", styles["BeeBody"]),
        Paragraph("Example 2: uncertain", styles["BeeH2"]),
        Paragraph("Record a0287 received probability 0.252. This is above 0.215 but below 0.825, so the model did not force a normal or abnormal answer. The final output was uncertain.", styles["BeeBody"]),
        Paragraph("Example 3: clear abnormal", styles["BeeH2"]),
        Paragraph("Record e01623 received probability 0.977, which is above the abnormal limit of 0.825. The final output was abnormal.", styles["BeeBody"]),
        Paragraph("The reference label is the dataset label. The final output is the result produced by the trained model.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("Evaluation method", styles["BeeH1"]),
        Paragraph("Every reported metric is calculated at recording level. Abnormal is treated as the positive class and normal as the negative class.", styles["BeeBody"]),
        metric_table([
            ["Term", "Meaning"],
            ["True positive (TP)", "An abnormal recording predicted as abnormal"],
            ["False positive (FP)", "A normal recording predicted as abnormal"],
            ["False negative (FN)", "An abnormal recording predicted as normal"],
            ["True negative (TN)", "A normal recording predicted as normal"],
        ], [5.0 * cm, 10.5 * cm]),
        Paragraph("Metrics", styles["BeeH2"]),
        metric_table([
            ["Metric", "How it is calculated", "What it answers"],
            ["Accuracy", "(TP + TN) / all recordings", "How often was the model correct overall?"],
            ["Balanced accuracy", "Average of abnormal recall and normal recall", "How well did it handle both classes equally?"],
            ["Precision", "TP / (TP + FP)", "When it said abnormal, how often was it right?"],
            ["Recall / sensitivity", "TP / (TP + FN)", "How many abnormal recordings did it find?"],
            ["Specificity", "TN / (TN + FP)", "How many normal recordings did it correctly reject?"],
            ["F1 score", "Harmonic mean of precision and recall", "How balanced were precision and recall?"],
            ["AUROC", "Ranking performance over all binary thresholds", "Did abnormal cases generally receive higher probabilities?"],
            ["Average precision", "Summary of the precision-recall curve", "How good was positive detection across thresholds?"],
            ["Coverage", "Clear normal/abnormal outputs / all inputs", "What fraction avoided an uncertain result?"],
        ], [3.5 * cm, 5.2 * cm, 6.8 * cm]),
        Paragraph("Threshold selection", styles["BeeH2"]),
        Paragraph("The binary threshold 0.295 was selected on validation predictions to maximize F1, with balanced accuracy as the next comparison. The final three-way search examined a lower limit from 0.05 to 0.40 and an upper limit from 0.45 to 0.90. It selected the highest validation coverage for which precision, recall, and specificity were each at least 99%, producing limits 0.215 and 0.825.", styles["BeeBody"]),
        Paragraph("The train/validation/test split prevents direct fitting on the test rows. However, more than one project experiment was reviewed during development, so these figures should be treated as a development benchmark rather than a completely independent clinical validation. Patient identifiers are also not consistently available across all source databases.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("The model and results", styles["BeeH1"]),
        Paragraph("The final CPU model combines Extra Trees and LightGBM. Both are fast models for structured numerical features. The model produces a probability between 0 and 1, where a larger value means that an abnormal recording is more likely.", styles["BeeBody"]),
        Paragraph("Binary test results", styles["BeeH2"]),
        metric_table([
            ["Metric", "Result"], ["Accuracy", pct(base["accuracy"])],
            ["Balanced accuracy", pct(base["balanced_accuracy"])], ["Precision", pct(base["precision"])],
            ["Recall", pct(base["recall_sensitivity"])], ["Specificity", pct(base["specificity"])],
            ["F1 score", pct(base["f1"])], ["AUROC", f"{base['roc_auc']:.3f}"],
            ["Average precision", f"{base['average_precision']:.3f}"],
        ], [8.0 * cm, 4.0 * cm]),
        Paragraph("Binary confusion counts", styles["BeeH2"]),
        metric_table([
            ["True normal", "False abnormal", "Missed abnormal", "True abnormal"],
            ["363", "24", "7", "92"],
        ], [3.9 * cm, 3.9 * cm, 3.9 * cm, 3.9 * cm]),
        Spacer(1, 0.25 * cm),
        Paragraph("The binary model detected most abnormal recordings, but 24 normal recordings were predicted as abnormal. Because abnormal recordings are only 20.5% of the data, these false positives reduce precision more strongly than they reduce overall accuracy. For this reason, the final version uses an uncertain range.", styles["BeeBody"]),
        Image(str(CHARTS / "roc_pr.png"), width=16.0 * cm, height=6.2 * cm),
        PageBreak(),
        Paragraph("Final three-way results", styles["BeeH1"]),
        Paragraph("The final decision rules were fixed using the validation set. A probability at or below 0.215 is reported as likely normal. A probability at or above 0.825 is reported as abnormal. Values between these limits are reported as uncertain.", styles["BeeBody"]),
        Image(str(CHARTS / "confusion.png"), width=11.5 * cm, height=8.4 * cm),
        metric_table([
            ["Measure", "Test result"],
            ["Clear predictions", f"{final['accepted_n']} of {final['accepted_n'] + final['uncertain_count']}"],
            ["Coverage", pct(final["coverage"])], ["Precision", pct(final["precision"])],
            ["Recall", pct(final["recall_sensitivity"])], ["Specificity", pct(final["specificity"])],
            ["Referral sensitivity", pct(final["referral_sensitivity"])],
        ], [8.0 * cm, 4.0 * cm]),
        Spacer(1, 0.25 * cm),
        Paragraph("Among the clear test predictions, 343 normal and 35 abnormal recordings were classified correctly. There were three false positives and one false negative. A further 104 recordings were marked uncertain.", styles["BeeBody"]),
        Paragraph("These results describe this dataset and test split only. The project is a research demonstration and is not a medical diagnosis tool. Patient identifiers are not available consistently across all six sources, so the evaluation is performed at recording level.", styles["BeeNote"]),
        PageBreak(),
        Paragraph("References", styles["BeeH1"]),
        Paragraph("1. Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart sound algorithms. Physiological Measurement. 2016;37(12):2181-2213. https://doi.org/10.1088/0967-3334/37/12/2181", styles["BeeSmall"]),
        Paragraph("2. Clifford GD, Liu C, Moody B, et al. Classification of normal/abnormal heart sound recordings: the PhysioNet/Computing in Cardiology Challenge 2016. Computing in Cardiology. 2016.", styles["BeeSmall"]),
        Paragraph("3. PhysioNet. Classification of Heart Sound Recordings, version 1.0.0. https://physionet.org/content/challenge-2016/1.0.0/", styles["BeeSmall"]),
    ]

    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(REPORT)


if __name__ == "__main__":
    main()
