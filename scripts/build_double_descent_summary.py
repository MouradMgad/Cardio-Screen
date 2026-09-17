from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "work" / "paper_media"
OUTPUT = ROOT / "outputs" / "PCA_Elastic_Net_Double_Descent_Summary.pdf"

GREEN = colors.HexColor("#0A5C3A")
DARK_GREEN = colors.HexColor("#073D29")
BLACK = colors.HexColor("#111714")
YELLOW = colors.HexColor("#F5C63D")
PALE_GREEN = colors.HexColor("#E9F4EE")
PALE_YELLOW = colors.HexColor("#FFF3C4")
MID_GREEN = colors.HexColor("#8EC5A7")
GRAY = colors.HexColor("#59615D")
WHITE = colors.white


def make_styles():
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9, leading=11, tracking=1.5, textColor=YELLOW,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=27, leading=31, textColor=WHITE, alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=12, leading=16, textColor=colors.HexColor("#DCECE4"),
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.8, leading=9.5, textColor=YELLOW,
        ),
        "cover_name": ParagraphStyle(
            "CoverName", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10, leading=12.5, textColor=WHITE,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.3, leading=11, textColor=WHITE,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=BLACK, spaceBefore=0,
            spaceAfter=7, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=GREEN, spaceBefore=7,
            spaceAfter=4, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.45, leading=13.3, textColor=BLACK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.05, leading=10.7, textColor=GRAY, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=7.5, leading=9.6, textColor=GRAY, alignment=TA_CENTER,
            spaceBefore=3, spaceAfter=6,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.4, leading=13.2, textColor=BLACK,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=19, alignment=TA_CENTER, textColor=BLACK,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=GREEN,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.15, leading=9.8, textColor=WHITE,
        ),
        "table_body": ParagraphStyle(
            "TableBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.05, leading=10.35, textColor=BLACK,
        ),
        "phase_title": ParagraphStyle(
            "PhaseTitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10, leading=12, alignment=TA_CENTER, textColor=GREEN,
        ),
        "phase_body": ParagraphStyle(
            "PhaseBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.2, leading=10.8, alignment=TA_CENTER, textColor=BLACK,
        ),
    }


def decorate(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLACK)
    canvas.rect(0, height - 0.34 * cm, width, 0.34 * cm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(0, height - 0.45 * cm, width, 0.11 * cm, fill=1, stroke=0)
    canvas.setStrokeColor(YELLOW)
    canvas.setLineWidth(1.1)
    canvas.line(1.55 * cm, 1.05 * cm, width - 1.55 * cm, 1.05 * cm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.55 * cm, 0.66 * cm, "PCA and Elastic Net vs Double Descent | Paper Summary")
    canvas.drawRightString(width - 1.55 * cm, 0.66 * cm, str(doc.page))
    canvas.restoreState()


def section_banner(text, s):
    t = Table([[Paragraph(text, s["h1"])]], colWidths=[16.7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
        ("BOX", (0, 0), (-1, -1), 0.9, GREEN),
        ("LINEBEFORE", (0, 0), (0, 0), 7, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def callout(text, s, fill=PALE_YELLOW):
    t = Table([[Paragraph(text, s["callout"])]], colWidths=[16.15 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 1.0, YELLOW if fill == PALE_YELLOW else GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def metrics(s, entries):
    cells = []
    for value, label in entries:
        inner = Table([
            [Paragraph(value, s["metric_value"])],
            [Paragraph(label, s["metric_label"])],
        ], colWidths=[3.82 * cm])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("BOX", (0, 0), (-1, -1), 0.8, MID_GREEN),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
        ]))
        cells.append(inner)
    t = Table([cells], colWidths=[4.12 * cm] * len(cells))
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def data_table(s, rows, widths):
    converted = []
    for i, row in enumerate(rows):
        style = s["table_head"] if i == 0 else s["table_body"]
        converted.append([Paragraph(str(v), style) for v in row])
    t = Table(converted, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.45, MID_GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
    ]
    for i in range(1, len(rows)):
        cmds.append(("BACKGROUND", (0, i), (-1, i), PALE_YELLOW if i % 2 else WHITE))
    t.setStyle(TableStyle(cmds))
    return t


def phases(s):
    row = []
    content = [
        ("1. Classical descent", "Increasing complexity reduces bias, so test error falls."),
        ("2. Interpolation peak", "Near parameters = samples, the model can fit noise and test error spikes."),
        ("3. Second descent", "Beyond interpolation, some over-parameterized solutions can generalize better again."),
    ]
    for i, (title, body) in enumerate(content):
        box = Table([
            [Paragraph(title, s["phase_title"])],
            [Paragraph(body, s["phase_body"])],
        ], colWidths=[5.05 * cm])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN if i != 1 else PALE_YELLOW),
            ("BOX", (0, 0), (-1, -1), 1.0, GREEN if i != 1 else YELLOW),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        row.append(box)
    outer = Table([row], colWidths=[5.4 * cm] * 3)
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return outer


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    s = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=1.65 * cm, rightMargin=1.65 * cm,
        topMargin=1.35 * cm, bottomMargin=1.35 * cm,
        title="PCA and Elastic Net vs Double Descent - Paper Summary",
        author="Mourad Gad and Iliyas Shaizada",
        subject="Designed summary of the supplied research paper",
    )
    story = []

    cover = Table([
        [Paragraph("RESEARCH PAPER SUMMARY", s["cover_kicker"])],
        [Paragraph("PCA and Elastic Net<br/>vs Double Descent", s["cover_title"])],
        [Paragraph(
            "How dimensionality reduction and structured regularization change instability in high-degree polynomial regression.",
            s["cover_subtitle"],
        )],
    ], colWidths=[16.7 * cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, YELLOW),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("TOPPADDING", (0, 0), (-1, 0), 14),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
        ("TOPPADDING", (0, 1), (-1, 1), 24),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 23),
    ]))
    story.extend([Spacer(1, 0.6 * cm), cover, Spacer(1, 0.45 * cm)])
    story.append(metrics(s, [
        ("100", "SYNTHETIC SAMPLES"),
        ("1-129", "POLYNOMIAL DEGREES"),
        ("80 / 20", "TRAIN / TEST SPLIT"),
        ("2", "STABILIZATION METHODS"),
    ]))
    story.append(Spacer(1, 0.38 * cm))
    story.append(callout(
        "<b>Central finding.</b> PCA and Elastic Net both control the instability associated with double descent, but they do so differently. PCA removes low-variance directions and can eliminate the sharp test-error spike. Elastic Net shrinks and sparsifies coefficients, producing a heavily smoothed curve that may prevent the model from reaching the overfitting regime.",
        s, PALE_GREEN,
    ))
    story.append(Spacer(1, 0.45 * cm))
    people = Table([
        [Paragraph("AUTHORS", s["cover_label"]), Paragraph("SUPERVISOR", s["cover_label"])],
        [Paragraph("Mourad Gad<br/>Iliyas Shaizada", s["cover_name"]), Paragraph("Dr. Ahmed Elsayed", s["cover_name"])],
        [Paragraph(
            "Menoufia National University, Egypt<br/>Cardiff University Kazakhstan, Kazakhstan",
            s["cover_meta"],
        ), Paragraph(
            "Menoufia National University<br/>Faculty of Engineering, Menoufia University, Egypt<br/>ahmed.elsayed.melegy.nassar@gmail.com",
            s["cover_meta"],
        )],
    ], colWidths=[8.2 * cm, 8.5 * cm])
    people.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLACK),
        ("BOX", (0, 0), (-1, -1), 1.1, YELLOW),
        ("LINEAFTER", (0, 0), (0, -1), 0.7, YELLOW),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(people)

    story.append(PageBreak())
    story.extend([
        section_banner("1. Research problem and core ideas", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The paper investigates double descent: a non-monotonic relationship between model complexity and generalization error. Instead of test error following only the classical U-shaped bias-variance curve, error can rise sharply near the interpolation threshold and then fall again as the model becomes even more over-parameterized.",
            s["body"],
        ),
        phases(s),
        Spacer(1, 0.28 * cm),
        Paragraph("Why the interpolation region becomes unstable", s["h2"]),
        Paragraph(
            "When the number of effective parameters approaches the number of training samples, some eigenvalues of the Gram matrix become very small. The fitted coefficients can then react strongly to small perturbations and noise, producing large variance and a spike in test mean squared error.",
            s["body"],
        ),
        callout(
            "<b>Research question:</b> Can PCA or Elastic Net control the unstable directions and coefficients that produce this behavior in polynomial regression?",
            s,
        ),
        Spacer(1, 0.25 * cm),
        Paragraph("PCA: change the feature space", s["h2"]),
        data_table(s, [
            ["Action", "Mechanism", "Expected effect"],
            ["Center and rotate the polynomial features", "Produces orthogonal principal components ordered by captured variance", "Removes correlation between transformed predictors"],
            ["Retain only leading components", "Drops directions with very small variance", "Reduces effective dimensionality and variance amplification"],
            ["Retain nearly all components", "Keeps unstable low-variance directions", "Repeated or sharp test-error spikes can remain"],
        ], [4.2 * cm, 6.0 * cm, 6.3 * cm]),
        Paragraph("Elastic Net: constrain the coefficients", s["h2"]),
        Paragraph(
            "Elastic Net combines L1 regularization, which promotes sparse coefficients, with L2 regularization, which shrinks coefficients and stabilizes groups of correlated predictors. In this paper it acts more strongly than PCA: rather than merely reshaping the peak, it heavily smooths the regression path and prevents large high-degree error spikes.",
            s["body"],
        ),
        callout(
            "<b>Difference in mechanism:</b> PCA removes unstable directions before regression; Elastic Net keeps the expanded feature space but limits how strongly the fitted model can use it.",
            s, PALE_GREEN,
        ),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("2. Experimental methodology", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "The experiments use a deliberately small synthetic nonlinear regression problem so that model complexity can be increased beyond the number of available samples. This makes the interpolation and over-parameterized regimes visible.",
            s["body"],
        ),
        metrics(s, [
            ("100", "SAMPLES"),
            ("1", "INPUT VARIABLE"),
            ("1-129", "DEGREE RANGE"),
            ("MSE", "EVALUATION METRIC"),
        ]),
        Spacer(1, 0.28 * cm),
        Paragraph("Dataset and target", s["h2"]),
        callout(
            "Inputs are sampled uniformly from -1 to 1. The target is <b>y = sin(2*pi*X) + noise</b>, where the added Gaussian noise has standard deviation 1. A fixed random seed is used for an 80% training and 20% test split.",
            s,
        ),
        Paragraph("Compared configurations", s["h2"]),
        data_table(s, [
            ["Configuration", "Exact setting", "Purpose"],
            ["Polynomial regression baseline", "Degrees 1 through 129; regression on the full polynomial basis", "Reveal instability as effective parameter count increases"],
            ["PCA plus regression", "Several retained-variance levels, ranging from almost 100% down to 95%", "Measure how discarding low-variance directions changes the curve"],
            ["Elastic Net", "alpha = 0.02; L1 ratios 0.2, 0.4, 0.5, 0.7, and 0.9", "Compare different mixtures of sparsity and L2 shrinkage"],
        ], [4.3 * cm, 6.6 * cm, 5.6 * cm]),
        Paragraph("Data journey", s["h2"]),
    ])
    pipeline = Table([
        [Paragraph("Generate noisy sine data", s["phase_title"]), Paragraph("Create polynomial features", s["phase_title"]), Paragraph("Apply PCA or Elastic Net", s["phase_title"]), Paragraph("Record train/test MSE", s["phase_title"])],
        [Paragraph("100 samples<br/>one input variable", s["phase_body"]), Paragraph("Degree 1 to 129", s["phase_body"]), Paragraph("Vary retained variance<br/>or L1 ratio", s["phase_body"]), Paragraph("Plot error against degree", s["phase_body"])],
    ], colWidths=[4.1 * cm] * 4)
    pipeline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE_GREEN),
        ("BACKGROUND", (0, 1), (-1, 1), PALE_YELLOW),
        ("GRID", (0, 0), (-1, -1), 0.8, GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        pipeline,
        Spacer(1, 0.3 * cm),
        Paragraph("Evaluation", s["h2"]),
        Paragraph(
            "For every polynomial degree, training and test mean squared error are calculated. MSE averages the squared difference between observed and predicted targets, so large unstable predictions are penalized heavily. The implementation uses Python, NumPy, scikit-learn, and Matplotlib in a CPU-only environment.",
            s["body"],
        ),
        callout(
            "This design makes the comparison interpretable: polynomial degree changes model complexity, PCA changes effective dimensionality, Elastic Net changes coefficient freedom, and train/test MSE shows the resulting stability or instability.",
            s, PALE_GREEN,
        ),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("3. PCA findings", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "PCA changes the double-descent curve by controlling the effective number and geometry of the polynomial features. The paper's strongest contrast is between retaining almost every principal component and retaining only enough components to explain 95% of the variance.",
            s["body"],
        ),
        Image(str(MEDIA / "image6.jpeg"), width=12.5 * cm, height=7.79 * cm),
        Paragraph(
            "Figure 2 from the paper. With nearly 100% retained variance, test MSE becomes extremely unstable at high polynomial degrees, while training error sometimes approaches zero.",
            s["caption"],
        ),
        Image(str(MEDIA / "image8.png"), width=12.5 * cm, height=7.71 * cm),
        Paragraph(
            "Figure 4 from the paper. At 95% retained variance, both curves remain smooth; the sharp high-degree test-error spikes disappear.",
            s["caption"],
        ),
        Paragraph("Interpretation", s["h2"]),
        data_table(s, [
            ["Observation", "Interpretation"],
            ["Nearly full variance retention preserves repeated spikes", "Orthogonalization alone does not guarantee stability when low-variance directions remain"],
            ["Lower retention reduces the effective parameter count", "The model moves away from the most unstable near-singular regime"],
            ["The 95% curve becomes nearly flat", "Discarding low-variance components suppresses variance amplification, but also limits the complexity the model can express"],
        ], [7.1 * cm, 9.4 * cm]),
    ])

    story.append(PageBreak())
    story.extend([
        section_banner("4. Elastic Net findings and overall conclusion", s),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Elastic Net produces a different form of stabilization. With alpha fixed at 0.02, the combined L1 and L2 penalty keeps the fitted polynomial from developing large coefficients as degree increases. Train and test MSE therefore change only slightly after the first few degrees.",
            s["body"],
        ),
    ])
    elastic_figures = Table([
        [Image(str(MEDIA / "image9.png"), width=8.0 * cm, height=4.94 * cm), Image(str(MEDIA / "image10.png"), width=7.0 * cm, height=5.49 * cm)],
        [Paragraph("Figure 5. Elastic Net train and test MSE remain smooth across degrees.", s["caption"]), Paragraph("Figure 6. Degree-129 fits for five L1 ratios are similar and strongly regularized.", s["caption"])],
    ], colWidths=[8.2 * cm, 8.2 * cm])
    elastic_figures.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([
        elastic_figures,
        Paragraph("PCA and Elastic Net compared", s["h2"]),
        data_table(s, [
            ["Method", "What it controls", "Effect reported in the paper", "Main trade-off"],
            ["PCA", "Number and variance of retained feature directions", "Progressively removes spikes; at 95% retention the curve is stable", "Too much reduction may prevent interpolation and remove useful detail"],
            ["Elastic Net", "Magnitude and sparsity of fitted coefficients", "Heavily smooths high-degree regression across all tested L1 ratios", "Regularization may be strong enough to prevent the double-descent regime entirely"],
        ], [2.8 * cm, 4.2 * cm, 5.2 * cm, 4.3 * cm]),
        Paragraph("Overall conclusion", s["h2"]),
        callout(
            "The paper concludes that low-variance directions are a major source of instability in this synthetic polynomial-regression setting. PCA can eliminate the sharp interpolation spike by discarding those directions. Elastic Net reaches stability through coefficient shrinkage and sparsity, producing a smoother but more strongly constrained model.",
            s, PALE_GREEN,
        ),
        Paragraph("Scope and limitations", s["h2"]),
        Paragraph(
            "The evidence comes from one synthetic dataset with 100 samples, one 80/20 split, one noise level, and a single Elastic Net alpha. The results demonstrate mechanisms clearly, but they should not yet be generalized to large real-world datasets without repeated seeds, confidence intervals, wider hyperparameter searches, and additional datasets.",
            s["body"],
        ),
        Paragraph("Reference foundation", s["h2"]),
        Paragraph(
            "The paper builds on work by Gedon, Ribeiro, and Schon on principal-component regression and double descent; Nakkiran et al. on deep double descent; Zou and Hastie on Elastic Net; Bach on random projections; Pezeshki et al. on multi-scale feature learning; and Yilmaz and Heckel on regularization-wise double descent.",
            s["small"],
        ),
    ])

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == "__main__":
    build()
