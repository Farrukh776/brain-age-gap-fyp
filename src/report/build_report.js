// Build the FYP-I report (Word) on the FAST FYP-I Final Report template layout (step 10).
//
// Run from the repo root (needs the npm package `docx`, e.g. NODE_PATH=<folder>/node_modules):
//     python -m src.report.figures && python -m src.report.diagrams
//     node src/report/build_report.js
// Reads figures from docs/figures and outputs/figures, the logo from outputs/report/logo.png.
// Writes outputs/report/FYP-I_Report.docx (local only: it contains subject-level scatter plots).
// Open it in Word and update fields (Ctrl+A, F9) to fill the table of contents and the lists.
// All numbers come from docs/experiment_log.md, docs/decisions.md and docs/cohort_definition.md.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, Header, Footer, PageNumber,
  NumberFormat, LevelFormat, TableOfContents, SequentialIdentifier, TabStopType, VerticalAlign,
  FootnoteReferenceRun,
} = require("docx");

const ROOT = process.cwd();
const OUT = path.join(ROOT, "outputs", "report", "FYP-I_Report.docx");

// ------------------------------------------------------------------ style constants
const FONT = "Arial";
const NAVY = "1F3864";
const GREY = "595959";
const HEAD_FILL = "D9E2F3";
const LABEL_FILL = "F2F2F2";
const BODY_PT = 24;            // half-points: 12 pt
const LINE = 360;              // 1.5 line spacing
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440;   // A4, 1 inch margins
const TEXT_W = PAGE_W - 2 * MARGIN;                     // 9026 DXA

const TITLE = "Brain Age Gap for Predicting Progression from Mild Cognitive Impairment to Alzheimer's Disease";
const SHORT_TITLE = "Brain Age Gap for MCI-to-AD Progression";
const SUPERVISOR = "Dr. Nasir Uddin";
const TEAM = [["Syed Anfas", "23K-0574"], ["Asher Ahmed", "23K-0511"], ["Farrukh Imran", "23K-0776"]];

// ------------------------------------------------------------------ text helpers
// A "rich" item is a string, or an array of strings / {b: "..."} / {i: "..."} / {sup: "..."}.
function runs(rich, base = {}) {
  const parts = Array.isArray(rich) ? rich : [rich];
  return parts.map((p) => {
    if (typeof p === "string") return new TextRun({ ...base, text: p });
    if (p.b !== undefined) return new TextRun({ ...base, text: p.b, bold: true });
    if (p.i !== undefined) return new TextRun({ ...base, text: p.i, italics: true });
    if (p.sup !== undefined) return new TextRun({ ...base, text: p.sup, superScript: true });
    if (p.hl !== undefined) return new TextRun({ ...base, text: p.hl, highlight: "yellow" });
    return p;
  });
}

const P = (rich, opts = {}) => new Paragraph({
  children: runs(rich), alignment: AlignmentType.JUSTIFIED,
  spacing: { line: LINE, after: 120 }, ...opts,
});
const PC = (rich, opts = {}) => P(rich, { alignment: AlignmentType.CENTER, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)], pageBreakBefore: true });
const H1nb = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const bullet = (rich) => new Paragraph({
  children: runs(rich), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 60 },
});
const numbered = (rich, instance = 1) => new Paragraph({
  children: runs(rich), numbering: { reference: "numbers", level: 0, instance },
  alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 60 },
});
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const blank = (after = 0) => new Paragraph({ children: [], spacing: { after } });

// ------------------------------------------------------------------ captions, figures, tables
function caption(label, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, keepLines: true,
    style: "Caption",
    children: [
      new TextRun({ text: `${label} `, italics: true }),
      new SequentialIdentifier(label),
      new TextRun({ text: `: ${text}`, italics: true }),
    ],
  });
}

function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT, keepNext: true, keepLines: true, spacing: { before: 160, after: 80 },
    style: "Caption",
    children: [
      new TextRun({ text: "Table ", italics: true }),
      new SequentialIdentifier("Table"),
      new TextRun({ text: `: ${text}`, italics: true }),
    ],
  });
}

function figure(file, widthIn, text) {
  const buf = fs.readFileSync(path.join(ROOT, file));
  // PNG size from the IHDR chunk
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96);
  const hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
        altText: { title: text, description: text, name: path.basename(file) } })],
    }),
    caption("Figure", text),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(rich, width, { fill, bold = false, align = AlignmentType.LEFT, size = 20, keepNext = false } = {}) {
  const paras = (Array.isArray(rich) && rich.length && Array.isArray(rich[0]) ? rich : [rich]).map((r) =>
    new Paragraph({ children: runs(r, bold ? { bold, size } : { size }), alignment: align, keepNext,
      spacing: { line: 276, after: 0 } }));
  return new TableCell({
    children: paras, borders, width: { size: width, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 }, verticalAlign: VerticalAlign.TOP,
  });
}

// headers: array of strings; rows: array of arrays of rich; widths: DXA, sum = TEXT_W
function table(headers, rows, widths, { firstColFill = false, size = 20 } = {}) {
  // keepNext on every row but the last keeps a table on one page (all our tables fit on one)
  const head = headers ? [new TableRow({ tableHeader: true, cantSplit: true,
    children: headers.map((h, i) => cell(h, widths[i], { fill: HEAD_FILL, bold: true, size, keepNext: true })) })] : [];
  const body = rows.map((r, k) => new TableRow({ cantSplit: true,
    children: r.map((c, i) => cell(c, widths[i], { size, keepNext: k < rows.length - 1,
      fill: firstColFill && i === 0 ? LABEL_FILL : undefined, bold: firstColFill && i === 0 })) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths, rows: [...head, ...body] });
}
const W = (...fr) => { const s = fr.reduce((a, b) => a + b, 0); const w = fr.map((f) => Math.floor(TEXT_W * f / s)); w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0); return w; };

// ------------------------------------------------------------------ headers and footers
const header = new Header({ children: [new Paragraph({
  alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6", space: 4 } },
  children: [new TextRun({ text: `${SHORT_TITLE} – FYP-I Final Report`, italics: true, size: 18, color: GREY })],
})] });
const footer = new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "FAST School of Computing, NUCES Karachi  |  Page ", size: 18, color: GREY }),
    new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GREY })],
})] });

// ------------------------------------------------------------------ cover page
const logo = fs.readFileSync(path.join(ROOT, "outputs", "report", "logo.png"));
const coverLine = (text, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 60 },
  children: [new TextRun({ text, font: FONT, size: o.size ?? 24, bold: o.bold, italics: o.italics, color: o.color })] });
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 240 },
    children: [new ImageRun({ type: "png", data: logo, transformation: { width: 120, height: 117 },
      altText: { title: "FAST-NUCES logo", description: "National University of Computer and Emerging Sciences logo", name: "logo" } })] }),
  coverLine(TITLE, { size: 34, bold: true, color: NAVY, after: 360 }),
  coverLine("FYP-I FINAL REPORT", { size: 28, bold: true, after: 300 }),
  coverLine("PROJECT SUPERVISOR", { bold: true }),
  coverLine(SUPERVISOR, { after: 200 }),
  coverLine("PROJECT CO-SUPERVISOR", { bold: true }),
  coverLine("N/A", { after: 200 }),
  coverLine("PROJECT TEAM", { bold: true }),
  ...TEAM.map(([n, id]) => coverLine(`${n}    ${id}`)),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 300 },
    children: [new TextRun({ text: "for the Alzheimer's Disease Neuroimaging Initiative", italics: true, size: 20, color: GREY }),
      new FootnoteReferenceRun(1)] }),
  coverLine("Submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science.", { size: 22, after: 300 }),
  coverLine("FAST SCHOOL OF COMPUTING", { size: 26, bold: true }),
  coverLine("NATIONAL UNIVERSITY OF COMPUTER AND EMERGING SCIENCES", { size: 22, bold: true }),
  coverLine("KARACHI CAMPUS", { size: 22, bold: true, after: 240 }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "[Month Year]", highlight: "yellow", size: 24 })] }),
];

const ADNI_BYLINE = "Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf";

// ------------------------------------------------------------------ front matter
const frontHeading = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true, outlineLevel: 0 });

const approval = [
  frontHeading("Approval Page"),
  P(`The undersigned certify that the Final Year Project (FYP-I) titled "${TITLE}", carried out by the students listed below, has been examined and found satisfactory for FYP-I.`),
  blank(120),
  table(null, [
    ["Project Title", TITLE],
    ["Project Supervisor", SUPERVISOR],
    ["Project Co-Supervisor", "N/A"],
    ["Project Team", TEAM.map(([n, id]) => [`${n} – ${id}`])],
    ["Submission Date", [{ hl: "[DD Month YYYY]" }]],
  ], W(1, 2.3), { firstColFill: true, size: 22 }),
  blank(360),
  table(["Supervisor", "Co-Supervisor (if any)"], [
    [`Name: ${SUPERVISOR}`, "Name: N/A"],
    ["Signature: ______________________", "Signature: ______________________"],
    ["Date: __________________________", "Date: __________________________"],
  ], W(1, 1), { size: 22 }),
];

const declaration = [
  frontHeading("Declaration"),
  P("We declare that this report is our own original work, prepared under the guidance of our supervisor. All sources, ideas, figures and code taken from other works have been properly cited and acknowledged. This work has not been submitted previously, in whole or in part, for any other degree, course or publication."),
  blank(240),
  table(["Student Name", "Signature", "Date"], TEAM.map(([n, id]) => [`${n} – ${id}`, "", ""]), W(1.6, 1.3, 0.9), { size: 22 }),
];

const acknowledgements = [
  frontHeading("Acknowledgements"),
  P(`We are grateful to our supervisor, ${SUPERVISOR}, for guidance throughout FYP-I, for helping us think through the research question, and for approving the data-handling arrangements that made GPU training possible. We also thank the FAST School of Computing for the resources and the time to work on this project.`),
  P("Data collection and sharing for the Alzheimer's Disease Neuroimaging Initiative (ADNI) is funded by the National Institute on Aging (National Institutes of Health Grant U19 AG024904). The grantee organization is the Northern California Institute for Research and Education. In the past, ADNI has also received funding from the National Institute of Biomedical Imaging and Bioengineering, the Canadian Institutes of Health Research, and private sector contributions through the Foundation for the National Institutes of Health (FNIH) including generous contributions from the following: AbbVie, Alzheimer's Association; Alzheimer's Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics."),
];

const abstract = [
  frontHeading("Abstract"),
  P("Mild cognitive impairment is often an early stage of Alzheimer's disease, but only some people with it develop dementia within a few years. Knowing early who is likely to progress would help with care planning and with choosing participants for clinical trials. One promising signal is the brain age gap: a model learns what a healthy brain looks like at each age from magnetic resonance imaging (MRI) scans, and the gap is the difference between the age it predicts for a person and that person's real age. This project asks whether the brain age gap adds useful information beyond chronological age. In FYP-I we built the full pipeline on data from the Alzheimer's Disease Neuroimaging Initiative: 935 cognitively unimpaired people for the brain-age model and 453 people with mild cognitive impairment, each labelled by whether they developed dementia within 36 months. Every scan was preprocessed and checked by eye. A three-dimensional convolutional neural network predicted the age of unseen healthy people with a mean absolute error of 3.57 years. After correcting a known age bias, people who later progressed had a brain age gap 2.4 years larger than those who stayed stable. We then trained baseline models on MRI features alone and on MRI features plus age; on a held-out test set, adding age changed the area under the curve by only 0.002. In FYP-II we will add the brain age gap to these models, test whether it improves prediction beyond age, and study the models' explanations and calibration."),
  new Paragraph({ spacing: { before: 120 }, children: [new TextRun({ text: "Keywords: ", bold: true }),
    new TextRun("brain age gap, Alzheimer's disease, mild cognitive impairment, structural MRI, deep learning, ADNI")] }),
];

const toc = [
  frontHeading("Table of Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];
const lof = [
  frontHeading("List of Figures"),
  new TableOfContents("List of Figures", { hyperlink: true, captionLabelIncludingNumbers: "Figure" }),
];
const lot = [
  frontHeading("List of Tables"),
  new TableOfContents("List of Tables", { hyperlink: true, captionLabelIncludingNumbers: "Table" }),
];

const ABBR = [
  ["AD", "Alzheimer's disease"], ["ADNI", "Alzheimer's Disease Neuroimaging Initiative"],
  ["AUC", "Area under the receiver operating characteristic curve (ROC-AUC)"], ["BAG", "Brain age gap"],
  ["CI", "Confidence interval"], ["CN", "Cognitively normal (unimpaired)"], ["CNN", "Convolutional neural network"],
  ["CV", "Cross-validation"], ["E1–E5", "Experiments 1 to 5 (nested progression models)"],
  ["GPU", "Graphics processing unit"], ["MAE", "Mean absolute error"], ["MCI", "Mild cognitive impairment"],
  ["MNI", "Montreal Neurological Institute (standard brain template space)"], ["MRI", "Magnetic resonance imaging"],
  ["N4", "N4 bias-field correction"], ["PCA", "Principal component analysis"],
  ["PR-AUC", "Area under the precision-recall curve"], ["QC", "Quality control"], ["RID", "ADNI subject number"],
  ["RMSE", "Root mean squared error"], ["ROC", "Receiver operating characteristic"], ["SD", "Standard deviation"],
  ["T", "Tesla (MRI field strength)"], ["T1", "T1-weighted MRI"], ["XAI", "Explainable artificial intelligence"],
];
const abbreviations = [
  frontHeading("List of Abbreviations"),
  table(["Abbreviation", "Full Form"], ABBR.map(([a, f]) => [a, f]), W(1, 3.2)),
];

// ------------------------------------------------------------------ chapter 1
const ch1 = [
  H1nb("1. Introduction"),
  P("This chapter explains why predicting progression from mild cognitive impairment (MCI) to Alzheimer's disease (AD) matters, what problem the project addresses, and what it sets out to achieve by the end of FYP-II."),
  H2("1.1 Background and Motivation"),
  P("Dementia affected about 57 million people worldwide in 2021, with nearly 10 million new cases each year, and Alzheimer's disease is its most common cause [1]. Before dementia, many people pass through a stage called MCI: their memory or thinking is worse than expected for their age, but they still manage everyday life. MCI does not always lead to dementia. In the American Academy of Neurology practice guideline, about 15% of people with MCI aged over 65 developed dementia within two years [2]; others stay stable for years, and some improve."),
  P("This uncertainty is the practical problem. If clinicians and researchers could tell early which people with MCI are likely to progress, they could plan care sooner and choose suitable people for trials of new treatments. Structural MRI is a natural place to look. It is widely available, does not use radiation, and shows the brain shrinkage (atrophy) that comes with Alzheimer's disease."),
  P(["A promising way to summarise a whole scan in one number is ", { i: "brain age" }, " [3], [4]. A model is trained on scans of healthy people to predict their age. When it is applied to someone new, the difference between the predicted age and the real age is the brain age gap (BAG). A positive gap means the brain looks older than expected. BAG has been linked with many neurological and psychiatric conditions [5], and an earlier study on the same public dataset we use found that it helped predict which people with MCI went on to develop AD [6]."]),
  P("There is a catch. BAG is calculated from age, and the people most likely to progress also tend to be older. A model that seems to gain from BAG may therefore just be using age in another form [7]. Before BAG can be called useful, it has to be tested against age directly and fairly. That is the starting point of this project."),
  H2("1.2 Problem Statement"),
  P("It is not known whether the brain age gap from a structural MRI scan tells us anything about a person's chance of progressing from MCI to Alzheimer's disease that their chronological age does not already tell us. Answering this is harder than it looks, for four reasons:"),
  bullet("BAG and age are closely related, so a study that does not compare against an age-only model cannot separate the two."),
  bullet("Brain-age models tend to over-estimate the age of younger people and under-estimate older people. How this bias is corrected can change later results [7]."),
  bullet("Data leakage, for example the same person appearing in both the training and the test data, has inflated many reported results in this field (Chapter 2)."),
  bullet("Studies use different cohorts, scanners, preprocessing and test sets, so their numbers cannot be compared directly."),
  P("The people affected are researchers who want to use BAG as a marker of disease risk and, later, clinical trial designers. Our project is a research study; it is not a clinical tool."),
  H2("1.3 Objectives"),
  P("Each objective below can be checked at the end of FYP-II. The first three are already met (Chapter 6)."),
  numbered([{ b: "Cohorts. " }, "Build two cohorts from ADNI, cognitively normal (CN) people for brain age and MCI people with a 36-month outcome, with subject-level train, validation and test splits in which no person appears twice. This is checked by automated tests."]),
  numbered([{ b: "Brain-age model. " }, "Train a brain-age model on healthy people only, with a lower mean absolute error (MAE) than a ridge-regression baseline on a held-out test set, and an age-bias correction fitted on training data only."]),
  numbered([{ b: "BAG. " }, "Compute the bias-corrected BAG for every MCI subject and test, with a pre-specified test, whether it differs between people who progress and people who stay stable."]),
  numbered([{ b: "Nested experiments. " }, "Run five progression experiments (E1–E5) that use the same people, splits, seeds and model types and differ only in their inputs. Test whether BAG improves prediction beyond age (E4 against E2) with the DeLong test and report confidence intervals."]),
  numbered([{ b: "Trust. " }, "Explain the models' predictions and check whether their predicted probabilities are well calibrated (FYP-II)."]),
  H2("1.4 Scope"),
  P([{ b: "In scope: " }, "T1-weighted structural MRI from ADNI; brain-age regression; BAG; prediction of MCI-to-AD progression from baseline data; the nested experiments E1–E5, including an optional small set of clinical variables in E5; explainability and calibration."]),
  P([{ b: "Out of scope: " }, "clinical diagnosis or deployment; collecting new data; a new network architecture as the main contribution; other imaging types such as PET; genetics; changes in BAG over time (longitudinal BAG); external datasets; and a user-facing application, which is not planned. These are left out to keep the study controlled and achievable in two semesters."]),
  H2("1.5 Report Organization"),
  P("Chapter 2 reviews related work on brain age, bias correction and MRI-based prediction of progression, and states the research gap. Chapter 3 describes the proposed solution, the data, the methods and the tools. Chapter 4 summarises the requirements of the research pipeline, and Chapter 5 its design. Chapter 6 reports the work completed in FYP-I, the results so far and the test plan. Chapter 7 gives the project plan, roles and risks, and Chapter 8 concludes and outlines FYP-II."),
];

// ------------------------------------------------------------------ chapter 2
const ch2 = [
  H1("2. Related Work"),
  P("We found papers through Google Scholar and PubMed using search terms such as \"brain age\", \"brain age gap\", \"BrainAGE\", \"mild cognitive impairment conversion\", \"MRI\" and \"ADNI\". We focused on work from 2010 onwards and followed the references of the most relevant papers. The review is grouped into four themes."),
  H2("2.1 Review of Literature"),
  H3("2.1.1 Predicting brain age from MRI"),
  P("Franke et al. introduced the BrainAGE approach, which predicts age from T1-weighted scans of healthy people with a kernel regression model [3]. Reviews since then show that brain age has become a widely used summary of brain health [4], [5]. Deep learning changed how the models are built. Cole et al. trained a convolutional neural network (CNN) and reported an MAE of 4.16 years on grey-matter maps and 4.65 years on minimally processed scans, with reliable results between repeated scans [8]. Peng et al. designed a small, efficient CNN that reached an MAE of 2.14 years on about 13,000 UK Biobank participants using linearly registered T1 scans; in their tests a single network did equally well on linearly registered scans and on grey-matter maps, and it beat an elastic-net regression even with only 50 training subjects [9]. Bashyam et al. trained a network on 14,468 people and found that the most accurate age predictor is not always the most useful one for detecting disease [10]."),
  P([{ b: "Limitation for us: " }, "accuracy depends heavily on the size of the training set, and most of these models were trained on thousands of people. We have 654 healthy training subjects, so an MAE of 3 to 4 years is a realistic target."]),
  H3("2.1.2 Age bias and its correction"),
  P("Brain-age predictions are pulled towards the average age of the training set, so the gap is positive for young people and negative for old people even when nothing is wrong. Smith et al. studied this effect in 19,000 UK Biobank participants and proposed ways to correct it [11]. Le et al. used simulations to show that BAG remains linked to age and that any later analysis should include age [7]. de Lange and Cole compared common correction procedures and discussed how each affects later analyses [12]. The lessons for our work are that the correction must be fitted on training data only, and that age must be kept in the comparison models."),
  H3("2.1.3 The brain age gap in MCI and Alzheimer's disease"),
  P("Gaser et al. used ADNI data and found that BrainAGE predicted conversion from MCI to AD within three years better than cognitive scores and cerebrospinal-fluid markers, with an accuracy of up to 81% [6]. A later review by the same group reports that each extra year of BrainAGE at baseline was linked to about 10% higher risk of conversion [5]. These results are encouraging but have two weaknesses: they mix scanners of different field strengths, and they rarely test whether BAG still helps once age is already in the model."),
  H3("2.1.4 Predicting MCI-to-AD conversion from MRI"),
  P("Cuingnet et al. compared ten MRI classification methods on ADNI and found that, for separating MCI people who convert from those who do not, most methods were only modestly better than chance [13]. Moradi et al. reported an MRI-only area under the curve (AUC) of 0.77, which rose to 0.90 when age and cognitive tests were added; they also found that removing the effect of age from the MRI features helped [14]. Samper-González et al. built an open, reproducible framework for this kind of experiment and found that voxel-based and region-based features performed similarly [15]. Wen et al. reviewed CNN studies on AD classification and found that many may have suffered from data leakage, which inflated their results; with careful validation, CNNs did not clearly beat simpler models [16]."),
  P([{ b: "Lesson for us: " }, "an honest MRI-only AUC for progression is in the range of roughly 0.65 to 0.77, and strict separation of training and test data matters more than the choice of model."]),
  H2("2.2 Comparison of Existing Systems"),
  P("Table 1 compares the most relevant studies with our project."),
  tableCaption("Comparison of existing approaches"),
  table(["System / Paper", "Approach", "Strengths", "Limitations", "Relevance to our project"], [
    ["Franke et al. [3]", "Kernel regression on T1 scans of healthy people", "First BrainAGE method; simple and transparent", "Hand-crafted pipeline; small training set", "Defines the brain age idea we build on"],
    ["Cole et al. [8]", "CNN on grey-matter maps and raw scans", "Reliable and heritable brain age", "Large data needed; grey-matter maps need extra processing", "Shows CNNs work on minimally processed T1 scans"],
    ["Peng et al. [9]", "Lightweight CNN (SFCN) on linearly registered T1", "MAE 2.14 years; beats elastic net at every training size", "Trained on about 13,000 people", "Model design and input type close to ours"],
    ["Gaser et al. [6]", "BrainAGE to predict MCI-to-AD conversion (ADNI)", "Beat cognitive scores and fluid markers", "Mixed field strengths; no direct test against age", "The question our E4 vs E2 comparison tests"],
    ["Moradi et al. [14]", "Voxel MRI features + age + cognition", "AUC 0.77 (MRI) and 0.90 (combined)", "Feature selection on the same data; one dataset", "Benchmark for our MRI-only model; age matters"],
    ["Wen et al. [16]", "Review and reproducible CNN evaluation", "Exposes data leakage in published work", "Classification of current diagnosis, not BAG", "Basis for our leakage controls"],
    [{ b: "Our project" }, "Brain age (CNN) on CN; nested models E1–E5 on MCI", "Same people and folds in every experiment; test set used once; every scan checked", "Small test set (69 MCI); 3 T scans only", "Tests BAG beyond age directly"],
  ], W(1.15, 1.45, 1.45, 1.45, 1.4), { size: 18 }),
  H2("2.3 Research Gap"),
  P("Two lines of work rarely meet. Brain-age studies show that BAG is higher in people with MCI who progress, but they seldom test whether BAG adds anything once chronological age is part of the model. Conversion-prediction studies build strong classifiers but rarely isolate the contribution of BAG, and some report results that are inflated by leakage. This project fills that gap with a set of nested experiments (E1–E5) that keep everything fixed except the inputs: the same subjects, splits, seeds and model types, a bias correction fitted only on training data, and a test set used once, after each pipeline is frozen. A negative result, that BAG adds nothing beyond age, is as useful as a positive one and will be reported in the same way."),
];

// ------------------------------------------------------------------ chapter 3
const ch3 = [
  H1("3. Proposed Solution and Methodology"),
  H2("3.1 Solution Overview"),
  P("The solution is a research pipeline with six stages (Figure 1). Scans and clinical tables from ADNI are turned into two cohorts. Every scan is preprocessed and checked. A brain-age model is trained on healthy people only and then applied to people with MCI, giving each of them a BAG. Finally, a series of nested models predicts whether each MCI person progresses to AD within 36 months, and FYP-II adds explanations and a calibration check. FYP-I covered everything up to and including experiments E1 and E2."),
  ...figure("docs/figures/pipeline_overview.png", 6.3, "High-level overview of the proposed solution"),
  H2("3.2 Methodology and Development Process"),
  P("We followed an iterative, step-by-step research process, close to a light form of Agile. The work was split into a ten-step roadmap, and each step was planned, done, checked and recorded before the next one started. Every choice that could affect the results was written in a decision log before it was used (36 decisions so far), every model run in an experiment log, and every working session in a session log. We chose this process because in a research project the biggest risk is not a missing feature but a result that is quietly biased; writing the rules down before seeing the results protects against that, and makes the work easy to check."),
  H3("3.2.1 Data and cohorts"),
  P("Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial magnetic resonance imaging (MRI), positron emission tomography (PET), other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimer's disease (AD) [17]. The clinical and imaging protocols are described in [18] and [19]."),
  P("We built two cohorts from the raw ADNI tables using rules fixed before any model was trained (Figure 2). The brain-age cohort contains cognitively normal people who stayed normal at every later visit, so that the model learns normal ageing. The progression cohort contains people with MCI at baseline. A person is a progressor if they were diagnosed with dementia within 36 months, and stable if they had no dementia diagnosis and at least 36 months of follow-up; people who went back to normal were excluded because their label is uncertain. We used 3 T scans only, because 43% of MCI progressors but only 10% of healthy people had been scanned at 1.5 T, and BAG could otherwise reflect the scanner rather than the brain."),
  P("Each cohort was split once, by person, into 70% training, 15% validation and 15% test (seed 42), stratified by outcome (MCI) or age band (CN). The final data contain 935 CN people (654 for training with 1,755 scans across their visits, 140 for validation and 141 for testing) and 453 MCI people (316, 68 and 69), of whom 34% progressed."),
  ...figure("outputs/figures/cohort_flow.png", 6.3, "How subjects were selected for the two cohorts"),
  H3("3.2.2 MRI preprocessing and quality control"),
  P("Each T1 scan was corrected for intensity non-uniformity with N4 [20], registered with an affine (rigid plus scaling) transform to the MNI ICBM152 2009a template [21] using SimpleITK [22], cropped to 160 × 192 × 160 voxels of 1 mm, and normalised to zero mean and unit variance. We did not strip the skull or warp the brains non-linearly, so that atrophy stays visible in the image. Preprocessing took about 24 seconds per scan on a laptop."),
  P("Quality control was strict. Every one of the 2,511 baseline and follow-up scans was checked by eye on a picture of its slices. Misaligned scans were redone with a registration that tries several starting orientations. Scans with motion or other artefacts over the brain were replaced by another scan from the same session, or the person was removed. A final automatic check measured each scan's rotation against a mean image of all scans and caught about 20 slightly tilted scans the eye check had missed. It also found four people who had been enrolled twice under different IDs, which would have leaked a person between splits; the later enrolment was removed. After all checks, 1,388 people and 2,489 scans remained."),
  H3("3.2.3 Brain-age model"),
  P("We first trained a ridge-regression baseline on the scans shrunk to 4 mm (29,520 values per scan), with principal component analysis (PCA) fitted inside each cross-validation fold. We then trained a 3D CNN with five convolution blocks (about 1.2 million weights) on 2 mm images (80 × 96 × 80), with random flips and shifts during training, an L1 loss and early stopping on the validation set. The CNN was trained on a Google Colab GPU, using a checked copy of the data stored in the university's Google Drive with the supervisor's approval."),
  P("Brain-age models are biased with age (Section 2.1.2), so we fitted a linear correction on training data only. To get honest training predictions we trained five CNNs, each on four fifths of the training people, and used each to predict the fifth it had not seen. A line fitted to these out-of-fold predictions gives the bias, and the corrected gap is the raw gap minus that line. The final model is the average of the five CNNs plus this correction. It was chosen on the validation set and frozen, with the fingerprints (SHA-256 hashes) of its files committed to the repository, before the test set was used once."),
  H3("3.2.4 Progression experiments E1 and E2"),
  P("All experiments use the same MRI features: the 4 mm brain voxels of each MCI person's baseline scan. They are not trained to predict age, so they do not carry brain-age information into E1 by design. The model pipeline standardises the voxels, reduces them with PCA, standardises the components and feeds them to a classifier: L2-regularised logistic regression (the main model, fixed in advance) or a random forest (secondary), both weighted for the unequal class sizes. E2 is identical to E1 except that standardised age is added as one extra input after PCA."),
  P("Settings such as the number of PCA components were chosen by stratified 5-fold cross-validation (CV) on the 316 training people only, using ROC-AUC. The cut-off that turns a probability into a yes or no was chosen on training predictions only (Youden's J). E1 and E2 used exactly the same folds, which we checked with a fingerprint of the fold assignment. The validation set (68) was used to check the models, and the test set (69) was used once, after both pipelines had been frozen."),
  H3("3.2.5 Evaluation"),
  P("Brain-age models are measured with MAE, root mean squared error (RMSE), and Pearson and Spearman correlation, always on uncorrected predictions. Progression models are measured with ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1 and the confusion matrix, since accuracy alone is misleading with unequal classes. Confidence intervals use the DeLong method for AUC and a bootstrap (5,000 resamples) for the other measures. Two models scored on the same people are compared with the paired DeLong test [25], computed with the fast algorithm of Sun and Xu [26]. The difference in BAG between progressors and stable people is tested with Welch's t-test, with Hedges' g as the effect size and a Mann-Whitney U test as a check. Every test was written down before it was run."),
  H2("3.3 Tools and Technologies"),
  P("Table 2 lists the tools we used. Everything is free and open source except the cloud GPU, which was used within Google Colab's free tier."),
  tableCaption("Technology stack"),
  table(["Layer / Purpose", "Tool / Technology", "Justification"], [
    ["Language and environment", "Python 3.11, conda, pinned package versions", "Standard for machine learning; exact versions make results repeatable"],
    ["Data handling", "NumPy, pandas, YAML configuration files", "Fast arrays and tables; settings kept out of the code"],
    ["Medical imaging", "SimpleITK [22], NiBabel, nilearn", "N4 correction, registration and reading of MRI files"],
    ["Machine learning", "scikit-learn [23]", "Ridge, logistic regression, random forest, PCA and cross-validation"],
    ["Deep learning", "PyTorch [24]", "3D CNN training on GPU"],
    ["Statistics and plots", "SciPy, Matplotlib", "Statistical tests and report figures"],
    ["Testing", "pytest", "Automatic checks of splits, configurations and metrics"],
    ["Version control", "Git and GitHub (code and documents only)", "History of every change; no ADNI data ever committed"],
    ["Compute", "Laptop CPU; Google Colab GPU", "All steps on the laptop except CNN training"],
  ], W(1.3, 1.6, 2.4)),
  H2("3.4 Feasibility Study"),
  tableCaption("Feasibility of the project"),
  table(null, [
    ["Technical feasibility", "Shown in practice: the whole FYP-I pipeline, including the CNN, was built and run with the team's skills, a laptop and the free Colab GPU. The remaining work (E3–E5, explanations, calibration) uses the same tools and data."],
    ["Operational feasibility", "The users are researchers, not patients or clinicians. The pipeline runs from documented commands (Appendix B), and automated tests catch the most dangerous mistakes, such as a person appearing in two splits. No clinical use is intended."],
    ["Economic feasibility", "No cost: all software is open source, ADNI data are free for approved researchers, and the GPU came from Colab's free tier. The ADNI Data Use Agreement forbids sharing the data, so scans and subject tables never leave approved storage and are never committed to GitHub."],
  ], W(1, 3), { firstColFill: true }),
];

// ------------------------------------------------------------------ chapter 4
const ch4 = [
  H1("4. Requirements"),
  P("This is a research project, so the requirements describe what the analysis pipeline must do and the standards its results must meet, rather than the features of an application. If the supervisor asks for a full Software Requirements Specification, it will expand the tables below."),
  H2("4.1 Functional Requirements (Summary)"),
  tableCaption("Functional requirements"),
  table(["ID", "Requirement", "Priority"], [
    ["FR-001", "The system shall build the CN and MCI cohorts from the raw ADNI tables using the fixed cohort rules, and report every exclusion.", "Must"],
    ["FR-002", "The system shall create subject-level 70/15/15 splits once (seed 42) and refuse to overwrite them.", "Must"],
    ["FR-003", "The system shall preprocess every T1 scan (N4, affine registration to MNI, crop, normalisation) and produce quality-control pictures.", "Must"],
    ["FR-004", "The system shall train brain-age models on CN training subjects only and report MAE, RMSE, Pearson and Spearman correlation.", "Must"],
    ["FR-005", "The system shall fit the age-bias correction on training data only and apply it to all other subjects.", "Must"],
    ["FR-006", "The system shall compute the corrected BAG for every MCI subject and store it with the outcome label.", "Must"],
    ["FR-007", "The system shall train experiments E1–E5 with identical subjects, folds, seeds and model types, changing only the inputs.", "Must"],
    ["FR-008", "The system shall compare nested experiments with the paired DeLong test and give confidence intervals.", "Must"],
    ["FR-009", "The system shall explain model predictions (for example with SHAP or Grad-CAM) and assess calibration.", "Should"],
    ["FR-010", "The system shall optionally add a small set of clinical variables (E5).", "Could"],
  ], W(0.7, 4.3, 0.8)),
  H2("4.2 Non-Functional Requirements (Summary)"),
  tableCaption("Non-functional requirements"),
  table(["ID", "Requirement (measurable)", "Category"], [
    ["NFR-001", "No ADNI scan, table, subject ID or model file is ever committed to the repository; checked by .gitignore rules and a scan of the full git history.", "Privacy"],
    ["NFR-002", "No subject appears in more than one split or in both cohorts; checked by automated tests.", "Validity"],
    ["NFR-003", "Each test set is used exactly once, after the pipeline is frozen; the scripts refuse to run on it without a flag or a second time.", "Validity"],
    ["NFR-004", "Running a pipeline twice with the same configuration gives the same results (seed 42, pinned versions, file hashes checked).", "Reproducibility"],
    ["NFR-005", "All paths and settings live in configuration files, so the code runs unchanged on another machine.", "Portability"],
    ["NFR-006", "Preprocessing takes under one minute per scan on a laptop; the full test suite runs in under one minute.", "Performance"],
    ["NFR-007", "Every result-affecting choice is recorded in the decision log before it is used.", "Transparency"],
  ], W(0.8, 4.1, 1.1)),
  H2("4.3 Use Case Diagram"),
  P("Not applicable: the project has no interactive end users or application. Its only users are the team and other researchers, who run the pipeline from the command line in the order given in Appendix B. Figure 1 shows the flow of the pipeline instead."),
];

// ------------------------------------------------------------------ chapter 5
const ch5 = [
  H1("5. Design"),
  P("This chapter summarises the design of the pipeline. Further detail is in the code, the configuration files and the decision log in the project repository."),
  H2("5.1 System Architecture"),
  P("The code is split into one Python package per stage (Figure 3). Every script reads its settings from a configuration file, reads its inputs from the local data folders, and writes its results to a local output folder. Only code, configuration and documents are kept in the public GitHub repository; all ADNI-derived files stay on the team's approved storage. All steps run on a laptop except CNN training, which ran on a Colab GPU."),
  ...figure("docs/figures/architecture.png", 6.3, "System architecture"),
  H2("5.2 Database Design"),
  P("The project does not use a database server; the data are small enough to keep in CSV and NumPy files. The tables are linked by the ADNI subject number (RID) and the image ID of each scan (Figure 4). Two rules keep the design safe: which scan to use always comes from a quality-checked table, never from a folder listing, and each table has one row per subject unless it says otherwise."),
  ...figure("docs/figures/data_design.png", 6.3, "Main data tables and how they are linked (in place of an entity-relationship diagram)"),
  H2("5.3 Key Design Diagrams"),
  P("The most important design rule is how a model moves from training to the test set (Figure 5). The rules are written down first. The model is trained and tuned on training data only, and checked on the validation set. If a setting changes, tuning starts again on training data. When the model is final, the choice and the fingerprints of the model files are committed, which freezes it. Only then is the test set used, once: the evaluation script checks the fingerprints, checks that the model still reproduces its validation results, and refuses to run on the test set a second time. This rule was applied to both the brain-age model and E1/E2."),
  ...figure("docs/figures/freeze_workflow.png", 6.3, "Workflow from training to the one-time test"),
  H2("5.4 User Interface Design"),
  P("Not applicable: no user interface is planned. The pipeline's outputs are tables, logs and figures, which are reported in Chapter 6."),
];

// ------------------------------------------------------------------ chapter 6
const ch6 = [
  H1("6. Preliminary Implementation and Test Plan"),
  P("FYP-I set out to build and test everything up to experiment E2. This chapter lists what was completed, presents the results, and describes how the work is and will be tested."),
  H2("6.1 Work Completed in FYP-I"),
  bullet([{ b: "Data access and setup: " }, "ADNI access, a public code repository with strict rules against committing data, and a reproducible Python environment."]),
  bullet([{ b: "Cohorts and splits: " }, "two cohorts of 935 CN and 453 MCI people, built from the raw ADNI tables with written rules and subject-level splits (Figure 2)."]),
  bullet([{ b: "Preprocessing: " }, "2,489 scans preprocessed and checked by eye, with tilt and duplicate-person checks."]),
  bullet([{ b: "Brain-age model: " }, "ridge baseline and 3D CNN, age-bias correction fitted on training data, final model frozen and tested once."]),
  bullet([{ b: "BAG: " }, "corrected BAG for all 453 MCI people and a pre-specified comparison of progressors and stable people."]),
  bullet([{ b: "E1 and E2: " }, "MRI-only and MRI-plus-age models, frozen and tested once, with the pre-specified E2 vs E1 comparison."]),
  bullet([{ b: "Quality and reproducibility: " }, "16 automated tests, pinned package versions, and a step-by-step guide to reproduce every result (Appendix B)."]),
  H2("6.2 Proof of Concept / Preliminary Results"),
  H3("6.2.1 Brain age"),
  P("Table 6 shows the brain-age results. The final model (the average of five CNNs) had the lowest validation error and was the only candidate tested as the final model; the other two are shown for comparison only. On 141 healthy people it had never seen, it predicted age with an MAE of 3.57 years (95% CI 3.10 to 4.07). This is in line with published models trained on similar numbers of people. Before correction, the gap fell by 0.40 years for every year of age; after the correction fitted on training data, the slope was -0.025, so the bias was removed on unseen people (Figure 6)."),
  tableCaption("Brain-age results (uncorrected predictions)"),
  table(["Model", "Set", "MAE (years)", "RMSE", "Pearson r", "Spearman ρ"], [
    ["Ridge baseline", "Validation (140)", "3.79", "4.88", "0.71", "–"],
    ["Single CNN (cnn-1)", "Validation (140)", "3.45", "4.33", "0.79", "–"],
    [{ b: "Final: 5-CNN average" }, "Validation (140)", "3.42", "4.30", "0.79", "0.77"],
    [{ b: "Final: 5-CNN average" }, { b: "Test, once (141)" }, { b: "3.57" }, { b: "4.57" }, { b: "0.74" }, { b: "0.72" }],
    ["Ridge baseline (comparison)", "Test (141)", "3.83", "4.90", "0.70", "0.68"],
    ["Single CNN (comparison)", "Test (141)", "3.60", "4.77", "0.74", "0.73"],
  ], W(2.1, 1.5, 1, 0.8, 0.9, 0.9)),
  ...figure("outputs/figures/brain_age_test.png", 6.3, "Brain-age model on the CN test set: (a) predicted against actual age; (b) the age bias before and after correction"),
  H3("6.2.2 Brain age gap"),
  P("Across all MCI people the corrected BAG averaged +2.52 years, compared with +0.15 years in healthy test subjects, so brains with MCI looked older than their age. Following the plan fixed in advance, we compared progressors and stable people in the training and validation sets only (384 people), keeping the MCI test set for later. Progressors had a mean BAG of +4.15 years (SD 3.52, n = 130) and stable people +1.75 years (SD 3.54, n = 254): a difference of 2.40 years (95% CI 1.65 to 3.17), Hedges' g = 0.68 (0.46 to 0.92), Welch p < 0.001, confirmed by the Mann-Whitney test (Figure 7). The two groups differed in age by only 1.3 years (p = 0.085). This shows an association, not yet a prediction; whether BAG helps prediction beyond age is the question for E3 and E4."),
  ...figure("outputs/figures/bag_groups.png", 4.2, "Corrected brain age gap by 36-month outcome (MCI training and validation sets)"),
  H3("6.2.3 Experiments E1 and E2"),
  P("Table 7 shows the results of E1 (MRI only) and E2 (MRI plus age). With logistic regression, the main model, the MRI-only baseline reached a validation AUC of 0.684 but only 0.576 on the test set, where its confidence interval includes 0.5. With 69 test people the uncertainty is large, and the intervals for validation and test overlap widely; still, the honest reading is that our MRI-only baseline is weak. This is at the lower end of published MRI-only results (Section 2.1.4), as expected for simple intensity features after affine registration rather than grey-matter maps."),
  P("The pre-specified comparison was clear. Adding age changed the test AUC by +0.002 (DeLong 95% CI -0.011 to +0.014, p = 0.77), and the two models' ROC curves lie almost on top of each other (Figure 8). We therefore find no evidence that chronological age adds discrimination beyond the MRI features, and the interval rules out a gain larger than about 0.015 AUC. A likely reason is that the voxel features already show how old a brain looks. This sets up the main question for FYP-II: does BAG add something that age does not?"),
  tableCaption("E1 and E2 results (logistic regression is the main model; test set used once)"),
  table(["Model", "Set", "ROC-AUC (95% CI)", "PR-AUC", "Sensitivity", "Specificity", "F1"], [
    ["E1 logistic regression", "Validation", "0.684 (0.547–0.813)", "0.556", "0.48", "0.76", "0.49"],
    ["E2 logistic regression", "Validation", "0.685 (0.548–0.813)", "0.557", "0.48", "0.76", "0.49"],
    [{ b: "E1 logistic regression" }, { b: "Test" }, { b: "0.576 (0.426–0.726)" }, "0.521", "0.33", "0.76", "0.37"],
    [{ b: "E2 logistic regression" }, { b: "Test" }, { b: "0.578 (0.429–0.727)" }, "0.518", "0.38", "0.78", "0.42"],
    ["E1 random forest", "Test", "0.617 (0.474–0.759)", "0.468", "0.33", "0.71", "0.36"],
    ["E2 random forest", "Test", "0.615 (0.481–0.749)", "0.448", "0.58", "0.56", "0.48"],
  ], W(1.8, 0.95, 1.6, 0.75, 0.9, 0.9, 0.6), { size: 18 }),
  P("Chance level for PR-AUC is the share of progressors (0.34). The random forest is a secondary model; its E2 vs E1 difference was -0.002 (p = 0.97).", { spacing: { before: 160, line: LINE, after: 120 } }),
  ...figure("docs/figures/roc_curves.png", 6.3, "ROC curves of E1 and E2 (logistic regression) on (a) validation and (b) the test set"),
  H3("6.2.4 Limitations so far"),
  bullet("The MCI test set has 69 people, so confidence intervals are wide and only large differences can be detected."),
  bullet("Our MRI features are image intensities after affine registration, which are simpler than the grey-matter maps used in many studies."),
  bullet("Only 3 T scans were used, which avoids scanner bias but reduced the number of progressors."),
  bullet("The corrected BAG is computed with chronological age, so it still carries some age information. E4 (MRI + age + BAG) against E2 is therefore the key comparison."),
  H2("6.3 Test Plan"),
  P("Testing in this project means checking that the pipeline cannot produce a biased or leaked result. Sixteen automated tests (pytest) already run in about 20 seconds: some use simulated data and run anywhere, and others check the real local data and skip themselves on a machine without it. In FYP-II every new experiment will pass the same tests plus new ones for E3–E5, explanations and calibration. Success means that all tests pass before any test-set run, and that each test-set run happens once, with hashes that match the committed ones."),
  tableCaption("Test plan"),
  table(["Test ID", "Feature / Requirement", "Test approach", "Planned phase"], [
    ["TC-001", "FR-002, NFR-002: no subject in two splits or both cohorts", "Unit test on split tables", "FYP-I (passing)"],
    ["TC-002", "FR-003: correct scans used", "Feature table matches the quality-checked BAG table", "FYP-I (passing)"],
    ["TC-003", "FR-007: E1 and E2 differ only by age", "Compare configuration files key by key", "FYP-I (passing); extend to E3–E5"],
    ["TC-004", "FR-007: identical folds", "Fold fingerprint and saved fold assignment compared", "FYP-I (passing)"],
    ["TC-005", "FR-008: DeLong test correct", "Compare with scikit-learn AUC and bootstrap on simulated data", "FYP-I (passing)"],
    ["TC-006", "NFR-003: test set used once", "Evaluation script must refuse a second run", "FYP-I (passing)"],
    ["TC-007", "NFR-004: reproducibility", "Rerun gives identical predictions; hashes match", "FYP-I (done); every FYP-II run"],
    ["TC-008", "FR-006, FR-007: E3 and E4 use the frozen BAG", "BAG values match the frozen BAG table", "FYP-II"],
    ["TC-009", "FR-009: calibration", "Calibration curve, Brier score and expected calibration error on validation", "FYP-II"],
    ["TC-010", "FR-009: explanation stability", "Explanations repeated across folds and seeds compared", "FYP-II"],
    ["TC-011", "NFR-001: no data in git", "Scan of full git history before each submission", "FYP-I (done); FYP-II"],
  ], W(0.8, 2.2, 2.3, 1.3), { size: 18 }),
];

// ------------------------------------------------------------------ chapter 7
const ch7 = [
  H1("7. Project Plan"),
  H2("7.1 Work Breakdown and Timeline"),
  P("The work is broken into ten roadmap steps: (1) ADNI access, (2) repository, (3) environment, (4) cohorts and splits, (5) MRI preprocessing, (6) brain-age model, (7) BAG, (8) E1, (9) E2, and (10) FYP-I deliverables, followed in FYP-II by E3–E5, statistics, explanations, calibration and the thesis. Figure 9 shows the timeline. FYP-II is shown in months because its dates have not been fixed yet."),
  ...figure("docs/figures/gantt.png", 6.3, "Project timeline (Gantt chart)"),
  H2("7.2 Team Roles and Responsibilities"),
  P("All three members worked on every part of FYP-I together. On our supervisor's advice, each member leads one of the three streams from our proposal; all of us share the literature review, evaluation, report writing and presentations."),
  tableCaption("Team roles and responsibilities"),
  table(["Member", "Role", "Responsibilities"], [
    ["Farrukh Imran (23K-0776)", "Brain Age AI lead", "MRI preprocessing and quality control, brain-age baseline and 3D CNN, age-bias correction, BAG computation"],
    ["Syed Anfas (23K-0574)", "Progression AI lead", "Cohort definition, outcome labels and splits, MCI features, progression models E1–E5, statistical comparisons"],
    ["Asher Ahmed (23K-0511)", "Trustworthy AI and Integration lead", "Experiment tracking, automated tests and reproducibility, figures; explainability, calibration and final integration in FYP-II"],
  ], W(1.5, 1.4, 3.2)),
  H2("7.3 Risk Management"),
  tableCaption("Risks and how we manage them"),
  table(["Risk", "Likelihood", "Impact", "Mitigation"], [
    ["Small test set gives wide intervals, so a real effect of BAG is missed", "H", "H", "Pre-specified tests, confidence intervals, and a final BAG comparison on all 453 people; report \"no evidence\" honestly rather than \"no effect\""],
    ["Data leakage inflates results", "L", "H", "Subject-level splits, automated tests, test set used once with hash checks"],
    ["Breach of the ADNI Data Use Agreement", "L", "H", "Data never in git; .gitignore and history scans; shared storage limited to approved accounts; personal-account data deleted (done)"],
    ["BAG carries age information, confusing E3 vs E1", "M", "M", "Treat E4 vs E2 as the primary comparison"],
    ["Weak MRI baseline limits all experiments", "M", "M", "Same features in all experiments keep the comparison fair; discuss a grey-matter sensitivity analysis with the supervisor"],
    ["Free GPU becomes unavailable", "M", "M", "Cached 2 mm data ready for the university lab GPU; FYP-II models are mostly CPU-friendly"],
    ["Supervisor review of the cohort rules still pending", "M", "M", "Review requested; any change will be logged and its effect reported"],
  ], W(2.3, 0.75, 0.6, 3.0), { size: 18 }),
  H2("7.4 Plan for FYP-II"),
  P("The main deliverables and milestones for FYP-II are:"),
  numbered([{ b: "E3 and E4: " }, "add the frozen BAG to the E1 and E2 pipelines; compare E4 with E2 (primary) and E3 with E1."], 2),
  numbered([{ b: "E5 (optional): " }, "add a small, pre-chosen set of clinical variables and compare with E4."], 2),
  numbered([{ b: "Statistics: " }, "pre-specified tests for every comparison, and the BAG group comparison repeated on all 453 MCI people once E1–E5 are frozen."], 2),
  numbered([{ b: "Explainability: " }, "explanations of the progression and brain-age models, judged on anatomical plausibility and stability."], 2),
  numbered([{ b: "Calibration: " }, "calibration curves, Brier score and expected calibration error."], 2),
  numbered([{ b: "Integration and thesis: " }, "a clean, reproducible package of the whole pipeline, the final thesis, slides and defence."], 2),
];

// ------------------------------------------------------------------ chapter 8
const ch8 = [
  H1("8. Conclusion and Future Work"),
  H2("8.1 Conclusion"),
  P("FYP-I delivered the complete foundation for answering our research question. We built two carefully checked cohorts from ADNI and preprocessed and inspected every scan (objective 1). Our brain-age model predicts the age of unseen healthy people with an error of 3.57 years, better than the ridge baseline, and its age bias was removed with a correction fitted on training data only (objective 2). The brain age gap was clearly larger in people with MCI who later developed dementia, by 2.4 years on average (objective 3). Finally, the first two nested experiments showed that adding chronological age to MRI features does not improve prediction (part of objective 4)."),
  P("Just as important is how the work was done. Every result-affecting choice was written down before it was used, no person appears in more than one split, and each test set was used once, after the model was frozen. Some results are modest, especially the MRI-only baseline on the test set, and we report them as they are. This careful approach means that the answer to our main question in FYP-II, whether positive or negative, can be trusted."),
  H2("8.2 Future Work"),
  P("In FYP-II we will complete experiments E3 to E5, run the pre-specified statistical comparisons, and add explanations and calibration to the models (Section 7.4). Beyond this project, the same design could be tested on an external dataset to see whether the findings hold elsewhere, on grey-matter maps instead of intensity features, and on changes in brain age over time, which may say more about progression than a single scan."),
];

// ------------------------------------------------------------------ references (IEEE, order of first citation)
const REFS = [
  "World Health Organization, \"Dementia,\" Fact sheet. Accessed: Nov. 28, 2026. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/dementia",
  "R. C. Petersen et al., \"Practice guideline update summary: Mild cognitive impairment,\" Neurology, vol. 90, no. 3, pp. 126–135, 2018.",
  "K. Franke, G. Ziegler, S. Klöppel, and C. Gaser, \"Estimating the age of healthy subjects from T1-weighted MRI scans using kernel methods: Exploring the influence of various parameters,\" NeuroImage, vol. 50, no. 3, pp. 883–892, 2010.",
  "J. H. Cole and K. Franke, \"Predicting age using neuroimaging: Innovative brain ageing biomarkers,\" Trends in Neurosciences, vol. 40, no. 12, pp. 681–690, 2017.",
  "K. Franke and C. Gaser, \"Ten years of BrainAGE as a neuroimaging biomarker of brain aging: What insights have we gained?,\" Frontiers in Neurology, vol. 10, Art. no. 789, 2019.",
  "C. Gaser, K. Franke, S. Klöppel, N. Koutsouleris, and H. Sauer, \"BrainAGE in mild cognitive impaired patients: Predicting the conversion to Alzheimer's disease,\" PLoS ONE, vol. 8, no. 6, Art. no. e67346, 2013.",
  "T. T. Le et al., \"A nonlinear simulation framework supports adjusting for age when analyzing BrainAGE,\" Frontiers in Aging Neuroscience, vol. 10, Art. no. 317, 2018.",
  "J. H. Cole et al., \"Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker,\" NeuroImage, vol. 163, pp. 115–124, 2017.",
  "H. Peng, W. Gong, C. F. Beckmann, A. Vedaldi, and S. M. Smith, \"Accurate brain age prediction with lightweight deep neural networks,\" Medical Image Analysis, vol. 68, Art. no. 101871, 2021.",
  "V. M. Bashyam et al., \"MRI signatures of brain age and disease over the lifespan based on a deep brain network and 14 468 individuals worldwide,\" Brain, vol. 143, no. 7, pp. 2312–2324, 2020.",
  "S. M. Smith, D. Vidaurre, F. Alfaro-Almagro, T. E. Nichols, and K. L. Miller, \"Estimation of brain age delta from brain imaging,\" NeuroImage, vol. 200, pp. 528–539, 2019.",
  "A.-M. G. de Lange and J. H. Cole, \"Commentary: Correction procedures in brain-age prediction,\" NeuroImage: Clinical, vol. 26, Art. no. 102229, 2020.",
  "R. Cuingnet et al., \"Automatic classification of patients with Alzheimer's disease from structural MRI: A comparison of ten methods using the ADNI database,\" NeuroImage, vol. 56, no. 2, pp. 766–781, 2011.",
  "E. Moradi, A. Pepe, C. Gaser, H. Huttunen, and J. Tohka, \"Machine learning framework for early MRI-based Alzheimer's conversion prediction in MCI subjects,\" NeuroImage, vol. 104, pp. 398–412, 2015.",
  "J. Samper-González et al., \"Reproducible evaluation of classification methods in Alzheimer's disease: Framework and application to MRI and PET data,\" NeuroImage, vol. 183, pp. 504–521, 2018.",
  "J. Wen et al., \"Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation,\" Medical Image Analysis, vol. 63, Art. no. 101694, 2020.",
  "Alzheimer's Disease Neuroimaging Initiative, \"ADNI manuscript citations,\" rev. Apr. 3, 2024. Accessed: Nov. 28, 2026. [Online]. Available: https://adni.loni.usc.edu",
  "R. C. Petersen et al., \"Alzheimer's Disease Neuroimaging Initiative (ADNI): Clinical characterization,\" Neurology, vol. 74, no. 3, pp. 201–209, 2010.",
  "C. R. Jack Jr. et al., \"The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods,\" Journal of Magnetic Resonance Imaging, vol. 27, no. 4, pp. 685–691, 2008.",
  "N. J. Tustison et al., \"N4ITK: Improved N3 bias correction,\" IEEE Transactions on Medical Imaging, vol. 29, no. 6, pp. 1310–1320, 2010.",
  "V. Fonov et al., \"Unbiased average age-appropriate atlases for pediatric studies,\" NeuroImage, vol. 54, no. 1, pp. 313–327, 2011.",
  "B. C. Lowekamp, D. T. Chen, L. Ibáñez, and D. Blezek, \"The design of SimpleITK,\" Frontiers in Neuroinformatics, vol. 7, Art. no. 45, 2013.",
  "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
  "A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.",
  "E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, \"Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach,\" Biometrics, vol. 44, no. 3, pp. 837–845, 1988.",
  "X. Sun and W. Xu, \"Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves,\" IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389–1393, 2014.",
];
const references = [
  H1("References"),
  ...REFS.map((r, i) => new Paragraph({
    children: [new TextRun(`[${i + 1}]`), new TextRun({ text: "\t" + r })],
    tabStops: [{ type: TabStopType.LEFT, position: 600 }], indent: { left: 600, hanging: 600 },
    alignment: AlignmentType.LEFT, spacing: { line: 276, after: 100 },
  })),
];

// ------------------------------------------------------------------ appendices
const appendices = [
  H1("Appendices"),
  H2("Appendix A: Key decisions"),
  P("The project's decision log contains 36 dated decisions, each with its reason. Table 11 lists those that most affect the results."),
  tableCaption("Key decisions from the decision log"),
  table(["No.", "Decision", "Reason"], [
    ["7", "3 T scans only, for both cohorts", "Field strength differed between groups; BAG could reflect the scanner"],
    ["9", "Progression = dementia within 36 months; reverters excluded", "Common window in the literature; uncertain labels removed"],
    ["13", "N4, affine registration to MNI, no skull stripping or non-linear warping", "Keeps atrophy visible; fast on a laptop"],
    ["20, 21", "Duplicate enrolments removed; tilt check against a study mean image", "Prevents a person leaking between splits; catches misalignment"],
    ["30", "Bias correction fitted on out-of-fold training predictions only", "Training-only correction; in-sample predictions were over-fitted"],
    ["31", "Final brain-age model frozen with file hashes before the test set", "Shows the test result could not influence the choice"],
    ["32", "BAG group test fixed in advance, on training and validation only", "Keeps MCI test labels unused until E1–E5 are frozen"],
    ["33", "Same 4 mm voxel features in E1–E5; logistic regression primary", "Fair comparison; fixed before any MCI model was trained"],
    ["36", "E1 and E2 frozen and the E2 vs E1 test fixed before the test set", "Pre-specified test and interpretation"],
  ], W(0.55, 2.9, 2.9), { size: 18 }),
  H2("Appendix B: Reproducing the results"),
  P("Run from the repository root with the project environment active; settings are in the configs folder. Steps that use a test set run once and refuse a second run."),
  ...[
    "python -m src.cohort.build_cohorts  and  python -m src.cohort.make_splits",
    "python -m src.preprocessing.preprocess --scans <list>, then the QC and check scripts",
    "python -m src.brain_age.extract_features, train_baseline, train_cnn (GPU), bias_correction, predict",
    "python -m src.brain_age.bag_table  and  python -m src.brain_age.bag_stats",
    "python -m src.features.mci_features",
    "python -m src.progression.train --config configs/e1.yaml (and e2.yaml)",
    "python -m src.progression.evaluate --set test --confirm-test (done once)",
    "python -m pytest (16 tests)",
  ].map((c) => new Paragraph({ numbering: { reference: "numbers2", level: 0 }, spacing: { line: 276, after: 60 },
    children: [new TextRun({ text: c, font: "Consolas", size: 18 })] })),
  H2("Appendix C: Supervisor approvals"),
  P("The supervisor approved the following data-handling arrangements, recorded in the decision log on 23 November 2026: storing the preprocessed data in the university Google Drive (decision 27), training the CNN on Google Colab with the university account (decision 28), and with one personal account on condition that all ADNI data are deleted from it afterwards (decision 29); this deletion has been completed. Review of the cohort definition is still pending."),
];

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "FYP team", title: TITLE, description: "FYP-I Final Report",

  footnotes: { 1: { children: [new Paragraph({ children: [new TextRun({ text: ADNI_BYLINE, size: 16 })] })] } },
  styles: {
    default: { document: { run: { font: FONT, size: BODY_PT } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: NAVY },
        paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, keepLines: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: NAVY },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, keepLines: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: NAVY },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, keepLines: true, outlineLevel: 2 } },
      { id: "Caption", name: "caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 20, italics: true, color: "262626" },
        paragraph: { spacing: { before: 60, after: 200 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      children: cover },
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        pageNumbers: { formatType: NumberFormat.LOWER_ROMAN } } },
      headers: { default: header }, footers: { default: footer },
      children: [...approval, ...declaration, ...acknowledgements, ...abstract, ...toc, ...lof, ...lot, ...abbreviations] },
    { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      headers: { default: header }, footers: { default: footer },
      children: [...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...references, ...appendices] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log(`Saved ${path.relative(ROOT, OUT)}`);
});
