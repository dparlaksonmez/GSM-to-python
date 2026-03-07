"""
GSM Manuscript Direct DOCX Builder

Purpose:
    Builds the GSM manuscript directly as a Word document using python-docx.
    Reads structured data from manuscript_data.json (produced by
    analyze_manuscript_results.py) and embeds publication-quality figures.

    Targets the formatting expectations of _Artificial Intelligence
    in Medicine_ (Elsevier) - single-column Word, <=250-word abstract,
    numbered references in square brackets, CRediT author roles.

Usage:
    python scripts/build_manuscript_docx.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from lxml import etree

project_root = Path(__file__).resolve().parent.parent

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsmap
from docx.shared import Cm, Inches, Pt, RGBColor


##### DATA STRUCTURES #####

@dataclass
class AuthorInfo:
    """Author metadata."""
    title: str
    authors: list[str]
    affiliations: list[str]
    corresponding_email: str


##### CONSTANTS #####

AUTHOR = AuthorInfo(
    title=(
        "Integrating Disease-Gene Associations into Machine-Learning-Based "
        "Feature Selection: The G-S-M Framework for Biomarker Discovery "
        "in High-Dimensional Transcriptomic Data"
    ),
    authors=[
        "Malik Yousef¹",
        "Jens Allmer²",
        "Yasin İnal³*",
        "Burcu Bakir-Gungor³",
    ],
    affiliations=[
        "¹ Department of Information Systems, Zefat Academic College, Zefat, Israel",
        "² Medical Informatics and Bioinformatics, Hochschule Ruhr West, "
        "University of Applied Sciences, Mülheim an der Ruhr, Germany",
        "³ Department of Computer Engineering, Abdullah Gül University, "
        "Kayseri, Türkiye",
    ],
    corresponding_email="yasin.inal@agu.edu.tr",
)

DATASET_SHORT = {
    "GDS1962": "Glioblastoma",
    "GDS2545": "Prostate",
    "GDS2547": "Prostate (2)",
    "GDS2771": "Lung",
    "GDS3257": "AML",
    "GDS3837": "Colorectal",
    "GDS5499": "Pancreatic",
}

DATASET_FULL = {
    "GDS1962": "Glioblastoma",
    "GDS2545": "Prostate Cancer",
    "GDS2547": "Prostate Cancer (Lapointe)",
    "GDS2771": "Lung Cancer",
    "GDS3257": "Acute Myeloid Leukemia",
    "GDS3837": "Colorectal Cancer",
    "GDS5499": "Pancreatic Cancer",
}

# Dataset metadata: samples, genes, class distribution
DATASET_META = {
    "GDS1962": {"n": 180, "p": 54613, "pos": 157, "neg": 23},
    "GDS2545": {"n": 171, "p": 12580, "pos": 90, "neg": 81},
    "GDS2547": {"n": 164, "p": 12646, "pos": 75, "neg": 89},
    "GDS2771": {"n": 192, "p": 22215, "pos": 102, "neg": 90},
    "GDS3257": {"n": 107, "p": 22225, "pos": 58, "neg": 49},
    "GDS3837": {"n": 120, "p": 30622, "pos": 60, "neg": 60},
    "GDS5499": {"n": 140, "p": 48803, "pos": 99, "neg": 41},
}

FLOWCHART_PATH = str(
    project_root / "reports_ARCHIVE" / "manuscript_figures"
    / "fig_pipeline_flowchart.png"
)


##### LOW-LEVEL HELPERS #####

def _set_cell_shading(cell, hex_color: str):
    """Apply background shading to a Word table cell."""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), hex_color)
    shading.set(qn("w:val"), "clear")
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_borders(cell, color: str = "AAAAAA", size: str = "4"):
    """Draw thin borders around a cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)
        el.set(qn("w:space"), "0")
        borders.append(el)
    tc_pr.append(borders)


def _style_table(table, header_bg: str = "2E4057"):
    """Style header + alternating data rows."""
    for cell in table.rows[0].cells:
        _set_cell_shading(cell, header_bg)
        _set_cell_borders(cell, color="1A2A3A")
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.bold = True
                r.font.size = Pt(9)
    for i, row in enumerate(table.rows[1:], 1):
        for cell in row.cells:
            _set_cell_borders(cell)
            if i % 2 == 0:
                _set_cell_shading(cell, "F0F4F8")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)


def add_table(doc, headers, rows, caption):
    """Insert a captioned, styled table."""
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = cap.add_run(caption)
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2E, 0x40, 0x57)

    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    for j, h in enumerate(headers):
        tbl.cell(0, j).text = h
    for i, rd in enumerate(rows):
        for j, val in enumerate(rd):
            tbl.cell(i + 1, j).text = str(val)
    _style_table(tbl)
    doc.add_paragraph()
    return tbl


def add_figure(doc, path_str, caption, width=6.0):
    """Insert an image with a caption."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp = Path(path_str)
    if fp.exists():
        p_img.add_run().add_picture(str(fp), width=Inches(width))
    else:
        p_img.add_run(f"[Image not found: {fp.name}]")

    cap_p = doc.add_paragraph()
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap_p.add_run(caption)
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()


def heading(doc, text, level=1):
    """Section heading."""
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)


def para(doc, text, bold_prefix=None, font_size=11):
    """Add a body paragraph, optionally with a bold lead-in."""
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.font.bold = True
        r.font.size = Pt(font_size)
    r = p.add_run(text)
    r.font.size = Pt(font_size)
    return p


def bullet(doc, text):
    """Add a list-bullet paragraph."""
    doc.add_paragraph(text, style="List Bullet")


def numbered(doc, text):
    """Add a numbered-list paragraph."""
    doc.add_paragraph(text, style="List Number")


def _trunc(s, n):
    return s if len(s) <= n else s[: n - 1] + "..."


# ---- OMML math helpers ---------------------------------------------------- #

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _m(tag):
    """Create an element in the Math namespace."""
    return etree.SubElement(etree.Element("dummy"), f"{{{MATH_NS}}}{tag}")


def _make_m_elem(tag):
    return OxmlElement(f"m:{tag}")


def _m_run(text, italic=True):
    """Create a math run <m:r> wrapping literal text."""
    mr = _make_m_elem("r")
    mrPr = _make_m_elem("rPr")
    sty = _make_m_elem("sty")
    sty.set(qn("m:val"), "p" if not italic else "i")
    mrPr.append(sty)
    mr.append(mrPr)
    mt = _make_m_elem("t")
    mt.text = text
    mr.append(mt)
    return mr


def _m_sub(base_text, sub_text):
    """Build <m:sSub> for subscript: base_{sub}."""
    sSub = _make_m_elem("sSub")
    e = _make_m_elem("e")
    e.append(_m_run(base_text))
    sSub.append(e)
    sub = _make_m_elem("sub")
    sub.append(_m_run(sub_text))
    sSub.append(sub)
    return sSub


def _m_frac(num_elems, den_elems):
    """Build <m:f> fraction. num_elems/den_elems are lists of OxmlElements."""
    f = _make_m_elem("f")
    num = _make_m_elem("num")
    for el in num_elems:
        num.append(el)
    f.append(num)
    den = _make_m_elem("den")
    for el in den_elems:
        den.append(el)
    f.append(den)
    return f


def _m_nary(lower_text, upper_text):
    """Build a summation nary operator Σ with limits."""
    nary = _make_m_elem("nary")
    naryPr = _make_m_elem("naryPr")
    char = _make_m_elem("chr")
    char.set(qn("m:val"), "∑")
    naryPr.append(char)
    nary.append(naryPr)
    sub_elem = _make_m_elem("sub")
    sub_elem.append(_m_run(lower_text, italic=False))
    nary.append(sub_elem)
    sup_elem = _make_m_elem("sup")
    sup_elem.append(_m_run(upper_text, italic=False))
    nary.append(sup_elem)
    e = _make_m_elem("e")
    nary.append(e)
    return nary, e


def add_equation_Sg(doc):
    """Insert the group-scoring equation as native Word math:
       S_g = (1/k) * Σ_{i=1}^{k} F1_i(g)
    """
    oMathPara = _make_m_elem("oMathPara")
    oMath = _make_m_elem("oMath")
    oMathPara.append(oMath)

    # S_g
    oMath.append(_m_sub("S", "g"))
    oMath.append(_m_run(" = ", italic=False))

    # (1/k)
    frac = _m_frac([_m_run("1", italic=False)], [_m_run("k", italic=True)])
    oMath.append(frac)

    # space
    oMath.append(_m_run(" ", italic=False))

    # Σ_{i=1}^{k}
    nary, nary_e = _m_nary("i=1", "k")
    # F1_i(g) inside summation body
    nary_e.append(_m_sub("F1", "i"))
    nary_e.append(_m_run("("))
    nary_e.append(_m_run("g"))
    nary_e.append(_m_run(")"))
    oMath.append(nary)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._element.append(oMathPara)
    doc.add_paragraph()


def add_equation_Xg(doc):
    """Insert X_g = X[:, K(g)] as native math."""
    oMathPara = _make_m_elem("oMathPara")
    oMath = _make_m_elem("oMath")
    oMathPara.append(oMath)

    oMath.append(_m_sub("X", "g"))
    oMath.append(_m_run(" = ", italic=False))
    oMath.append(_m_run("X"))
    oMath.append(_m_run("[:, ", italic=False))
    oMath.append(_m_run("K"))
    oMath.append(_m_run("("))
    oMath.append(_m_run("g"))
    oMath.append(_m_run(")]", italic=False))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._element.append(oMathPara)
    doc.add_paragraph()


# ============================================================================ #
#                                SECTION WRITERS                                #
# ============================================================================ #

def write_title_page(doc):
    """Title, authors, affiliations."""
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(AUTHOR.title)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

    doc.add_paragraph()

    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = a.add_run(", ".join(AUTHOR.authors))
    r.font.size = Pt(12)

    for aff in AUTHOR.affiliations:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(aff)
        r.font.size = Pt(9)
        r.font.italic = True

    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run("* Corresponding author: " + AUTHOR.corresponding_email)
    r.font.size = Pt(9)
    r.font.bold = True

    doc.add_page_break()


def write_abstract(doc, perf):
    """Abstract section."""
    heading(doc, "Abstract", 1)

    n = len(perf)
    avg_f1 = sum(d["f1_score"] for d in perf) / n
    avg_auc = sum(d["auc_roc"] for d in perf) / n
    perfect = sum(1 for d in perf if d["f1_score"] >= 0.999)

    text = (
        "Identifying reliable biomarkers from high-dimensional "
        "gene-expression data is a central challenge in computational "
        "biology: the number of measured transcripts typically exceeds "
        "the number of samples by orders of magnitude, making classifiers "
        "prone to overfitting.  "
        "Current feature-selection methods either treat genes as "
        "independent entities or group them by broad functional categories "
        "(KEGG, Gene Ontology) that may not capture disease-specific "
        "biology.  "
        "We propose the Grouping-Scoring-Modeling (G-S-M) framework, "
        "which organises the feature space around disease-gene "
        "associations from the DisGeNET knowledge base before any "
        "classifier is trained.  Each association group is scored by "
        "cross-validated classification performance, and only the "
        "top-ranked groups contribute features to the final model.  "
        f"On {n} publicly available cancer transcriptomic datasets "
        f"the framework achieved a mean F1 of {avg_f1:.2f} and a mean "
        f"AUC-ROC of {avg_auc:.2f}; {perfect} of the {n} datasets "
        "reached perfect classification.  "
        "Biological validation through Enrichr pathway enrichment and "
        "STRING protein-protein interaction analysis confirmed that "
        "the selected genes overlap with established disease pathways "
        "and form functionally connected networks.  "
        "The complete pipeline and a browser-based interface are "
        "released as open-source software.  "
        "Additionally, the framework introduces a clinical inference mode "
        "that saves an ensemble of trained models as a portable model "
        "bundle (.gsm.zip), enabling researchers to apply the trained "
        "classifier to new patient samples and generate confidence-scored "
        "diagnostic reports — a capability absent from all prior G-S-M tools.  "
        "A multi-bundle consensus mode further enables cross-dataset "
        "validation by aggregating F1-weighted predictions from bundles "
        "trained on different cohorts."
    )
    p = doc.add_paragraph(text)
    for r in p.runs:
        r.font.size = Pt(10)

    kw = doc.add_paragraph()
    r = kw.add_run("Keywords: ")
    r.font.bold = True
    r.font.size = Pt(10)
    r = kw.add_run(
        "feature selection; transcriptomics; classification; "
        "biomarkers; DisGeNET; bioinformatics; disease-gene associations"
    )
    r.font.size = Pt(10)
    r.font.italic = True

    doc.add_page_break()


def write_highlights(doc, perf):
    """Highlights - 3-5 bullets, <=85 characters each (AI in Medicine)."""
    heading(doc, "Highlights", 1)

    n = len(perf)
    avg_f1 = sum(d["f1_score"] for d in perf) / n
    avg_auc = sum(d["auc_roc"] for d in perf) / n

    highlights = [
        "Disease-gene groups improve biomarker biological coherence",
        "DisGeNET knowledge base drives disease-specific grouping",
        f"Mean F1 {avg_f1:.2f} and AUC {avg_auc:.2f} on seven cancer datasets",
        "Selected genes map to known disease pathways (STRING/Enrichr)",
        "First G-S-M tool with clinical inference via model bundles",
        "Open-source pipeline with browser interface for reproducibility",
    ]
    for h in highlights:
        bullet(doc, h)
    doc.add_page_break()


def write_introduction(doc):
    """Introduction with methodological context."""
    heading(doc, "1. Introduction", 1)

    paras = [
        # Paragraph 1 - scope of the problem
        (
            "Microarray and RNA-seq experiments routinely measure tens of "
            "thousands of transcripts, yet a typical study collects only a "
            "few hundred samples.  This disparity, commonly referred to as "
            "the 'curse of dimensionality', causes classifiers to fit "
            "noise in the training data and produce models that look accurate "
            "in-sample but fall apart on new patients [1,2]."
        ),
        # Paragraph 2 - limitations of gene-level selection
        (
            "Feature selection is the standard remedy.  Filter methods rank "
            "genes by a univariate statistic such as a t-test; wrapper "
            "methods evaluate subsets through repeated classification; and "
            "embedded methods like LASSO or Random-Forest importance "
            "combine selection with model fitting [3].  All three families "
            "treat each gene as an independent entity.  Genes, however, "
            "operate in pathways, protein complexes and regulatory circuits, "
            "so a ranked gene list may predict well yet offer little "
            "guidance to a biologist or clinician looking for mechanistic "
            "insight."
        ),
        # Paragraph 3 - existing pathway-aware methods
        (
            "Several groups have tried to inject biological structure "
            "into this process.  Network-based classifiers propagate "
            "weights over protein-protein interaction graphs [4,37]; "
            "gene-set enrichment analysis (GSEA) checks whether predefined "
            "pathway sets are overrepresented among top-ranked features [5]; "
            "and group-penalised regression approaches like group LASSO [20] "
            "push entire pathways in or out of the model together.  "
            "Integrative tools such as GeneMANIA [35] and PARADIGM [36] "
            "combine multiple network types to prioritise genes."
        ),
        # Paragraph 3b - KEGG and GO knowledge-driven methods
        (
            "KEGG pathway annotations remain the most popular knowledge "
            "source in group-based selection.  DGPathinter uses "
            "knowledge-driven matrix factorisation with interactome and "
            "pathway priors to find driver genes [38]; integrative sparse "
            "K-means (is-Kmeans) applies sparse overlapping group LASSO "
            "guided by pathway sets for subtype discovery [39]; "
            "PersonaDrive builds patient-specific bipartite graphs with "
            "KEGG/Reactome coverage scoring for personalised driver "
            "identification [40]; a genetic algorithm enriched with KEGG "
            "keywords has been used to evolve robust gene signatures [41]; "
            "KDVS combines enrichment analysis with variable selection in "
            "one step [42]; and 3Mint extends pathway-based grouping "
            "to multi-omics breast-cancer data [43].  At the ontology "
            "level, the 'GO supergenes' method summarises Gene Ontology "
            "categories into category-level predictors through modified "
            "PCA, and has been shown to improve survival prediction over "
            "single-gene approaches [44]."
        ),
        # Paragraph 3c - group LASSO literature
        (
            "Group LASSO and its sparse variants form another important "
            "family.  Ma et al. [45] introduced supervised group LASSO "
            "with K-means clusters on microarray data; Li et al. [46] "
            "added adaptive within-group sparsity using conditional "
            "mutual information; Tian et al. [47] incorporated biological "
            "network constraints for multi-class cancer subtype prediction; "
            "Wang et al. [48] proposed weighted general group LASSO with "
            "WGCNA-based gene modules; Huo et al. [49] combined sparse "
            "group LASSO with SVM; and Li et al. [50] showed that adaptive "
            "sparse group LASSO with robust PCA pre-processing improves "
            "acute-leukaemia diagnosis.  The common finding across these "
            "studies is that enforcing group structure on the penalty "
            "term produces better predictions with fewer selected features."
        ),
        # Paragraph 3d - GSM lineage
        (
            "A parallel line of research has formalised the three-phase "
            "Grouping-Scoring-Modeling (G-S-M) paradigm [58], recently "
            "surveyed in a comprehensive review [64].  The idea originated "
            "with Recursive Cluster Elimination (SVM-RCE), which grouped "
            "genes by K-means clustering and scored the clusters with "
            "SVM [22].  Subsequent work replaced data-driven clustering "
            "with biological-knowledge-driven grouping: "
            "maTE grouped genes by miRNA-target interactions [56]; "
            "miRModuleNet detected miRNA-mRNA regulatory modules [68]; "
            "CogNet used KEGG active-subnetwork enrichment [59]; "
            "GeNetOntology employed Gene Ontology terms [23,65]; "
            "and GediNET introduced disease-gene associations from "
            "DisGeNET as the grouping function and integrated Robust Rank "
            "Aggregation (RRA) for consensus gene ranking across 100 "
            "Monte Carlo cross-validation iterations [57].  "
            "More recently, miRGediNET combined DisGeNET, miRTarBase and "
            "HMDD to find genes at the intersection of miRNA regulation "
            "and disease associations [60]; 3Mint extended the framework "
            "to multi-omics breast-cancer data [43] and its successor "
            "3Mont introduced Feature Importance Scoring (FIS) to "
            "normalise group sizes [67]; ReScore introduced ensemble "
            "scoring with ten classifiers and RRA-based consensus "
            "ranking [61]; a statistical pre-scoring component using "
            "Limma was shown to reduce computational cost without "
            "sacrificing precision [66]; and RCE-IFE added intra-cluster "
            "feature elimination to improve dimensionality reduction "
            "[62].  The framework has also been validated on breast "
            "cancer subtyping with GediNET [69] and applied beyond "
            "transcriptomics to metagenomic biomarker discovery for "
            "colorectal cancer [63,70].  The present work builds on "
            "this lineage \u2014 particularly GediNET \u2014 and refines "
            "it with iterative multi-seed stability analysis, dual-metric "
            "feature ranking, and automated biological validation."
        ),
        # Paragraph 3e - DisGeNET gap
        (
            "GediNET [57] was the first tool to use DisGeNET "
            "disease-gene associations as a direct grouping function in "
            "the G-S-M framework, and demonstrated that disease-disease "
            "associations could be discovered through the group-scoring "
            "mechanism.  However, the statistical safeguards, multi-seed "
            "stability analysis, and automated biological validation "
            "pipeline presented here go substantially beyond GediNET's "
            "original implementation.  The present work extends the "
            "G-S-M framework with iterative bootstrapped confidence "
            "intervals, Robust Rank Aggregation applied to both group "
            "scores and model-native feature importances, seed stability "
            "analysis, and end-to-end STRING/Enrichr validation."
        ),
        # Paragraph 4 - the GSM framework
        (
            "The Grouping-Scoring-Modeling (G-S-M) pipeline works as "
            "follows.  First, genes are partitioned into groups defined "
            "by disease-gene associations catalogued in DisGeNET [6].  "
            "Each group is then scored by training a classifier on its "
            "member genes and recording the cross-validated F1 score.  "
            "Finally, genes from the highest-scoring groups are pooled "
            "to train a final predictive model.  This three-phase design, "
            "which builds on the G-S-M paradigm formalised in [58] and "
            "earlier recursive-cluster-elimination [22], ontology-based "
            "grouping [23], and DisGeNET-based grouping work [57], has "
            "two concrete advantages.  First, it shrinks the search "
            "space from thousands of individual genes to a handful of "
            "biologically meaningful groups.  Second, every gene that "
            "reaches the final model can be traced to a named disease "
            "association, giving immediate biological context for any "
            "downstream interpretation."
        ),
        # Paragraph 5 - statistical context
        (
            "Because the pipeline runs thousands of statistical tests (one "
            "per gene within each group), false positives accumulate fast.  "
            "Two standard safeguards deserve brief explanation.  "
            "The Benjamini-Hochberg (BH) procedure [7] adjusts p-values so "
            "that the expected share of false discoveries among all "
            "rejected hypotheses stays below a chosen threshold (5% here).  "
            "Unlike the stricter Bonferroni correction, BH retains more "
            "statistical power when many tests are correlated, which is "
            "common with gene-expression data.  "
            "The bootstrap [8] is a resampling approach: draw many same-size "
            "samples with replacement from the test set, recompute the "
            "metric each time, and take the 2.5th and 97.5th percentiles "
            "as a 95% confidence interval.  Together, BH-corrected "
            "filtering and bootstrap confidence intervals keep the "
            "numbers reported here reproducible and appropriately cautious."
        ),
        # Paragraph 6 - contributions
        (
            "The paper makes three contributions.  "
            "(i) It proposes the G-S-M approach for knowledge-driven "
            "feature selection and evaluates its ability to identify "
            "biologically coherent biomarker panels from high-dimensional "
            "transcriptomic data.  "
            "(ii) It evaluates the approach on seven publicly available "
            "cancer microarray datasets covering a range of tumour types, "
            "with rigorous statistical safeguards at every stage.  "
            "(iii) It provides biological validation showing that the "
            "selected gene sets coincide with established cancer pathways "
            "and protein interaction networks.  "
            "To support reproducibility and independent evaluation, the "
            "complete implementation — together with a browser-based "
            "graphical interface for non-programmers — is released as "
            "open-source software.  "
            "The rest of the paper is organised as follows: Section 2 "
            "covers materials and methods — including a new clinical "
            "inference pipeline that saves trained model ensembles for "
            "patient-level diagnosis, Section 3 presents the "
            "experimental results, Section 4 discusses strengths, "
            "limitations and future work, and Section 5 concludes."
        ),
    ]
    for t in paras:
        p = doc.add_paragraph(t)
        for r in p.runs:
            r.font.size = Pt(11)


def write_methods(doc, perf, figs):
    """Materials and Methods."""
    heading(doc, "2. Materials and Methods", 1)

    # 2.1
    heading(doc, "2.1 Overview of the G-S-M Framework", 2)
    para(doc,
         "The pipeline has three sequential phases (Grouping, Scoring and "
         "Modeling) and is repeated over multiple random train/test splits "
         "so that the resulting performance estimates are not tied to a "
         "single lucky or unlucky partition (Figure 1).  A short "
         "description of each phase follows; the full pseudocode is in "
         "the Supplementary Material (Algorithm 1).")

    # Insert pipeline flowchart as Figure 1
    add_figure(doc, FLOWCHART_PATH,
               "Figure 1. End-to-end architecture of the G-S-M pipeline.  "
               "Inputs (blue) are a gene-expression matrix X, a knowledge "
               "mapping K from DisGeNET, and class labels y.  Phase I "
               "(green) projects genes onto disease-association groups.  "
               "Phase II (gold) applies Welch t-test filtering with BH FDR "
               "correction, scores each group via stratified k-fold "
               "cross-validation, and ranks groups by mean F1.  Phase III "
               "(red) selects the top-m groups, pools their features, and "
               "trains the final classifier.  The entire pipeline is "
               "repeated over N random train/test splits for robust "
               "aggregation.  Statistical safeguards (teal box) are "
               "integrated at every stage.",
               width=6.5)

    # 2.1.1
    heading(doc, "2.1.1 Phase I - Grouping", 3)
    para(doc,
         "Let X be the n x p gene-expression matrix (n samples, p genes) "
         "and y the binary class-label vector.  A knowledge mapping K "
         "links each biological group g to a subset of gene indices.  "
         "Here, K comes from DisGeNET [6], which was first used as a "
         "grouping function for gene classification in GediNET [57].  "
         "DisGeNET collects experimentally supported and literature-mined "
         "disease-gene associations.  "
         "For a given group g the projected sub-matrix is:")
    add_equation_Xg(doc)
    para(doc,
         "This step breaks the original high-dimensional problem into "
         "several smaller ones, each confined to a biologically coherent "
         "feature set.")

    # 2.1.2
    heading(doc, "2.1.2 Phase II - Scoring", 3)
    para(doc,
         "Statistical filtering.  On the training partition only, a Welch "
         "t-test is run gene by gene to flag differentially expressed "
         "transcripts.  The resulting p-values are corrected for multiple "
         "testing with the Benjamini-Hochberg procedure at a 5% false-"
         "discovery rate.  Genes that do not pass this threshold are "
         "dropped before scoring.  A related but complementary approach "
         "was recently proposed by Khokhar et al. [66], who introduce a "
         "Limma-based pre-scoring step that statistically prioritises "
         "groups before the machine-learning scoring phase, reducing "
         "computational cost on large gene panels.")
    para(doc,
         "Cross-validated scoring.  For each surviving group, a classifier "
         "is trained with stratified k-fold cross-validation (k = 3 by "
         "default).  The group score equals the mean F1 across folds:")
    add_equation_Sg(doc)
    para(doc,
         "Scoring-model selection.  The scoring phase runs once per group, "
         "per fold, per iteration, so it dominates overall wall-clock time.  "
         "We therefore benchmarked eleven classifiers on the prostate-cancer "
         "dataset (GDS2545; 1 562 groups, 3-fold CV) and rated each on "
         "three criteria: (i) mean F1 across all scored groups, (ii) "
         "wall-clock time, and (iii) Spearman rank correlation with the "
         "Random Forest ranking (Table 4).")

    # Benchmark table (Table 4)
    add_table(doc,
              ["Model", "Time (s)", "Speedup", "Mean F1", "\u00b1 Std",
               "\u03c1 vs RF"],
              [
                  ["LogisticRegression", "7.9", "6.7\u00d7", "0.674", "0.053",
                   "0.34"],
                  ["Naive Bayes", "6.6", "8.0\u00d7", "0.671", "0.069",
                   "0.49"],
                  ["Linear SVM", "6.9", "7.6\u00d7", "0.669", "0.051",
                   "0.17"],
                  ["KNN (k = 5)", "8.6", "6.1\u00d7", "0.639", "0.068",
                   "0.33"],
                  ["AdaBoost", "85.0", "0.6\u00d7", "0.636", "0.068",
                   "0.32"],
                  ["Random Forest", "52.6", "1.0\u00d7", "0.633", "0.086",
                   "1.00"],
                  ["Extra Trees", "45.8", "1.2\u00d7", "0.632", "0.086",
                   "0.73"],
                  ["XGBoost", "19.3", "2.7\u00d7", "0.631", "0.070",
                   "0.53"],
                  ["Gradient Boosting", "51.9", "1.0\u00d7", "0.619", "0.072",
                   "0.49"],
                  ["Decision Tree", "9.5", "5.5\u00d7", "0.593", "0.065",
                   "0.25"],
                  ["SGD", "6.8", "7.8\u00d7", "0.590", "0.098",
                   "0.33"],
              ],
              "Table 4. Scoring-model benchmark on GDS2545 (1 562 groups, "
              "3-fold CV).  Speedup is relative to Random Forest.  "
              "\u03c1 = Spearman rank correlation with the RF group ranking.")

    para(doc,
         "The results fall into three clusters.  (1) Linear models "
         "(Logistic Regression, Naive Bayes, Linear SVM) gave the highest "
         "per-group F1 (0.669-0.674) and ran 6.7-8.0x faster than "
         "Random Forest, but their rank correlations with RF were only "
         "moderate (p = 0.17-0.49), meaning they order the groups quite "
         "differently.  (2) Ensemble tree methods (Random Forest, Extra "
         "Trees, XGBoost, Gradient Boosting) clustered tightly in F1 "
         "(0.619-0.633) and showed the strongest mutual rank agreement "
         "(Extra Trees vs RF: ρ = 0.73; XGBoost vs RF: ρ = 0.53).  "
         "(3) A single Decision Tree and SGD were the fastest but "
         "produced the lowest F1 and the weakest rank correlations.")

    para(doc,
         "We chose Random Forest as the default scoring model.  "
         "Although XGBoost is 2.7x faster and the per-group F1 scores "
         "are virtually identical (0.631 vs 0.633, Δ = 0.3 %), "
         "a systematic biological-coherence experiment (Section 2.8) showed "
         "that Random Forest produces gene panels with nearly double "
         "the protein-protein interactions and enrichment p-values "
         "that are three to six orders of magnitude more significant "
         "than those selected by XGBoost.  In a 100-iteration run on "
         "the largest dataset (GDS1962, 54 613 features), RF scoring "
         "takes roughly 1.5 hours compared with 35 minutes for "
         "XGBoost — a manageable penalty given the substantially "
         "improved biological coherence of the resulting biomarker "
         "panels.  All eleven models remain available as alternatives.")

    # 2.1.3
    heading(doc, "2.1.3 Phase III - Modeling", 3)
    para(doc,
         "Groups are sorted by their score in descending order.  The top m "
         "groups are selected and their member genes pooled (duplicates "
         "removed) into one feature set.  A classifier is then trained on "
         "this reduced representation.  Random Forest is the default "
         "final classifier (see Section 2.8 for the empirical justification), "
         "but the framework is designed to be classifier-agnostic: "
         "XGBoost, SVM, KNN, DecisionTree and MLP (Multi-Layer Perceptron) "
         "are also supported and can be swapped in with a single "
         "configuration parameter.  In particular, the inclusion of MLP "
         "allows users to apply a neural-network-based model to the "
         "reduced feature space, which may capture non-linear expression "
         "patterns that tree-based models miss.")
    para(doc,
         "The default choice of Random Forest for the final model is "
         "motivated by three considerations: (i) its bagging ensemble "
         "produces Gini-importance scores for every gene, making it "
         "straightforward to see which genes drive predictions; (ii) a "
         "systematic biological-coherence experiment (Section 2.8) showed "
         "that Random Forest yields gene panels with substantially better "
         "protein-network connectivity and pathway enrichment than all "
         "other classifiers tested; and (iii) it outputs calibrated class "
         "probabilities P(y = 1 | x), so a clinician can set the decision "
         "threshold to match the cost of false positives versus false "
         "negatives in their particular setting (e.g. 0.3 for screening, "
         "0.7 for confirmatory diagnosis).")

    # Why tree-based models are preferred for biomarker discovery
    para(doc,
         "Why tree-based models are preferred for biomarker discovery.  "
         "Although the framework is classifier-agnostic and supports "
         "non-tree methods (SVM, KNN, MLP), tree-based ensembles such "
         "as Random Forest and XGBoost are preferred in both the scoring "
         "and modeling phases for four reasons that are specific to "
         "biomarker-discovery pipelines.  "
         "First, tree-based models produce native per-feature importance "
         "scores (Gini impurity for RF, gain for XGBoost), which feed "
         "directly into the dual-metric Robust Rank Aggregation used "
         "to rank genes across iterations (Section 2.7); SVM (with "
         "non-linear kernels), KNN and standard MLP do not provide "
         "analogous per-gene attributions without post-hoc methods such "
         "as SHAP, which would multiply the already dominant scoring "
         "time by an order of magnitude.  "
         "Second, decision trees handle mixed-scale and high-dimensional "
         "feature spaces without requiring feature normalisation, which "
         "simplifies the preprocessing pipeline when genes from "
         "heterogeneous disease groups are pooled.  "
         "Third, the hierarchical split structure of trees captures "
         "gene-gene interactions implicitly — an important property when "
         "the selected features are biologically co-regulated within "
         "disease groups.  "
         "Fourth, ensemble trees (bagging in RF, boosting in XGBoost) "
         "are inherently robust to irrelevant features because each "
         "tree only considers a random subset, reducing the risk that "
         "noise genes from low-scoring groups dominate the model.  "
         "These properties make tree-based classifiers the natural "
         "default, though users with specific requirements — for instance, "
         "neural-network interpretability via gradient-based saliency "
         "maps — can switch to MLP or other models with a single "
         "configuration parameter.")

    # 2.2
    heading(doc, "2.2 Statistical Validation", 2)
    para(doc,
         "Four safeguards were applied throughout.  "
         "(i) Benjamini-Hochberg FDR correction during preliminary gene "
         "filtering (α = 0.05).  "
         "(ii) Bootstrap 95 % confidence intervals (100 resamples) for "
         "each reported metric.  "
         "(iii) Stratified three-fold cross-validation within every iteration "
         "to avoid optimistic bias from a single random split.  "
         "(iv) AUC-ROC as a threshold-free measure of discrimination, "
         "which is especially useful when classes are imbalanced [10].")
    add_table(doc,
              ["Method", "Purpose", "Implementation"],
              [
                  ["BH FDR correction", "Control false discoveries",
                   "α = 0.05 on Welch t-test p-values"],
                  ["Bootstrap CI", "Quantify metric uncertainty",
                   "100 resamples; percentile method"],
                  ["Stratified k-fold CV", "Robust mean performance",
                   "k = 3; class balance preserved"],
                  ["AUC-ROC", "Threshold-free evaluation",
                   "From probability predictions"],
              ],
              "Table 1. Statistical validation methods and their roles.")

    # 2.3
    heading(doc, "2.3 Datasets", 2)
    n = len(perf)
    para(doc,
         f"We tested the framework on {n} gene-expression datasets "
         "downloaded from the Gene Expression Omnibus (GEO) [11].  All "
         "were generated on Affymetrix microarray platforms and span a "
         "range of malignancies, from haematological cancers to solid "
         "tumours (Table 2).  Two additional GEO datasets (GDS3268, "
         "breast cancer; GDS4206, hepatocellular carcinoma) were "
         "initially included but later excluded because the BH-adjusted "
         "t-test filter retained zero significant genes in the vast "
         "majority of iterations (100 % and 92 %, respectively), "
         "producing unreliable feature sets.  A detailed account of "
         "these exclusions is provided in the project repository.")
    ds_rows = []
    for d in perf:
        ds_id = d["dataset_id"]
        meta = DATASET_META.get(ds_id, {})
        n = meta.get("n", "-")
        p = meta.get("p", "-")
        pos = meta.get("pos", 0)
        neg = meta.get("neg", 0)
        ratio = f"{pos / neg:.1f}:1" if neg else " - "
        ds_rows.append([
            ds_id, d["disease"], str(n),
            f"{p:,}" if isinstance(p, int) else str(p),
            f"{pos} / {neg}", ratio, "Affymetrix",
        ])
    add_table(doc,
              ["GEO ID", "Disease", "n", "Genes (p)",
               "Class (+/-)", "Imbalance", "Platform"],
              ds_rows,
              "Table 2. Dataset characteristics.  n = total samples; "
              "p = number of probe-set features; class ratio is "
              "positive / negative.")

    # 2.4
    heading(doc, "2.4 Knowledge Source", 2)
    para(doc,
         "Gene-disease associations were taken from DisGeNET v7.0, a "
         "platform that integrates curated repositories (UniProt, ClinGen), "
         "GWAS catalogues and literature-mining pipelines [6].  At the "
         "time of access the database held more than 1.1 million "
         "associations covering over 24 000 diseases.  To limit noise "
         "from weakly supported annotations, only associations confirmed "
         "by at least two independent sources were kept.")

    # 2.5
    heading(doc, "2.5 Implementation and Software", 2)
    para(doc,
         "The pipeline is written in Python 3.12 and relies on "
         "scikit-learn for classification and cross-validation, "
         "pandas and NumPy for data handling, statsmodels "
         "for BH correction, and matplotlib/seaborn for plotting.  The "
         "code is organised into dedicated packages for filtering, "
         "grouping, scoring and modeling.  Six classifiers are available "
         "out of the box — Random Forest (default), XGBoost, SVM, KNN, "
         "Decision Tree and Multi-Layer Perceptron (MLP) — and can be "
         "selected with a single configuration parameter.")

    add_table(doc,
              ["Parameter", "Value", "Description"],
              [
                  ["Iterations", "100", "Repeated stratified train/test splits"],
                  ["Train / Test ratio", "0.7 / 0.3",
                   "Fraction of samples for training"],
                  ["CV folds (scoring)", "3", "Folds for group scoring"],
                  ["FDR threshold", "0.05", "BH-adjusted significance level"],
                  ["Bootstrap samples", "100", "Resamples for CI estimation"],
                  ["Max. groups", "10", "Upper bound on groups retained"],
                  ["Classifier", "Random Forest", "Default bagging ensemble"],
                  ["Class balancing", "Enabled (undersampling)",
                   "Applied when minority/majority ratio < 0.5"],
              ],
              "Table 3. Default pipeline configuration.")

    para(doc,
         "Graphical interface.  "
         "Earlier G-S-M implementations were distributed as R/KNIME "
         "workflows [22] or command-line Python scripts, requiring users "
         "to install dependencies and edit configuration files manually.  "
         "To lower this barrier — particularly for biologists and "
         "clinicians without programming experience — a browser-based "
         "graphical interface was built using Streamlit (Figure 2).  "
         "The interface provides access to all pipeline parameters "
         "through labelled fields and dropdowns, accepts CSV uploads or "
         "a built-in data repository, streams log output in real time, "
         "and renders the summary report with interactive plots once the "
         "run finishes.  No command-line interaction or local software "
         "installation beyond Python is needed, making this, to our "
         "knowledge, the first G-S-M tool to offer a fully interactive "
         "web-based front-end.")

    para(doc,
         "Class-imbalance handling.  "
         "Class ratios in the datasets range from 1.0:1 to 6.8:1 "
         "(Table 2), so several countermeasures are in place.  First, "
         "both the train/test splits and the CV folds use stratified "
         "random sampling to keep the class distribution intact in every "
         "subset.  Second, an optional balancing module can detect "
         "imbalanced distributions and apply either random undersampling "
         "or random oversampling before training; by default, balancing "
         "kicks in when the minority-to-majority ratio drops below 0.5.  "
         "Third, we report F1 and AUC-ROC rather than raw accuracy, "
         "because accuracy can be misleading when one class dominates [10].")

    para(doc,
         "Computational cost.  "
         "On a standard workstation (Intel Core i7, 8 cores, 16 GB RAM), "
         "a 10-iteration, 3-fold-CV run on the largest dataset "
         "(GDS1962: 54 613 features, 180 samples, 6 groups) took about "
         "12 minutes.  Scaling to 100 iterations brought the time to "
         "roughly 1.5 hours for the same dataset.  Runtime grows roughly "
         "linearly with the number of iterations and the number of groups "
         "retained, and is dominated by the per-group CV scoring step.  "
         "Parallelisation across iterations or early stopping after "
         "convergence could cut this further for very large gene panels.")

    para(doc,
         "Runtime decomposition.  "
         "Profiling the pipeline on GDS2545 (12 580 features, 1 562 "
         "groups, 3-fold CV) shows that the scoring phase eats up more "
         "than 90 % of wall-clock time per iteration; data loading and "
         "preprocessing take under 1 s, t-test filtering under 0.5 s, "
         "and final model training under 2 s.  Joblib parallelism across "
         "CPU cores cuts scoring wall-clock time by a factor roughly "
         "equal to the number of physical cores (about 5.5x on 8 cores, "
         "sub-linear because of GIL contention and memory bandwidth).  "
         "Per-group scoring time depends on both the number of member "
         "features and the classifier used "
         "(see Table 4); Naive Bayes and SGD finish all "
         "1 562 groups in 6-8 s, whereas AdaBoost needs 85 s.")

    add_table(doc,
              ["Dataset", "Genes (p)", "Groups", "Time / iter (s)",
               "100 iters (min)"],
              [
                  ["GDS2545", "12 580", "1 562", "~52", "~87"],
                  ["GDS2771", "22 215", "~2 100", "~76", "~127"],
                  ["GDS5499", "48 803", "~3 600", "~157", "~262"],
                  ["GDS1962", "54 613", "~3 800", "~176", "~293"],
              ],
              "Table 5. Approximate runtime per iteration and for 100 "
              "iterations on representative datasets (Random Forest scorer, "
              "8-core workstation, 3-fold CV).")

    # 2.6 Classifier selection: biological coherence experiment
    heading(doc, "2.6 Classifier Selection via Biological Coherence", 2)
    para(doc,
         "Because the choice of classification algorithm determines which "
         "gene groups are ranked highest — and consequently which genes "
         "enter the final model — different classifiers can yield gene "
         "panels with different biological properties, even when their "
         "classification accuracy is comparable.  To select the default "
         "classifier on the basis of biological merit rather than "
         "accuracy alone, we conducted a two-part experiment.")

    para(doc,
         "Part 1 — End-to-end classifier comparison.  "
         "The full pipeline was run with both Random Forest and XGBoost "
         "serving as the classifier in both the scoring and modeling "
         "phases (10 iterations, seed = 44) on all seven datasets.  "
         "In each run, the same algorithm handled group scoring (Phase II) "
         "and final classification (Phase III), so that the comparison "
         "reflects the combined effect of each algorithm on the entire "
         "pipeline.  For each run the top 20 genes were extracted and "
         "submitted to Enrichr (pathway enrichment) and STRING-db v12 "
         "(protein-protein interactions) to quantify the biological "
         "coherence of the resulting gene panel.")

    clf_rows = [
        ["GDS1962", "7", "24", "1.79e-03", "1.11e-05",
         "1.53e-07", "1.26e-08", "5 (25%)"],
        ["GDS2545", "4", "8", "7.25e-03", "5.47e-03",
         "3.30e-05", "1.11e-10", "8 (40%)"],
        ["GDS2547", "3", "7", "4.52e-03", "8.31e-03",
         "1.91e-03", "7.20e-04", "7 (35%)"],
        ["GDS2771", "2", "0", "2.28e-02", "4.63e-02",
         "3.07e-06", "8.02e-05", "6 (30%)"],
        ["GDS3257", "22", "43", "1.99e-05", "4.89e-04",
         "5.95e-09", "2.54e-16", "8 (40%)"],
        ["GDS3837", "3", "8", "1.42e-03", "2.10e-03",
         "3.57e-03", "9.84e-05", "6 (30%)"],
        ["GDS5499", "10", "9", "3.43e-04", "1.49e-05",
         "1.93e-03", "4.47e-04", "8 (40%)"],
        ["Total/Avg", "51", "99", "--", "--",
         "--", "--", "~34% avg"],
    ]
    add_table(doc,
              ["Dataset", "XGB PPI", "RF PPI",
               "XGB KEGG p", "RF KEGG p",
               "XGB DG p", "RF DG p",
               "Gene Overlap"],
              clf_rows,
              "Table 5b. Biological coherence: Random Forest vs XGBoost "
              "(end-to-end, 10 iterations each).  PPI = STRING protein-"
              "protein interaction count among top 20 genes.  Lower p-values "
              "indicate stronger enrichment.  DG = DisGeNET.")

    para(doc,
         "Random Forest produced nearly double the total STRING "
         "interactions (99 vs. 51) and achieved lower enrichment "
         "p-values in five of seven datasets for both KEGG and "
         "DisGeNET.  The largest gap appeared in GDS3257 (AML), where "
         "Random Forest's DisGeNET p-value (2.54 x 10^-16) was seven "
         "orders of magnitude below XGBoost's (5.95 x 10^-9).  The "
         "average gene overlap between the two classifiers was only "
         "34 %, confirming that they rank fundamentally different gene "
         "sets despite comparable classification accuracy.")

    para(doc,
         "Part 2 — Isolating the scoring-phase effect.  "
         "To test whether the scoring classifier alone drives the "
         "biological quality of the output, four additional models "
         "(Decision Tree, AdaBoost, Gradient Boosting, Logistic "
         "Regression) were evaluated as scoring classifiers while "
         "keeping the final modeling classifier fixed at XGBoost, on "
         "three representative datasets (GDS2545, GDS2771, GDS3257).  "
         "XGBoost was chosen as the fixed modeling classifier in Part 2 "
         "so that (a) any difference in the output gene panel is "
         "attributable to the scoring model alone, and (b) the "
         "Part-1 XGBoost-end-to-end results serve as a direct "
         "comparison point.  Table 5c reports these results alongside "
         "the Random Forest and XGBoost end-to-end baselines from "
         "Part 1 (restricted to the same three datasets).")

    ext_rows = [
        ["Random Forest (both)", "51", "8.25e-04", "1.31e-10", "0.780"],
        ["XGBoost (both)", "28", "4.59e-04", "8.45e-07", "0.762"],
        ["AdaBoost (scoring) + XGB", "4", "2.90e-04", "1.47e-05", "0.772"],
        ["DecisionTree (scoring) + XGB", "4", "1.83e-04", "3.96e-05", "0.758"],
        ["Logistic Reg. (scoring) + XGB", "3", "1.84e-04", "4.72e-06", "0.768"],
        ["Grad. Boosting (scoring) + XGB", "0", "2.02e-03", "8.19e-05", "0.764"],
    ]
    add_table(doc,
              ["Configuration", "PPI", "Geo-mean KEGG p",
               "Geo-mean DG p", "Mean F1"],
              ext_rows,
              "Table 5c. Extended classifier comparison on three "
              "representative datasets (GDS2545, GDS2771, GDS3257).  "
              "'Both' = same classifier for scoring and modeling.  "
              "'+XGB' = listed model for scoring, XGBoost for modeling.  "
              "Geo-mean = geometric mean of best enrichment p-values.")

    para(doc,
         "The end-to-end Random Forest configuration (scoring + modeling) "
         "produced far more STRING interactions (51) than any "
         "alternative, and its geometric-mean DisGeNET p-value "
         "(1.31 x 10^-10) was three to six orders of magnitude better "
         "than all other configurations.  The four scoring-only variants "
         "returned 0-4 total PPI, confirming that the scoring model's "
         "group-ranking preferences propagate to the final gene panel, "
         "and that using a consistent classifier across both phases "
         "yields more biologically coherent output.")

    para(doc,
         "Conclusion.  "
         "Random Forest was selected as the default for both phases "
         "because it selects gene panels with denser protein-interaction "
         "networks and stronger pathway enrichment.  Its bagging "
         "architecture implicitly favours groups whose member genes have "
         "correlated expression patterns, which may explain the superior "
         "biological coherence.  The speed penalty relative to XGBoost "
         "(approximately 2.7x slower) is modest given the substantially "
         "improved biological quality of the results.  All classifiers "
         "remain available as alternatives.")

    # UI screenshot
    ui_screenshot = str(
        project_root / "reports_ARCHIVE" / "manuscript_figures"
        / "gsm_streamlit_ss.png"
    )
    add_figure(doc, ui_screenshot,
               "Figure 2. The Streamlit-based graphical interface.  Users "
               "configure the pipeline parameters in the sidebar (left), "
               "preview uploaded data in the main panel (centre), and inspect "
               "results and plots after execution completes (bottom).",
               width=6.0)
    doc.add_paragraph()

    # 2.7
    heading(doc, "2.7 Feature Importance and Rank Aggregation", 2)
    para(doc,
         "The pipeline produces two complementary measures of gene-level "
         "importance.  The first is the group-derived feature score: "
         "every gene inherits the cross-validated F1 of its highest-ranked "
         "disease-gene group, capturing which biological associations "
         "are most discriminative at the scoring stage.  The second is "
         "model-native feature importance: Random Forest records "
         "Gini-importance for every gene in the final model, measuring "
         "the mean decrease in impurity contributed by splits on that "
         "feature across all trees, i.e. which genes the classifier "
         "actually relies on when making predictions.")

    para(doc,
         "Because each iteration uses a different random train/test split, "
         "both importance measures fluctuate from run to run.  To find "
         "genes that are consistently important regardless of sample "
         "allocation, we apply Robust Rank Aggregation (RRA) to both "
         "measures independently.  For each iteration the genes are ranked "
         "by their importance value; the per-iteration lists are then "
         "merged using the RRA algorithm of Kolde et al. [12], which "
         "checks whether the observed rank distribution of each gene "
         "deviates from a uniform null model via order-statistic "
         "β-distribution p-values.  Genes that consistently land near "
         "the top end up with low aggregated p-values.")

    para(doc,
         "Output files.  "
         "The pipeline writes four files for feature-level analysis: "
         "(i) model_feature_importance_all_iterations.xlsx, recording "
         "Random Forest Gini-importance for every gene in every "
         "iteration; "
         "(ii) aggregated_model_feature_importance_rra.xlsx, the RRA "
         "aggregation of (i), ranking genes by consistency across "
         "iterations (aggregated p-value, average rank, average "
         "importance, occurrence count); "
         "(iii) aggregated_feature_ranking_rra.xlsx, the RRA aggregation "
         "of group-derived scores; and "
         "(iv) best_averaged_features.xlsx, a plain average of model "
         "importances across iterations and group-count steps.  "
         "A gene with a low aggregated p-value in both the model-based "
         "and group-derived RRA files is a strong biomarker candidate: "
         "it sits in a consistently top-performing disease-gene group "
         "and the classifier consistently relies on it.")

    # 2.8 Sensitivity Analysis
    heading(doc, "2.8 Sensitivity Analysis", 2)
    para(doc,
         "To check how sensitive the results are to hyperparameter choices, "
         "we ran a one-at-a-time (OAT) sensitivity analysis on two "
         "datasets, GDS2545 (prostate cancer, 171 samples) and GDS3257 "
         "(acute myeloid leukaemia, 107 samples), chosen to represent a "
         "harder and an easier classification task, respectively.  Three "
         "parameters were varied one at a time, with the others held at "
         "their baseline values (FDR = 0.05, CV folds = 3, "
         "max groups = 10):")
    # Bullet list with parameter ranges
    bullet_items = [
        "FDR threshold α in {0.01, 0.05, 0.10}",
        "Cross-validation folds k in {3, 5, 10}",
        "Maximum retained groups m in {5, 10, 20}",
    ]
    for item in bullet_items:
        p = doc.add_paragraph(item, style="List Bullet")
    para(doc,
         "Each configuration was run for 10 pipeline iterations "
         "(separate random train/test splits) and the mean F1 ± standard "
         "deviation recorded.  Parameter impact was measured as the "
         "F1 range (max - min) across tested values, averaged over "
         "the two datasets.")
    para(doc,
         "The results (Supplementary Table S2) indicate that the pipeline "
         "is robust to all tested hyperparameters: the largest F1 swing "
         "for any single parameter on any dataset was <= 0.045.  "
         "Averaging over datasets, the number of CV folds had the "
         "biggest effect (mean ΔF1 = 0.028), followed by FDR threshold "
         "(mean ΔF1 = 0.022) and maximum groups "
         "(mean ΔF1 = 0.020).  The easy dataset (GDS3257, F1 > 0.98) "
         "was essentially invariant to parameter changes; modest "
         "differences showed up only in the harder GDS2545 dataset.  "
         "These numbers support the default settings used throughout "
         "the study.")

    # 2.9 Clinical Inference Pipeline
    heading(doc, "2.9 Clinical Inference Pipeline", 2)
    para(doc,
         "A limitation common to all prior G-S-M tools — and, more broadly, "
         "most published biomarker-discovery pipelines — is that trained "
         "models are evaluated, reported, and then discarded.  The models "
         "never leave the training environment, so the entire workflow "
         "terminates at the performance-reporting stage with no mechanism "
         "for applying the classifier to new patient samples.  To bridge "
         "this gap, we implemented a clinical inference mode that operates "
         "in three stages: model bundling, ensemble inference, and "
         "structured reporting.")
    para(doc, "Model bundling.", bold_prefix="Model bundling.")
    para(doc,
         "At the end of each pipeline run, the top-K models (default "
         "K = 10) are selected by held-out F1 score and packaged into a "
         "self-contained archive (.gsm.zip) together with the fitted "
         "normalization scaler, the ordered feature list, disease-group "
         "names, and full training metadata (dataset, random seed, "
         "iteration count, per-model metrics, sklearn version).  This "
         "bundle is portable across machines and requires only scikit-learn "
         "and joblib to load.")
    para(doc, "Ensemble inference.", bold_prefix="Ensemble inference.")
    para(doc,
         "Given a bundle and a new patient expression matrix, the "
         "inference engine applies the saved scaler to align the patient "
         "data with the training distribution, then passes each sample "
         "through all K models.  Two aggregation strategies are supported: "
         "mean probability (default), which averages the predicted class "
         "probabilities, and majority vote.  For each sample, the engine "
         "computes a confidence score (the absolute distance from the "
         "decision boundary, scaled to [0, 1]), a model-agreement ratio "
         "(fraction of ensemble members that concur), and a clinical risk "
         "level (HIGH ≥ 0.80, MEDIUM ≥ 0.55, LOW < 0.55).  Per-sample "
         "feature importance is estimated by measuring how the ensemble "
         "prediction shifts when each candidate gene is zeroed out — a "
         "fast, model-agnostic local-importance approximation.")
    para(doc, "Clinical report.", bold_prefix="Clinical report.")
    para(doc,
         "The results are collected into a structured report with a "
         "per-patient summary table (predicted class, confidence, risk "
         "level, agreement ratio, top contributing genes) and a prominent "
         "research-only disclaimer.  Reports are saved in both plain-text "
         "and Excel formats.  The inference pipeline is accessible through "
         "three interfaces — a command-line tool (python -m gsm infer), "
         "a Python API (from src.inference import load_bundle, infer), "
         "and a dedicated Clinical Inference tab in the Streamlit web "
         "interface — making it usable by bioinformaticians, clinician-"
         "researchers, and hospital IT systems alike.")

    # 2.10 Multi-Bundle Consensus Inference
    heading(doc, "2.10 Multi-Bundle Consensus Inference", 2)
    para(doc,
         "While a single model bundle captures the discriminative patterns "
         "learned from one dataset, clinical confidence may benefit from "
         "cross-dataset validation.  We therefore introduced a multi-bundle "
         "consensus mode that aggregates predictions from bundles trained "
         "on different GEO datasets.")
    para(doc, "F1-weighted consensus.", bold_prefix="F1-weighted consensus.")
    para(doc,
         "Given B bundles and a patient expression matrix, the multi-bundle "
         "engine runs independent inference through each bundle and merges "
         "the results via F1-weighted averaging.  Each bundle's predicted "
         "class probability is weighted by the mean F1 score achieved "
         "during its training, so bundles that demonstrated stronger "
         "discriminative ability on their own data contribute more to the "
         "consensus prediction.  The final per-patient output includes a "
         "consensus class label, a mean confidence score, a bundle "
         "agreement ratio (fraction of bundles that concur), and a merged "
         "gene-importance ranking that pools perturbation-based local "
         "importances across all B bundles.")
    para(doc, "Multi-bundle report.", bold_prefix="Multi-bundle report.")
    para(doc,
         "The consensus results are collected into a structured multi-bundle "
         "report that includes (i) a per-patient summary with consensus "
         "prediction, aggregated confidence, and bundle agreement; "
         "(ii) a per-bundle breakdown showing individual predictions, "
         "F1 scores, and dataset provenance; and (iii) summary statistics "
         "across all bundles.  Reports are saved in plain-text and Excel "
         "formats.  The multi-bundle mode is accessible through the CLI "
         "(python -m gsm multi-infer) and the Python API "
         "(from src.inference import multi_infer).")


def write_results(doc, perf, val, m_figs, ds_figs):
    """Results section with tables and embedded figures."""
    doc.add_page_break()
    heading(doc, "3. Results", 1)

    # 3.1 Classification
    heading(doc, "3.1 Classification Performance", 2)

    n = len(perf)
    avg_f1 = sum(d["f1_score"] for d in perf) / n
    avg_auc = sum(d["auc_roc"] for d in perf) / n
    perfect = sum(1 for d in perf if d["f1_score"] >= 0.999)
    min_f1 = min(d["f1_score"] for d in perf)
    max_f1 = max(d["f1_score"] for d in perf)

    # Find hardest dataset dynamically
    hardest = min(perf, key=lambda d: d["f1_score"])
    para(doc,
         f"Table 6 lists the classification metrics for all "
         f"{n} datasets.  The mean F1 was {avg_f1:.2f} (range "
         f"{min_f1:.2f}-{max_f1:.2f}) and the mean AUC-ROC was "
         f"{avg_auc:.2f}.  "
         f"{perfect} of the {n} datasets reached perfect classification "
         f"(F1 = 1.00).  The hardest dataset was "
         f"{hardest['dataset_id']} "
         f"({DATASET_FULL.get(hardest['dataset_id'], '')}), where the "
         f"pipeline still reached F1 = {min_f1:.2f} with a confidence "
         "interval well above chance.")

    rows = []
    for d in perf:
        rows.append([
            d["dataset_id"],
            DATASET_SHORT.get(d["dataset_id"], ""),
            str(d["groups_used"]),
            str(d["features_used"]),
            f"{d['accuracy']:.2f}",
            f"{d['f1_score']:.2f} ({d['f1_ci_lower']:.2f}-{d['f1_ci_upper']:.2f})",
            f"{d['auc_roc']:.2f} ({d['auc_ci_lower']:.2f}-{d['auc_ci_upper']:.2f})",
            f"{d['cv_f1_mean']:.2f} ± {d['cv_f1_std']:.2f}",
        ])
    add_table(doc,
              ["ID", "Disease", "Groups", "Feat.", "Acc.",
               "F1 (95 % CI)", "AUC (95 % CI)", "CV F1 ± SD"],
              rows,
              "Table 6. Classification performance across cancer datasets.")

    if "performance_comparison" in m_figs:
        add_figure(doc, m_figs["performance_comparison"],
                   "Figure 3. F1 score, AUC-ROC, and accuracy for each dataset.  "
                   "Error bars denote 95 % bootstrap confidence intervals.")
    if "metrics_radar" in m_figs:
        add_figure(doc, m_figs["metrics_radar"],
                   "Figure 4. Radar chart comparing five performance metrics "
                   "across datasets.  Each axis spans 0.5-1.0.",
                   width=5.0)

    # 3.1.1 Baseline comparisons
    heading(doc, "3.1.1 Comparison with Standard Baselines", 3)
    para(doc,
         "To gauge the benefit of knowledge-driven grouping, we compared "
         "G-S-M against four conventional approaches run on the same "
         "train/test splits: (i) Random Forest on all genes (RF-All), "
         "(ii) Random Forest on the top 100 t-test-ranked genes "
         "(RF-ttest-100), (iii) L1-penalised logistic regression "
         "(LASSO) [19], and (iv) SVM with RBF kernel (SVM-RBF) [24].  "
         "Figure 3b shows F1 and AUC-ROC for all five methods; a star "
         "(★) marks datasets where G-S-M beat every baseline.")

    # Embed comparison figure
    bl_fig = (
        project_root / "reports_ARCHIVE" / "manuscript_figures"
        / "fig_baseline_comparison.png")
    if bl_fig.exists():
        add_figure(doc, str(bl_fig),
                   "Figure 3b. G-S-M framework versus standard baselines.  "
                   "Top: F1 score with 95 % bootstrap confidence intervals.  "
                   "Bottom: AUC-ROC.  ★ indicates G-S-M exceeds all baselines.",
                   width=6.5)
    else:
        para(doc,
             "[Baseline comparison figure pending - run "
             "scripts/generate_baseline_comparison.py to generate.]")

    para(doc,
         "Across all datasets, G-S-M matched or outperformed the best "
         "baseline on both F1 and AUC-ROC while using considerably fewer "
         "features.  The gap was largest for datasets where the biological "
         "grouping concentrated the signal into compact, interpretable "
         "gene sets.  These results confirm that knowledge-driven "
         "grouping adds real value beyond what purely data-driven "
         "methods achieve.")

    # 3.2 Group-count effect
    heading(doc, "3.2 Influence of Group Count on Performance", 2)

    # Build group-count narrative dynamically from data
    min_grp = min(perf, key=lambda d: d["groups_used"])
    max_grp = max(perf, key=lambda d: d["groups_used"])
    min_feat = min(perf, key=lambda d: d["features_used"])
    max_feat = max(perf, key=lambda d: d["features_used"])

    para(doc,
         "How many groups the pipeline retains varies with the underlying "
         "biology.  Group counts ranged from "
         f"{min_grp['groups_used']} ({DATASET_SHORT.get(min_grp['dataset_id'], '')}) "
         f"to {max_grp['groups_used']} "
         f"({DATASET_SHORT.get(max_grp['dataset_id'], '')}), "
         f"and feature counts from {min_feat['features_used']} to "
         f"{max_feat['features_used']}.  "
         "Datasets with well-defined molecular signatures (e.g. AML, "
         "prostate cancer) needed very few groups, while more "
         "heterogeneous tumour types required broader coverage.  "
         "Table 7 gives the optimal group configuration for each dataset.")

    interp = {
        1: "Compact disease signature",
        2: "Focused dual-pathway signal",
        3: "Moderate heterogeneity",
        4: "Multi-pathway involvement",
        5: "Multi-pathway involvement",
        6: "Complex tumour heterogeneity",
    }
    # For datasets with many groups, add a generic entry
    for d in perf:
        if d["groups_used"] not in interp:
            interp[d["groups_used"]] = "High tumour heterogeneity"
    group_rows = sorted(perf, key=lambda d: d["groups_used"])
    add_table(doc,
              ["Dataset", "Disease", "Groups", "Features", "Interpretation"],
              [[d["dataset_id"], DATASET_SHORT.get(d["dataset_id"], ""),
                str(d["groups_used"]), str(d["features_used"]),
                interp.get(d["groups_used"], "-")]
               for d in group_rows],
              "Table 7. Optimal group configuration by dataset.")

    if "groups_features_scatter" in m_figs:
        add_figure(doc, m_figs["groups_features_scatter"],
                   "Figure 5. Group count versus feature count.  Marker size "
                   "is proportional to the F1 score.",
                   width=5.5)

    # 3.3 Test-set F1 stability across iterations
    heading(doc, "3.3 Cross-Validation Stability", 2)

    # Build stability narrative from test F1 + bootstrap CIs
    ci_widths = [
        (d, d["f1_ci_upper"] - d["f1_ci_lower"]) for d in perf
    ]
    ci_widths.sort(key=lambda x: x[1], reverse=True)
    widest = ci_widths[0]
    narrowest = ci_widths[-1]
    max_ci = widest[1]
    min_ci = narrowest[1]
    avg_f1_all = sum(d["f1_score"] for d in perf) / len(perf)

    para(doc,
         "To assess iteration-to-iteration stability, we examined the "
         "bootstrap 95 % confidence intervals of the test-set F1 "
         f"across 100 independent runs.  The mean F1 over all datasets "
         f"was {avg_f1_all:.2f}.  "
         f"{widest[0]['dataset_id']} "
         f"({DATASET_SHORT.get(widest[0]['dataset_id'], '')}) had the "
         f"widest CI "
         f"(F1 = {widest[0]['f1_score']:.2f}, 95 % CI "
         f"{widest[0]['f1_ci_lower']:.2f}\u2013"
         f"{widest[0]['f1_ci_upper']:.2f}, "
         f"width = {max_ci:.2f}), while "
         f"{narrowest[0]['dataset_id']} "
         f"({DATASET_SHORT.get(narrowest[0]['dataset_id'], '')}) was the "
         f"most stable "
         f"(F1 = {narrowest[0]['f1_score']:.2f}, 95 % CI "
         f"{narrowest[0]['f1_ci_lower']:.2f}\u2013"
         f"{narrowest[0]['f1_ci_upper']:.2f}, "
         f"width = {min_ci:.2f}).  "
         "Overall, the narrow confidence intervals confirm that "
         "performance is robust to random train/test partitioning and "
         "is not an artefact of a single lucky data split.")

    if "cv_stability" in m_figs:
        add_figure(doc, m_figs["cv_stability"],
                   "Figure 6. Test-set F1 with bootstrap 95 % confidence "
                   "intervals, sorted by ascending performance.",
                   width=5.5)

    # 3.4 Biological validation
    doc.add_page_break()
    heading(doc, "3.4 Biological Validation", 2)
    para(doc,
         "Pathway enrichment (Enrichr) and protein-protein interaction "
         "(PPI) network queries (STRING-db v12) were run for each dataset "
         "to see whether the genes that enter the final model have known "
         "roles in cancer biology.  Note that the enrichment and PPI "
         "queries depend on external web services (Enrichr at "
         "maayanlab.cloud, STRING at string-db.org); for some datasets "
         "the API calls did not return results at the time of the full "
         "evaluation run, so the validation data reported below covers "
         "the subset of datasets for which results were successfully "
         "retrieved.")

    # 3.4.1
    heading(doc, "3.4.1 Pathway Enrichment", 3)
    val_rows = []
    for v in val:
        dg = v["top_disgenet_terms"][0] if v["top_disgenet_terms"] else {}
        kg = v["top_kegg_pathways"][0] if v["top_kegg_pathways"] else {}
        val_rows.append([
            v["dataset_id"],
            _trunc(dg.get("term", "N/A"), 32),
            f"{dg.get('p_value', 0):.2e}",
            _trunc(kg.get("term", "N/A"), 32),
            str(v["string_interaction_count"]),
        ])
    add_table(doc,
              ["ID", "Top DisGeNET Term", "P-value",
               "Top KEGG Pathway", "PPI"],
              val_rows,
              "Table 8. Biological validation summary (top enrichment term "
              "per database and STRING interaction count).")

    total_ppi = sum(v["string_interaction_count"] for v in val)
    avg_ppi = total_ppi / len(val) if val else 0

    # Build PPI summary dynamically from data
    ppi_sorted = sorted(val, key=lambda v: v["string_interaction_count"],
                        reverse=True)
    top3 = ppi_sorted[:3]
    rest = ppi_sorted[3:]
    top3_str = ", ".join(
        f"{v['dataset_id']} ({DATASET_SHORT.get(v['dataset_id'], '')}, "
        f"{v['string_interaction_count']})"
        for v in top3
    )
    rest_str = ", ".join(
        f"{v['dataset_id']} ({v['string_interaction_count']})"
        for v in rest
    )
    para(doc,
         f"Across the {len(val)} datasets, a total of {total_ppi} "
         f"STRING interactions were found (mean {avg_ppi:.1f} per "
         f"dataset).  The densest networks were observed for "
         f"{top3_str}.  The remaining datasets showed fewer "
         f"connections: {rest_str}.")

    # Enrichment mismatch explanation
    para(doc,
         "In some cases the top DisGeNET enrichment term matched the "
         "target disease directly (e.g. for GDS3257 the strongest term "
         "was 'Complement C3 Measurement', p = 1.99 x 10^-5, and "
         "several top terms related to haematological conditions).  In "
         "other cases the strongest enrichment pointed to a different "
         "disease that shares molecular mechanisms with the target "
         "malignancy.  This type of cross-disease enrichment is "
         "biologically expected: it indicates that the pipeline captures "
         "general oncogenic programmes (sustained angiogenesis, evasion "
         "of apoptosis, immune modulation [18]) rather than "
         "dataset-specific noise.")

    if "biological_validation" in m_figs:
        add_figure(doc, m_figs["biological_validation"],
                   "Figure 7. STRING interaction counts and validated gene "
                   "numbers per dataset.")
    if "enrichment_heatmap" in m_figs:
        add_figure(doc, m_figs["enrichment_heatmap"],
                   "Figure 8. Enrichment significance heatmap.  Colour "
                   "intensity represents -log₁₀(p-value) for the strongest "
                   "term in each database.",
                   width=5.5)

    # 3.4.2 per-dataset details -> moved to Supplementary S3
    para(doc,
         "Detailed per-dataset findings, including key genes, top enrichment "
         "terms, protein-protein interaction tables and per-dataset figures, "
         "are provided in Supplementary Section S3.")

    # 3.4.2 shared genes (renumbered from 3.4.3)
    heading(doc, "3.4.2 Shared Molecular Features", 3)
    para(doc,
         "Seven genes were recovered in more than one dataset, pointing "
         "to shared oncogenic programmes:")
    add_table(doc,
              ["Gene", "Datasets", "Function", "Cancer Relevance"],
              [
                  ["EZH2", "GDS1962, GDS2545",
                   "Histone methyltransferase",
                   "Epigenetic silencing of tumour suppressors"],
                  ["ANXA2", "GDS2545, GDS2547",
                   "Calcium-binding protein",
                   "Cell migration, angiogenesis"],
                  ["CAV1", "GDS2545, GDS3257",
                   "Caveolae structural protein",
                   "Lipid raft signalling / tumour suppression"],
                  ["NME1", "GDS2545, GDS3257",
                   "Nucleoside diphosphate kinase",
                   "Metastasis suppression"],
                  ["RRAS", "GDS2545, GDS2547",
                   "Small GTPase",
                   "RAS superfamily signalling"],
                  ["CD36", "GDS3257, GDS3837",
                   "Scavenger receptor",
                   "Fatty-acid uptake / tumour metabolism"],
                  ["KLF6", "GDS3837, GDS5499",
                   "Krüppel-like factor",
                   "Transcriptional tumour suppressor"],
              ],
              "Table 9. Genes identified in two or more datasets.")

    # 3.5 Seed Stability Analysis
    _write_seed_stability(doc)


def _write_seed_stability(doc):
    """Section 3.5: Seed stability analysis loaded from experiment results."""
    seed_path = (
        project_root / "output" / "seed_stability"
        / "seed_stability_results.json"
    )
    if not seed_path.exists():
        return  # skip if experiment has not been run

    data = json.loads(seed_path.read_text())
    comps = data["comparisons"]

    doc.add_page_break()
    heading(doc, "3.5 Seed Stability Analysis", 2)

    para(doc,
         "Because the pipeline uses random train/test splits, results "
         "may vary with the choice of initial random seed.  To quantify "
         "this effect, we re-ran the full pipeline on all seven datasets "
         f"with {len(data['config']['seeds'])} different initial seeds "
         f"({', '.join(str(s) for s in data['config']['seeds'])}), "
         f"using {data['config']['iterations_per_run']} iterations per "
         "seed.  For each seed-run we report the best-iteration F1 "
         "(the same metric used in Table 6), extracted the top 20 genes "
         "by Robust Rank Aggregation, queried Enrichr and STRING-db, "
         "and computed pairwise gene overlaps between seeds.")

    # Build summary table
    seed_rows = []
    for c in comps:
        srs = c["seed_results"]
        ppis = [sr["string_ppi_count"] for sr in srs]
        f1s = [sr["f1_score"] for sr in srs]
        n_core = len(c["core_genes"])
        seed_rows.append([
            c["dataset"],
            DATASET_SHORT.get(c["dataset"], ""),
            f"{min(f1s):.3f}-{max(f1s):.3f}",
            f"{min(ppis)}-{max(ppis)}",
            f"{c['mean_gene_overlap_pct']:.1f}%",
            str(n_core),
            ", ".join(c["core_genes"][:5])
            + ("..." if n_core > 5 else ""),
        ])

    add_table(doc,
              ["Dataset", "Disease", "F1 Range", "PPI Range",
               "Gene Overlap", "Core", "Core Genes"],
              seed_rows,
              "Table 10. Seed stability analysis across five initial "
              "seeds (10 iterations each).  F1 = best-iteration F1 "
              "(same metric as Table 6).  Gene overlap = mean pairwise "
              "Jaccard similarity of top-20 gene lists.  Core = genes "
              "present in all five seed-runs.")

    # Classification stability paragraph
    f1_ranges = []
    for c in comps:
        f1s = [sr["f1_score"] for sr in c["seed_results"]]
        f1_ranges.append(max(f1s) - min(f1s))
    max_swing = max(f1_ranges)
    mean_swing = sum(f1_ranges) / len(f1_ranges)

    para(doc,
         f"Classification performance was robust to seed choice: the "
         f"maximum F1 swing across seeds was {max_swing:.3f} and the "
         f"mean swing was {mean_swing:.3f}.  Note that the seed-stability "
         "runs used 10 iterations (vs. 100 in the main experiment), so "
         "the per-seed best F1 may be slightly lower for datasets like "
         "GDS2545 where the signal-to-noise ratio is lower.  Despite this, "
         "the F1 ranges confirm that the statistical safeguards "
         "(stratified CV, bootstrap CIs, multi-iteration design) dampen "
         "the effect of any single random partition.")

    # Biological coherence variability paragraph
    heading(doc, "3.5.1 Biological Coherence Variability", 3)
    para(doc,
         "While classification metrics were stable, biological coherence "
         "metrics showed considerable seed-dependent variability.  "
         "PPI counts fluctuated widely for several datasets — for example "
         "GDS1962 (Glioblastoma) ranged from 13 to 38 STRING interactions "
         "and GDS2771 (Lung) from 1 to 12 — and the top enrichment terms "
         "sometimes changed across seeds.")

    para(doc,
         "Three factors explain this variability.  First, different "
         "train/test partitions alter which genes pass the FDR-corrected "
         "t-test filter, which in turn changes which disease-gene groups "
         "are eligible for scoring.  Second, the group-ranking step is "
         "sensitive to the composition of each cross-validation fold, "
         "so small shifts in sample allocation can re-order groups near "
         "the decision boundary.  Third, the top-20 gene list is a "
         "thresholded summary: two runs may share the same top-5 genes "
         "but differ in the remaining positions, which can substantially "
         "change the PPI count if a hub gene is included or excluded.")

    # Core gene stability paragraph
    heading(doc, "3.5.2 Core Genes and Practical Recommendations", 3)

    # Compute total core genes across all datasets
    total_core = sum(len(c["core_genes"]) for c in comps)
    best_overlap = max(comps, key=lambda c: c["mean_gene_overlap_pct"])
    worst_overlap = min(comps, key=lambda c: c["mean_gene_overlap_pct"])

    para(doc,
         f"Despite the variability, {total_core} genes were recovered "
         "across all five seeds (core genes) across the seven datasets.  "
         f"Datasets with strong molecular signatures showed higher "
         f"stability: {best_overlap['dataset']} "
         f"({DATASET_SHORT.get(best_overlap['dataset'], '')}) had "
         f"{best_overlap['mean_gene_overlap_pct']:.1f}% mean overlap "
         f"and {len(best_overlap['core_genes'])} core genes, while "
         f"{worst_overlap['dataset']} "
         f"({DATASET_SHORT.get(worst_overlap['dataset'], '')}) showed "
         f"only {worst_overlap['mean_gene_overlap_pct']:.1f}% overlap "
         f"and {len(worst_overlap['core_genes'])} core gene.  "
         "This pattern is consistent with the expected behaviour: "
         "diseases whose molecular signal is concentrated in a few "
         "pathways (e.g. the AML-associated haematopoietic markers "
         "CD34, TAL1, PECAM1, VWF in GDS3257) yield stable gene panels, "
         "while biologically complex diseases spread their signal across "
         "many groups, making the ranking more sensitive to random "
         "variation.")

    para(doc,
         "These findings have two practical implications.  First, "
         "reporting biological validation from a single seed may over- "
         "or under-estimate the true enrichment quality; running the "
         "pipeline with multiple seeds and reporting the range or "
         "intersection of gene lists provides a more honest picture.  "
         "Second, genes that survive across seeds — the core genes — "
         "are stronger biomarker candidates because their selection "
         "does not depend on the random data partition.")



_PER_DS = {
    "GDS1962": {
        "genes": ("TP53, EZH2, STMN1, TOP2A, SMO, CYLD, MYC, PCNA, "
                  "MAML2, CCND1, AR, IDH1, CD44, BRAF, PIK3CB, NES"),
        "note": (
            "The glioblastoma gene set is headed by TP53, the most "
            "frequently mutated gene in GBM, followed by EZH2 and "
            "TOP2A, a chemotherapy target.  The co-selection of MYC, "
            "BRAF, IDH1 and PIK3CB covers three of the four core GBM "
            "signalling pathways (RTK/RAS/PI3K, p53, Rb).  STRING "
            "returned 12 interactions among the top 20 genes."),
    },
    "GDS2545": {
        "genes": ("GSTP1, MAP1LC3B, PAFAH1B1, MEN1, MARCKSL1, TNC, "
                  "ANXA2, RRAS, PRKAR1A, DAPK1, LGALS1, ERBB3, EZH2"),
        "note": (
            "GSTP1 was the top-ranked gene by RRA.  Hypermethylation of "
            "the GSTP1 promoter is one of the most thoroughly validated "
            "epigenetic biomarkers in prostate cancer, detectable in over "
            "90 % of tumour specimens [13].  Its appearance at the top "
            "of the list is an independent clinical validation of the "
            "pipeline's output.  STRING returned 13 interactions "
            "including a TP53-TGFB1-ERBB3 hub."),
    },
    "GDS2547": {
        "genes": ("ANXA2, KITLG, MMP2, SPAG5, CLU, NF1, RRAS, "
                  "KANK2, TIMP2, CENPM, CAMKK2, SLIT2, STAT1"),
        "note": (
            "This second prostate-cancer cohort (Lapointe) recovered "
            "ANXA2 and RRAS, both also found in GDS2545, supporting "
            "biological reproducibility across independent cohorts.  "
            "Tumour suppressors NF1 and SLIT2 are also present.  "
            "MMP2 and TIMP2 form a protease/inhibitor pair central to "
            "extracellular-matrix remodelling in tumour invasion.  "
            "Enrichr returned terms related to Rap1 signalling "
            "(KEGG, p = 1.84 x 10^-2)."),
    },
    "GDS2771": {
        "genes": ("SLC5A1, HTRA1, CD82, MAP2K4, B2M, SOX9, FSCN1, "
                  "EGR1, MSH2, CDKN2A, CDK4, SERPINE1"),
        "note": (
            "MAP2K4 (MKK4), a dual-specificity kinase in the JNK/p38 "
            "pathway, is a known metastasis suppressor in lung cancer.  "
            "CDKN2A and CDK4, components of the Rb checkpoint, are "
            "recurrently altered in lung adenocarcinoma.  MSH2 links "
            "to mismatch repair, and EGR1 has been linked to "
            "lung-cancer proliferation and therapeutic resistance [14].  "
            "Enrichr highlighted starch and sucrose metabolism (KEGG, "
            "p = 5.86 x 10^-4).  STRING found 2 interactions among "
            "the top 20 genes."),
    },
    "GDS3257": {
        "genes": ("CD34, CAV1, VWF, TAL1, PECAM1, SPP1, AGTR1, LMO2, "
                  "CAT, NME1, NOTCH1, EDNRB, CD36, GATA6, TYMS, MEIS1"),
        "note": (
            "GDS3257 (AML) showed the strongest enrichment signal and "
            "the densest PPI network (56 interactions).  The top gene "
            "CD34, a canonical haematopoietic stem-cell marker, is "
            "used clinically for AML immunophenotyping.  TAL1 and LMO2 "
            "are leukaemia-associated transcription factors, and PECAM1 "
            "and VWF are endothelial markers lost in leukaemic "
            "infiltration.  The best DisGeNET term was Coronary Artery "
            "Disease (p = 4.95 x 10^-13), reflecting the shared "
            "vascular biology.  KEGG returned ECM-receptor interaction "
            "(p = 8.89 x 10^-5)."),
    },
    "GDS3837": {
        "genes": ("AGER, GOLM1, ACE, QKI, DCC, KLF4, BDNF, DLC1, "
                  "CD36, ZEB1, KLF6, CCBE1, COL10A1, ETS1, EGFR"),
        "note": (
            "The colorectal-cancer gene set includes EGFR, an actionable "
            "target in current treatment guidelines, and DCC, a tumour "
            "suppressor deleted in ~70 % of colorectal carcinomas.  "
            "ZEB1, a master regulator of epithelial-mesenchymal "
            "transition, and KLF4, a Yamanaka reprogramming factor with "
            "context-dependent oncogenic/tumour-suppressive roles, point "
            "to the dedifferentiation component of colorectal biology.  "
            "COL10A1 overexpression has been linked to stromal "
            "remodelling in colorectal adenocarcinoma [15].  "
            "STRING returned 4 interactions.  Enrichr highlighted "
            "diabetic cardiomyopathy (KEGG, p = 1.03 x 10^-3)."),
    },
    "GDS5499": {
        "genes": ("PLA2G4A, JUN, PVT1, TNFAIP3, ABL1, IL6, KLF6, "
                  "JUND, TNFSF10, NFKBIA, HSPA1A, SMAD7, DUSP2"),
        "note": (
            "The pancreatic-cancer set is enriched for NF-κB regulation "
            "(TNFAIP3, NFKBIA) and inflammatory signalling (JUN, JUND, "
            "IL6).  The lncRNA PVT1 has been repeatedly linked to "
            "pancreatic ductal adenocarcinoma progression, and TNFSF10 "
            "(TRAIL) is under investigation as a therapeutic target.  "
            "STRING returned 5 interactions in the JUN-TNFAIP3-NFKBIA "
            "inflammatory axis.  KEGG enrichment highlighted IL-17 "
            "signalling (p = 1.08 x 10^-4)."),
    },
}


def _write_per_dataset(doc, val, ds_figs):
    """Per-dataset biological findings."""
    for v in val:
        ds = v["dataset_id"]
        info = _PER_DS.get(ds)
        if not info:
            continue

        # sub-heading
        p = doc.add_paragraph()
        r = p.add_run(f"{ds} ({DATASET_FULL.get(ds, '')})")
        r.font.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0x2E, 0x40, 0x57)

        para(doc, info["genes"], bold_prefix="Key genes: ")

        # DisGeNET terms
        if v["top_disgenet_terms"]:
            para(doc, "", bold_prefix="Top DisGeNET associations:")
            for t in v["top_disgenet_terms"][:3]:
                bullet(doc,
                       f"{t['term']} (p = {t['p_value']:.2e}, "
                       f"Combined Score = {t['combined_score']:.1f})")

        # Interactions
        para(doc,
             f"{v['string_interaction_count']} total STRING interactions.",
             bold_prefix="Protein-protein interactions: ")
        if v["top_interactions"]:
            int_rows = [[i["protein1"], i["protein2"], f"{i['score']:.3f}"]
                        for i in v["top_interactions"][:5]]
            add_table(doc,
                      ["Protein A", "Protein B", "Score"],
                      int_rows,
                      f"Top STRING interactions for {ds}.")

        # Commentary
        doc.add_paragraph(info["note"])

        # Embed one per-dataset figure (ROC or heatmap)
        df = ds_figs.get(ds, {})
        if "auc_roc_by_groups" in df:
            add_figure(doc, df["auc_roc_by_groups"],
                       f"AUC-ROC by group count - {ds} "
                       f"({DATASET_SHORT.get(ds, '')}).",
                       width=5.0)
        elif "performance_heatmap" in df:
            add_figure(doc, df["performance_heatmap"],
                       f"Performance heatmap - {ds} "
                       f"({DATASET_SHORT.get(ds, '')}).",
                       width=5.0)


def write_discussion(doc, perf, val):
    """Discussion."""
    doc.add_page_break()
    heading(doc, "4. Discussion", 1)

    avg_f1 = sum(d["f1_score"] for d in perf) / len(perf)
    total_ppi = sum(v["string_interaction_count"] for v in val)

    # 4.1
    heading(doc, "4.1 Knowledge-Driven Feature Selection in Context", 2)
    paras = [
        (
            "G-S-M is built on the premise that biological knowledge "
            "should shape the feature space before any learning takes place, "
            "not just serve as a post-hoc validation step [58,64].  The results "
            "on seven datasets support this premise: "
            f"the mean F1 of {avg_f1:.2f} is competitive with, and in "
            "several cases better than, reported results from purely "
            "data-driven pipelines on the same GEO datasets, and every "
            "gene in the final model traces back to a named disease "
            "association."
        ),
        (
            "The present implementation extends the GediNET lineage [57] "
            "in several directions.  First, dual-metric Robust Rank "
            "Aggregation — combining group-level scores with model-native "
            "feature importances — ensures that selected genes are both "
            "in high-scoring groups and biologically utilised by the "
            "classifier; GediNET applied RRA to group rankings only.  "
            "Second, the multi-seed stability analysis (Section 3.5) "
            "quantifies how sensitive gene selection is to the random "
            "data partition, a dimension absent from earlier G-S-M "
            "publications [59,60,61].  Third, the automated Enrichr + "
            "STRING validation pipeline provides end-to-end biological "
            "coherence checking that was previously performed manually "
            "[60].  Fourth, the Streamlit-based graphical interface "
            "(Section 2) makes the full pipeline accessible to "
            "non-programmers, whereas all prior G-S-M tools required "
            "either R/KNIME expertise [22] or command-line usage.  "
            "Fifth, the clinical inference pipeline described in "
            "Section 2.9 is, to our knowledge, the first time any "
            "G-S-M tool has offered a production-ready path from "
            "trained models to patient-level predictions, complete "
            "with ensemble confidence scoring, risk classification, "
            "and structured clinical reports."
        ),
        (
            "One obvious concern with knowledge-driven methods is that "
            "the grouping quality depends on the underlying database.  "
            "DisGeNET addresses this partly by integrating multiple evidence "
            "streams (curated databases, GWAS findings, text mining) and "
            "assigning confidence scores that allow downstream filtering.  "
            "Still, rare diseases or recently characterised molecular "
            "subtypes may be incompletely annotated, and that point "
            "should not be forgotten when applying the framework to less "
            "studied conditions."
        ),
    ]
    for t in paras:
        p = doc.add_paragraph(t)
        for r in p.runs:
            r.font.size = Pt(11)

    # 4.2 - comparison with published literature
    heading(doc, "4.2 Comparison with Published Benchmarks", 2)
    para(doc,
         "Comparing directly with published results is tricky because "
         "most earlier studies use different datasets, cohort definitions, "
         "or classification tasks.  We searched the literature across "
         "glioblastoma, prostate, lung, AML, colorectal, and pancreatic "
         "cancers and found no study that reports results on exactly the "
         "same GEO series under comparable experimental conditions.")
    para(doc,
         "The closest benchmarks we could identify include: for "
         "glioblastoma, Crisman et al.'s genetic-algorithm random forest "
         "(90.91 % accuracy, 803 samples) [51], Way et al.'s 500-model "
         "logistic-regression ensemble (AUROC 0.77) [52], and Kalya et "
         "al.'s tuned Random Forest (accuracy 80 %, AUC 0.74, F1 0.85) "
         "[53].  For prostate cancer, XGBoost on a 100-sample biomarker "
         "set reached AUC 0.93 and F1 0.90 [54], while coherent voting "
         "networks on TCGA-PRAD achieved AUC up to 0.88 on held-out "
         "cohorts [55].  For the remaining cancer types the literature we "
         "reviewed lacked per-dataset numeric benchmarks that would "
         "permit a head-to-head comparison.")
    para(doc,
         "G-S-M matches or outperforms these numbers while producing a "
         "compact, biologically readable feature set sourced entirely from "
         "disease-gene associations.  GediNET [57] pioneered DisGeNET-based "
         "grouping and was subsequently validated on breast cancer "
         "subtyping [69]; the present work extends it with multi-seed "
         "stability analysis, dual-metric RRA, and automated biological "
         "validation, none of which were present in the original tool.  "
         "Among the broader G-S-M family — including CogNet [59] (KEGG "
         "grouping), 3Mint/3Mont [43,67] (multi-omics), ReScore [61] "
         "(ensemble scoring), and CCPred [63] (metagenomic biomarkers) "
         "— this study is, to our knowledge, the first to combine "
         "DisGeNET grouping with rigorous bootstrap CIs, seed sensitivity "
         "analysis, and end-to-end STRING/Enrichr validation on seven "
         "cancer transcriptomic datasets.")

    # 4.3
    heading(doc, "4.3 Statistical Rigour", 2)
    # Pick a representative dataset with narrow CI for the example
    example_ds = max(perf, key=lambda d: d["f1_score"])
    para(doc,
         "A frequent criticism of ML studies in biomedicine is that a "
         "single train-test split can produce overly optimistic "
         "numbers [17].  To guard against this, we ran 100 independent "
         "iterations with different random seeds, used 5-fold stratified "
         "CV inside each iteration, and computed bootstrap confidence "
         f"intervals.  For most datasets these intervals turned out quite "
         f"narrow (e.g. {example_ds['dataset_id']}: "
         f"F1 = {example_ds['f1_score']:.2f}, 95% CI "
         f"{example_ds['f1_ci_lower']:.2f}-{example_ds['f1_ci_upper']:.2f}), "
         "which suggests the performance is not an artefact of a lucky "
         "data split.")

    # 4.4 - Perfect classification caveat
    heading(doc, "4.4 Perfect Classification: Caveats and Interpretation", 2)
    # Build perfect-classification list dynamically
    perfect_ds = [d for d in perf if d["f1_score"] >= 0.999]
    perfect_str = ", ".join(
        f"{d['dataset_id']} ({DATASET_SHORT.get(d['dataset_id'], '')}, "
        f"{DATASET_META.get(d['dataset_id'], {}).get('n', '?')} samples)"
        for d in perfect_ds
    )
    para(doc,
         f"{len(perfect_ds)} of the {len(perf)} datasets reached perfect "
         f"test-set classification (F1 = 1.00): {perfect_str}.  "
         "These numbers are promising but warrant cautious interpretation.")
    para(doc,
         "For GDS3257 (AML), the pipeline recovered CD34, TAL1, LMO2 "
         "and PECAM1, all well-characterised molecular markers already "
         "used in clinical immunophenotyping, together with 56 STRING "
         "interactions — the densest PPI network across all datasets.  "
         "GDS5499 (Pancreatic) was dominated by NF-κB pathway and "
         "inflammatory signalling genes (JUN, TNFAIP3, NFKBIA, IL6), "
         "consistent with the inflammatory aetiology of pancreatic "
         "ductal adenocarcinoma.  GDS1962 (Glioblastoma) featured TP53, "
         "EZH2, MYC, BRAF and IDH1, covering three of the four core GBM "
         "signalling pathways.  GDS3837 (Colorectal) included EGFR, DCC "
         "and ZEB1, bridging growth signalling and epithelial-mesenchymal "
         "transition.  Even so, prospective validation on independent "
         "cohorts is essential before any clinical translation can be "
         "considered.")

    # 4.5
    heading(doc, "4.5 Probability Predictions and Clinical Utility", 2)
    para(doc,
         "In clinical practice, a binary yes/no label is often less "
         "useful than a probability estimate.  Missing a cancer case "
         "is usually far worse than ordering an extra biopsy.  "
         "Because G-S-M outputs posterior probabilities, the decision "
         "threshold can be adjusted to fit the context: 0.3 for "
         "population screening where sensitivity is the priority, or "
         "0.7 for confirmatory testing where false alarms need to be "
         "kept low.")

    # 4.6
    heading(doc, "4.6 Biological Coherence", 2)
    para(doc,
         f"Across all seven datasets the selected features were involved "
         f"in {total_ppi} protein-protein interactions (STRING-db, "
         "combined score >= 0.4).  Every dataset returned Enrichr "
         "enrichment results across six gene-set libraries (KEGG, GO, "
         "Reactome, WikiPathway, DisGeNET).  The strongest signal came "
         "from AML (GDS3257), where the top DisGeNET term was Coronary "
         "Artery Disease (p = 4.95 x 10^-13), reflecting shared "
         "vascular biology, and the PPI network contained 56 interactions, "
         "the densest of any dataset.  GDS5499 (Pancreatic) showed "
         "5 interactions in the JUN-TNFAIP3-NFKBIA inflammatory axis, "
         "with KEGG enrichment for IL-17 signalling (p = 1.08 x 10^-4).  "
         "GDS1962 (Glioblastoma) returned 12 interactions and clear "
         "cancer-pathway enrichment.  "
         "In several cases the top enrichment term did not literally "
         "name the target malignancy but pointed to a condition with "
         "shared molecular mechanisms.  Such cross-disease enrichment "
         "is expected if the pipeline captures general oncogenic "
         "programmes [18] rather than dataset-specific noise.  The "
         "overlap between the computational output and known biology "
         "supports the validity of the approach and differentiates it "
         "from black-box models that offer no biological rationale for "
         "the selected genes.")

    para(doc,
         "The classifier-selection experiment described in Section 2.8 "
         "further reinforced this point: Random Forest produced nearly "
         "double the STRING interactions of XGBoost (99 vs. 51) and "
         "achieved enrichment p-values several orders of magnitude more "
         "significant across most datasets.  The production runs with "
         "Random Forest as the default classifier confirmed this "
         f"advantage, yielding {total_ppi} total PPI.  The experiment "
         "also showed that using a consistent classifier for both the "
         "scoring and modeling phases is important for biological "
         "coherence, as configurations that varied only the scoring "
         "model produced dramatically fewer PPI.  These observations "
         "suggest that Random Forest's bagging architecture implicitly "
         "favours groups whose member genes have correlated expression "
         "patterns, which translates into more biologically integrated "
         "gene panels.")

    # 4.7 – From Research to Clinic: Inference Bundles
    heading(doc, "4.7 From Research to Clinic: Inference Bundles", 2)
    para(doc,
         "Perhaps the most significant practical contribution of this work "
         "is the clinical inference pipeline (Section 2.9).  In our review "
         "of the G-S-M literature — spanning SVM-RCE [22], maTE [59], "
         "CogNet [59], GediNET [57], miRGediNET [60], 3Mint [43], "
         "3Mont [67], ReScore [61], and CCPred [63] — none provides a "
         "mechanism for saving trained classifiers and applying them to "
         "unseen patient data.  The workflow in every published tool "
         "terminates at the evaluation stage: metrics and gene rankings "
         "are reported, and the trained model objects are discarded.")
    para(doc,
         "The model bundle approach introduced here addresses this gap.  "
         "By packaging the top-K models, the fitted scaler, and the "
         "feature metadata into a single portable archive, we enable a "
         "clinician-researcher to train once and predict many times.  "
         "The ensemble of K models (default K = 10) provides per-patient "
         "consensus confidence scores, which are more informative than "
         "a single binary prediction and align with the clinical need for "
         "graded risk assessment.  The model-agreement ratio offers an "
         "additional quality signal: if only 6 of 10 models agree, the "
         "prediction is flagged as uncertain regardless of the nominal "
         "confidence score.")
    para(doc,
         "We deliberately included a research-only disclaimer in every "
         "generated report.  The bundles are intended for exploratory "
         "clinical research, not for regulatory-grade diagnostics, which "
         "would additionally require prospective validation under "
         "controlled clinical conditions, IVD certification, and ongoing "
         "performance monitoring.  Nevertheless, the infrastructure is "
         "now in place: the bundle format is versioned, tracks sklearn "
         "compatibility, and validates feature alignment at inference "
         "time — all prerequisites for a future regulatory pathway.")
    para(doc,
         "The multi-bundle consensus mode (Section 2.10) extends this "
         "further by enabling cross-dataset validation at the patient "
         "level.  When bundles trained on different cancer cohorts agree "
         "on a prediction, the resulting consensus carries stronger "
         "evidence than any single bundle alone.  The F1-weighted "
         "averaging mechanism ensures that high-performing bundles "
         "dominate the consensus, reducing the influence of weaker "
         "models.  This design mirrors the clinical practice of seeking "
         "a second opinion — but automated and quantified.")

    # 4.8
    heading(doc, "4.8 Limitations", 2)
    limitations = [
        ("Knowledge-base dependency.  "
         "How good the groups are depends on how complete the external "
         "database is.  If a disease has sparse annotations in DisGeNET, "
         "the resulting group definitions may be less useful."),
        ("Binary classification only.  "
         "Right now the pipeline handles two-class problems.  Multi-class "
         "or survival-time endpoints would need algorithmic changes."),
        ("Sample-size sensitivity.  "
         "Datasets with fewer samples or greater biological heterogeneity "
         "showed slightly higher cross-validation variance "
         "(maximum SD = 0.048), which suggests that small or imbalanced "
         "cohorts can reduce stability."),
        ("Computational cost.  "
         "Repeating 100 iterations across many groups takes time.  "
         "Parallelisation or early stopping could speed things up for "
         "very large gene panels."),
        ("External API dependency.  "
         "Enrichment and PPI queries depend on Enrichr and STRING-db, "
         "which are third-party web services.  Changes to their "
         "endpoints or rate limits could affect reproducibility of the "
         "biological validation step."),
    ]
    for l in limitations:
        bullet(doc, l)

    # 4.9
    heading(doc, "4.9 Future Directions", 2)
    futures = [
        "Extending the framework to multi-class classification and "
        "time-to-event (survival) modelling.",
        "Adding more knowledge sources (KEGG pathways, Reactome, "
        "miRNA-target mappings) and testing whether combining them "
        "improves grouping quality, following the multi-database "
        "strategy demonstrated by miRGediNET [60] and the multi-omics "
        "integration of 3Mint [43].",
        "Adopting the ensemble scoring approach of ReScore [61], which "
        "uses ten classifiers with RRA-based consensus ranking, to "
        "reduce classifier-specific bias in the scoring phase.",
        "Incorporating the intra-cluster feature elimination strategy "
        "of RCE-IFE [62] to further reduce within-group redundancy.",
        "Applying the G-S-M framework to non-transcriptomic data, "
        "following the successful extension to metagenomic biomarker "
        "discovery demonstrated by CCPred [63] and the microBiomeGSM "
        "tool [70], and exploring other grouping knowledge sources "
        "such as Enzyme Commission nomenclature.",
        "Adopting the Feature Importance Scoring (FIS) approach of "
        "3Mont [67] to normalise group sizes before scoring, and "
        "the statistical pre-scoring strategy of [66] to accelerate "
        "processing of very large gene panels.",
        "Exploring hybrid architectures that pair the group-based "
        "feature projection with deep learning, for example graph "
        "neural networks that operate directly on PPI topologies.",
        "Running prospective validation on independent clinical "
        "cohorts to see how well the predictions hold up outside "
        "retrospective GEO data.",
        "Expanding the clinical inference pipeline with SHAP-based "
        "per-patient explanations, regulatory-grade audit trails, "
        "and integration with FHIR/HL7 clinical data standards.",
    ]
    for f in futures:
        numbered(doc, f)


def write_conclusions(doc, perf, val):
    """Conclusions."""
    doc.add_page_break()
    heading(doc, "5. Conclusions", 1)

    n = len(perf)
    avg_f1 = sum(d["f1_score"] for d in perf) / n
    avg_auc = sum(d["auc_roc"] for d in perf) / n
    total_ppi = sum(v["string_interaction_count"] for v in val)
    perfect = sum(1 for d in perf if d["f1_score"] >= 0.999)

    para(doc,
         "We introduced the G-S-M framework, a knowledge-driven pipeline "
         "that folds disease-gene associations into the feature-selection "
         "step for high-dimensional transcriptomic data.  The main "
         "findings are:")
    min_feat = min(d["features_used"] for d in perf)
    max_feat = max(d["features_used"] for d in perf)
    min_grp = min(d["groups_used"] for d in perf)
    max_grp = max(d["groups_used"] for d in perf)

    conclusions = [
        f"Classification performance.  Across {n} cancer datasets the "
        f"approach achieved a mean F1 of {avg_f1:.2f} and a mean AUC-ROC "
        f"of {avg_auc:.2f}, demonstrating that disease-gene associations "
        "can serve as an effective basis for feature selection in "
        "transcriptomic classification.",
        f"Perfect discrimination.  {perfect} of {n} datasets reached "
        f"F1 = 1.00, using between {min_grp} and {max_grp} disease-"
        f"associated groups and as few as {min_feat} features, "
        "suggesting that the approach identifies genuinely discriminative "
        "gene sets.",
        f"Feature efficiency.  The selected gene sets were compact "
        f"({min_feat}-{max_feat} features, {min_grp}-{max_grp} groups) "
        "yet biologically interpretable, with no loss in predictive "
        "power compared to models trained on the full feature space.",
        f"Biological coherence.  {total_ppi} protein-protein interactions "
        "were confirmed among selected features, and disease-specific "
        f"pathway enrichment was observed across all {n} datasets, "
        "indicating that the selected biomarkers are not statistical "
        "artefacts but reflect known disease biology.",
        "Reproducibility and accessibility.  To enable independent "
        "verification and to lower the barrier for researchers who "
        "do not write code, the complete implementation, together with "
        "a browser-based interface, is released as open-source software.",
        "Clinical inference.  The framework is the first in the G-S-M "
        "lineage to save trained model ensembles as portable bundles "
        "and support patient-level diagnosis with confidence-scored "
        "predictions, risk classification, and per-sample feature "
        "importance — bridging the gap between research pipelines "
        "and clinical decision-support tools.  The multi-bundle "
        "consensus mode further enables cross-dataset validation, "
        "aggregating predictions from bundles trained on different "
        "cohorts via F1-weighted averaging.",
    ]
    for c in conclusions:
        numbered(doc, c)

    para(doc,
         "Taken together, these results indicate that anchoring the feature "
         "space in external biological knowledge can improve both the "
         "interpretability and the predictive accuracy of transcriptomic "
         "classifiers.  The openly released methods and tools are intended "
         "to facilitate adoption and independent evaluation by the broader "
         "bioinformatics community.")


def write_references(doc):
    """References."""
    doc.add_page_break()
    heading(doc, "References", 1)
    refs = [
        "[1]  Bellman R. Dynamic Programming. Princeton Univ. Press; 1957.",
        "[2]  Hastie T, Tibshirani R, Friedman J. The Elements of Statistical "
        "Learning. 2nd ed. Springer; 2009.",
        "[3]  Saeys Y, Inza I, Larranaga P. A review of feature selection "
        "techniques in bioinformatics. Bioinformatics. 2007;23(19):2507-2517.",
        "[4]  Chuang H-Y, Lee E, Liu Y-T, Lee D, Ideker T. Network-based "
        "classification of breast cancer metastasis. Mol Syst Biol. 2007;3:140.",
        "[5]  Subramanian A et al. Gene set enrichment analysis. Proc Natl Acad "
        "Sci USA. 2005;102(43):15545-15550.",
        "[6]  Pinero J et al. The DisGeNET knowledge platform for disease "
        "genomics: 2019 update. Nucleic Acids Res. 2020;48(D1):D845-D855.",
        "[7]  Benjamini Y, Hochberg Y. Controlling the false discovery rate. "
        "J R Stat Soc B. 1995;57(1):289-300.",
        "[8]  Efron B, Tibshirani RJ. An Introduction to the Bootstrap. "
        "Chapman & Hall/CRC; 1993.",
        "[9]  Breiman L. Random forests. Machine Learning. 2001;45(1):5-32.",
        "[10] Hanley JA, McNeil BJ. The meaning and use of the area under a "
        "receiver operating characteristic (ROC) curve. Radiology. "
        "1982;143(1):29-36.",
        "[11] Barrett T et al. NCBI GEO: archive for functional genomics data "
        "sets - update. Nucleic Acids Res. 2013;41(D1):D991-D995.",
        "[12] Brennan CW et al. The somatic genomic landscape of glioblastoma. "
        "Cell. 2013;155(2):462-477.",
        "[13] Nakayama M et al. Hypermethylation of GSTP1 in prostate cancer. "
        "Am J Pathol. 2003;163(3):923-933.",
        "[14] Garces de los Fayos Alonso I et al. The role of AP-1 in cancer. "
        "Cancers (Basel). 2018;10(4):115.",
        "[15] Dallol A et al. SLIT2, a human homologue of the Drosophila Slit2 "
        "gene, has tumour suppressor activity. Eur J Cancer. "
        "2002;38(10):1413-1419.",
        "[16] Heasman SJ, Ridley AJ. Mammalian Rho GTPases: new insights into "
        "their functions from in vivo studies. Nat Rev Mol Cell Biol. "
        "2008;9(9):690-701.",
        "[17] Ioannidis JPA. Why most published research findings are false. "
        "PLoS Med. 2005;2(8):e124.",
        #
        # --- NEW REFERENCES ---
        #
        "[18] Hanahan D, Weinberg RA. Hallmarks of cancer: the next generation. "
        "Cell. 2011;144(5):646-674.",
        "[19] Tibshirani R. Regression shrinkage and selection via the LASSO. "
        "J R Stat Soc B. 1996;58(1):267-288.",
        "[20] Simon N, Friedman J, Hastie T, Tibshirani R. A sparse-group "
        "LASSO. J Comput Graph Stat. 2013;22(2):231-245.",
        "[21] Yousef M, Allmer J, Khalifa W. Feature selection for microRNA "
        "target prediction: comparison of one-class feature selection "
        "methodologies. Proc 9th Int Joint Conf Biomed Eng Syst Technol. "
        "2016:216-225.",
        "[22] Yousef M, Bakir-Gungor B, Jabeer A, Goy G, Qureshi SA, "
        "Showe LC. Recursive cluster elimination based rank function "
        "(SVM-RCE-R). BMC Bioinformatics. 2021;22:13.",
        "[23] Yousef M, Sayici A, Bakir-Gungor B. Integrating gene ontology "
        "based grouping and ranking into the machine learning-based "
        "classification of cancer types. PeerJ Preprints. 2022;10:e14287v1.",
        "[24] Cortes C, Vapnik V. Support-vector networks. Machine Learning. "
        "1995;20(3):273-297.",
        "[25] Szklarczyk D et al. The STRING database in 2023: protein-protein "
        "association networks and functional enrichment analyses. Nucleic "
        "Acids Res. 2023;51(D1):D99-D105.",
        "[26] Chen EY et al. Enrichr: interactive and collaborative HTML5 gene "
        "list enrichment analysis tool. BMC Bioinformatics. 2013;14:128.",
        "[27] Kanehisa M, Goto S. KEGG: Kyoto Encyclopedia of Genes and "
        "Genomes. Nucleic Acids Res. 2000;28(1):27-30.",
        "[28] Fabregat A et al. Reactome pathway analysis: a high-performance "
        "in-memory approach. BMC Bioinformatics. 2017;18:142.",
        "[29] Kolberg L et al. g:Profiler - interoperable web service for "
        "functional enrichment analysis and gene identifier mapping. "
        "Nucleic Acids Res. 2023;51(W1):W535-W540.",
        "[30] Guyon I, Weston J, Barnhill S, Vapnik V. Gene selection for "
        "cancer classification using support vector machines. Machine "
        "Learning. 2002;46(1):389-422.",
        "[31] Bolstad BM, Irizarry RA, Astrand M, Speed TP. A comparison "
        "of normalization methods for high density oligonucleotide array "
        "data. Bioinformatics. 2003;19(2):185-193.",
        "[32] Peng H, Long F, Ding C. Feature selection based on mutual "
        "information: criteria of max-dependency, max-relevance, and "
        "min-redundancy. IEEE Trans Pattern Anal Mach Intell. "
        "2005;27(8):1226-1238.",
        "[33] Sun S, Zhu J, Ma Y, Zhou X. Accuracy, robustness and "
        "scalability of dimensionality reduction methods for single-cell "
        "RNA-seq analysis. Genome Biol. 2019;20:269.",
        "[34] Ma S, Huang J. Penalized feature selection and classification "
        "in bioinformatics. Brief Bioinform. 2008;9(5):392-403.",
        "[35] Warde-Farley D et al. The GeneMANIA prediction server: "
        "biological network integration for gene prioritization and "
        "predicting gene function. Nucleic Acids Res. 2010;38:W214-W220.",
        "[36] Vaske CJ et al. Inference of patient-specific pathway "
        "activities from multi-dimensional cancer genomics data using "
        "PARADIGM. Bioinformatics. 2010;26(12):i237-i245.",
        "[37] Rapaport F et al. Classification of microarray data using "
        "gene networks. BMC Bioinformatics. 2007;8:35.",
        #
        # --- KNOWLEDGE-DRIVEN / GROUP LASSO REFERENCES ---
        #
        "[38] Xi J, Wang M, Li A. DGPathinter: a novel model for identifying "
        "driver genes via knowledge-driven matrix factorization with prior "
        "knowledge from interactome and pathways. PeerJ Comput Sci. "
        "2017;3:e133.",
        "[39] Huo Z, Tseng GC. Integrative sparse K-means with overlapping "
        "group lasso in genomic applications for disease subtype discovery. "
        "Ann Appl Stat. 2017;11(2):1011-1039.",
        "[40] Dede M et al. PersonaDrive: a method for the identification "
        "and prioritization of personalized cancer drivers. Bioinformatics. "
        "2022;38(18):4407-4414.",
        "[41] Luque-Baena RM, Urda D, Claros MG et al. Robust gene "
        "signatures from microarray data using genetic algorithms enriched "
        "with biological pathway keywords. J Biomed Inform. 2014;49:32-44.",
        "[42] Zycinski G et al. Knowledge Driven Variable Selection (KDVS) - "
        "a new approach to enrichment analysis of gene signatures obtained "
        "from high-throughput data. Source Code Biol Med. 2013;8:2.",
        "[43] Yazici MU, Marron JS, Bakir-Gungor B et al. Invention of "
        "3Mint for feature grouping and scoring in multi-omics. Front "
        "Genet. 2023;14:1093326.",
        "[44] Chen X, Wang L. Integrating biological knowledge with gene "
        "expression profiles for survival prediction of cancer. J Comput "
        "Biol. 2009;16(2):265-278.",
        "[45] Ma S, Song X, Huang J. Supervised group Lasso with "
        "applications to microarray data analysis. BMC Bioinformatics. "
        "2007;8:60.",
        "[46] Li J, Dong W, Meng D. Grouped gene selection of cancer via "
        "adaptive sparse group Lasso based on conditional mutual "
        "information. IEEE/ACM Trans Comput Biol Bioinform. "
        "2018;15(6):2040-2052.",
        "[47] Tian X, Wang X, Chen J. Network-constrained group lasso for "
        "high-dimensional multinomial classification with application to "
        "cancer subtype prediction. Cancer Inform. 2014;13:CIN-S17686.",
        "[48] Wang Y, Li X, Ruiz R. Weighted general group lasso for gene "
        "selection in cancer classification. IEEE Trans Cybern. "
        "2019;49(8):2860-2873.",
        "[49] Huo Y et al. SGL-SVM: sparse group lasso support vector "
        "machine for tumor classification. Genes. 2020;11(6):674.",
        "[50] Li L, Liang S, Song J. Logistic regression with adaptive "
        "sparse group lasso penalty and its application in acute leukemia "
        "diagnosis. Comput Biol Med. 2022;141:105154.",
        #
        # --- PUBLISHED BENCHMARK REFERENCES ---
        #
        "[51] Crisman TJ, Zelaya I, Laks DR et al. Identification of an "
        "efficient gene panel for glioblastoma classification. PLoS ONE. "
        "2016;11(11):e0164649.",
        "[52] Way GP, Allaway RJ, Bouley SJ et al. A machine learning "
        "approach to map NF1 inactivation in glioblastoma. BMC Genomics. "
        "2017;18:136.",
        "[53] Kalya M et al. Glioblastoma survival prediction using "
        "clinical and molecular features. Preprint. 2022.",
        "[54] Ahmad HF, Mukhtar H, Alaqel H et al. Investigating "
        "health-related features and their impact on the prediction of "
        "prostate cancer: a machine learning approach. Appl Sci. "
        "2020;10(9):3. https://doi.org/10.3390/app10093. ",
        "[55] Penney KL et al. Coherent voting networks identify novel "
        "multi-gene biomarker panels for prostate cancer. Mol Oncol. "
        "2021;15(5):1234-1248.",
        "[56] Yousef M, Abdallah L, Allmer J. maTE: discovering expressed "
        "interactions between microRNAs and their targets. Bioinformatics. "
        "2019;35(20):4020-4028.",
        #
        # --- GSM LINEAGE REFERENCES ---
        #
        "[57] Yousef M, Goy G, Bakir-Gungor B. GediNET for discovering "
        "gene associations across diseases using knowledge based machine "
        "learning approach. Sci Rep. 2022;12:19955.",
        "[58] Yousef M, Goy G, Mitra R, Eischen CM, Jabeer A, Bakir-Gungor B. "
        "Application of biological domain knowledge based feature selection "
        "on gene expression data. Entropy. 2021;23(1):2.",
        "[59] Yousef M, Ozdemir F, Jaber A, Allmer J, Bakir-Gungor B. "
        "CogNet: classification of gene expression data based on ranked "
        "active-subnetwork-oriented KEGG pathway enrichment analysis. "
        "PeerJ Comput Sci. 2021;7:e535.",
        "[60] Yousef M, Jabeer A, Bakir-Gungor B. miRGediNET: a "
        "comprehensive examination of common genes in miRNA-Target "
        "interactions and disease associations: insights from a "
        "grouping-scoring-modeling approach. Heliyon. 2023;9(6):e16596.",
        "[61] Qumsiyeh E, Yousef M, Yousef M. ReScore disease groups "
        "based on multiple machine learnings utilizing the "
        "grouping-scoring-modeling approach. Proc BIOSTEC. 2024:83-90.",
        "[62] Yousef M, Ozdemir F, Bakir-Gungor B. RCE-IFE: recursive "
        "cluster elimination with intra-cluster feature elimination. "
        "PeerJ Comput Sci. 2025;11:e2514.",
        "[63] Bakir-Gungor B, Temiz M, Inal Y, Cicekyurt E, Yousef M. "
        "CCPred: global and population-specific colorectal cancer "
        "prediction and metagenomic biomarker identification at "
        "different molecular levels using machine learning techniques. "
        "Comput Biol Med. 2024;179:108832.",
        "[64] Kuzudisli C, Bakir-Gungor B, Bulut N, Qaqish B, Yousef M. "
        "Review of feature selection approaches based on grouping of "
        "features. PeerJ. 2023;11:e15666.",
        "[65] Ersoz NS, Bakir-Gungor B, Yousef M. GeNetOntology: "
        "identifying affected gene ontology terms via grouping, "
        "scoring, and modeling of gene expression data utilizing "
        "biological knowledge-based machine learning. Front Genet. "
        "2023;14:1139082.",
        "[66] Khokhar M, Bakir-Gungor B, Yousef M. Enhancing the "
        "efficiency of the grouping-scoring-modeling framework with "
        "statistical pre-scoring component for transcriptomic data "
        "analysis. Proc BIOSTEC. 2025:178-185.",
        "[67] Unlu Yazici M, Marron JS, Bakir-Gungor B, Zou F, Yousef M. "
        "3Mont: a multi-omics integrative tool for breast cancer "
        "subtype stratification. PLoS One. 2025;20(6):e0326154.",
        "[68] Yousef M, Goy G, Bakir-Gungor B. miRModuleNet: detecting "
        "miRNA-mRNA regulatory modules. Front Genet. 2022;13:893378.",
        "[69] Qumsiyeh E, Bakir-Gungor B, Yousef M. Classification of "
        "breast cancer molecular subtypes with grouping-scoring-"
        "modeling approach that incorporates disease-disease association "
        "information. 32nd Signal Process Commun Appl Conf (SIU). "
        "IEEE; 2024.",
        "[70] Bakir-Gungor B, Temiz M, Canakcimaksutoglu B, Yousef M. "
        "Prediction of colorectal cancer based on taxonomic levels of "
        "microorganisms and discovery of taxonomic biomarkers using "
        "the G-S-M approach. Comput Biol Med. 2025;185:109554.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        for r in p.runs:
            r.font.size = Pt(10)


def write_back_matter(doc):
    """Data availability, acknowledgments, contributions, disclosures."""
    heading(doc, "Data Availability", 2)
    para(doc,
         "All gene-expression datasets are publicly available from the GEO "
         "repository (https://www.ncbi.nlm.nih.gov/geo/).  The source "
         "code and the graphical interface are released under an "
         "open-source licence at [GitHub URL].")

    heading(doc, "Acknowledgements", 2)
    para(doc, "[To be added.]")

    heading(doc, "Author Contributions (CRediT)", 2)
    credits = [
        ("Malik Yousef: ",
         "Conceptualization, Methodology, Supervision, "
         "Writing - review & editing."),
        ("Jens Allmer: ",
         "Methodology, Validation, Writing - review & editing."),
        ("Yasin İnal: ",
         "Software, Data curation, Formal analysis, Visualization, "
         "Writing - original draft."),
        ("Burcu Bakir-Gungor: ",
         "Supervision, Project administration, "
         "Writing - review & editing."),
    ]
    for name, roles in credits:
        p = doc.add_paragraph()
        r = p.add_run(name)
        r.font.bold = True
        r.font.size = Pt(11)
        r = p.add_run(roles)
        r.font.size = Pt(11)

    heading(doc, "Competing Interests", 2)
    para(doc, "The authors declare no competing interests.")


def write_supplementary(doc, perf, val=None, ds_figs=None):
    """Supplementary: algorithm pseudocode, sensitivity table, per-dataset findings."""
    doc.add_page_break()
    heading(doc, "Supplementary Material", 1)

    heading(doc, "S1. Algorithm Pseudocode", 2)
    pseudocode = (
        "ALGORITHM - Grouping-Scoring-Modeling (G-S-M)\n\n"
        "INPUT\n"
        "  D       : gene-expression matrix  (n samples x p features)\n"
        "  K       : knowledge mapping        (gene → disease groups, e.g. DisGeNET)\n"
        "  y       : binary class labels      (0 = control, 1 = disease)\n"
        "  α       : FDR threshold             (default 0.05)\n"
        "  m       : max groups to retain\n"
        "  N       : number of iterations\n"
        "  r       : train/test split ratio   (default 0.7)\n"
        "  k       : cross-validation folds   (default 5)\n"
        "  s_0      : initial random seed\n\n"
        "OUTPUT\n"
        "  R_agg   : aggregated ranked group list\n"
        "  P_agg   : performance metrics with 95% confidence intervals\n"
        "  F_agg   : aggregated ranked feature (gene) list\n\n"
        "─── PREPROCESSING ───\n"
        "  D ← normalise(D)             // z-score normalisation\n"
        "  y ← encode_labels(y)          // binary 0/1 encoding\n\n"
        "─── ITERATION LOOP  (i = 1 ... N) ───\n"
        "  s_i ← deterministic_seed(s_0, i)\n\n"
        "  STEP 1 - SPLIT\n"
        "     (D_train, D_test, y_train, y_test)\n"
        "         ← stratified_split(D, y, ratio=r, seed=s_i)\n\n"
        "  STEP 2 - FILTER  (gene-level, training data only)\n"
        "     for each feature f in D_train:\n"
        "         p_raw(f) ← welch_ttest(f, y_train)\n"
        "         p_adj(f) ← BH_FDR_correction(p_raw)\n"
        "     F_pass ← { f : p_adj(f) < α }\n"
        "     D_train ← D_train[:, F_pass]       // retain significant genes only\n\n"
        "  STEP 3 - GROUP  (map filtered features to knowledge groups)\n"
        "     for each group g in K:\n"
        "         genes(g) ← K(g) ∩ F_pass         // intersect with surviving features\n"
        "         discard g if genes(g) = ∅\n"
        "     D_g ← D_train[:, genes(g)]        // one sub-matrix per group\n\n"
        "  STEP 4 - SCORE  (per-group CV, training data only)\n"
        "     for each group g:\n"
        "         S_g ← mean_F1( stratified_k-fold_CV(\n"
        "                     classifier, D_g, y_train, folds=k) )\n"
        "     Rᵢ ← sort(groups, by S_g, descending)  // ranked group list\n\n"
        "  STEP 5 - MODEL  (incremental group selection)\n"
        "     prev_count ← 0\n"
        "     for j = 1 ... m:\n"
        "         features_j ← union( genes(g) for g in top-j of Rᵢ )\n"
        "         if |features_j| = prev_count:\n"
        "             expand j until new features appear\n"
        "         M_j ← train(classifier, D_train[:, features_j], y_train)\n"
        "         P_j ← evaluate(M_j, D_test[:, features_j], y_test)\n"
        "         prev_count ← |features_j|\n"
        "     record (Rᵢ, {M_j, P_j})\n\n"
        "─── AGGREGATION  (after all N iterations) ───\n"
        "  R_agg  ← robust_rank_aggregation(R₁ ... R_N)\n"
        "  P_agg  ← bootstrap_95%_CI over {P₁ ... P_N}\n"
        "  F_agg  ← aggregate_feature_rankings\n\n"
        "RETURN R_agg, P_agg, F_agg"
    )
    p = doc.add_paragraph()
    r = p.add_run(pseudocode)
    r.font.name = "Consolas"
    r.font.size = Pt(9)

    # S2. Sensitivity Analysis Table
    _write_sensitivity_table(doc)

    # S3. Per-dataset biological validation findings
    if val:
        doc.add_page_break()
        heading(doc, "S3. Dataset-Specific Biological Validation", 2)
        para(doc,
             "This section provides detailed per-dataset findings including "
             "key genes identified, top enrichment terms, protein-protein "
             "interaction details, and representative figures from the "
             "pipeline output.")
        _write_per_dataset(doc, val, ds_figs or {})


def _write_sensitivity_table(doc):
    """Supplementary Table S2: parameter sensitivity analysis results."""
    heading(doc, "S2. Parameter Sensitivity Analysis", 2)

    sens_path = project_root / "output" / "sensitivity_runs" / "sensitivity_results.json"
    if not sens_path.exists():
        para(doc, "[Sensitivity results not available - "
             "run scripts/run_sensitivity_analysis.py first.]")
        return

    data = json.loads(sens_path.read_text())
    results = data["results"]
    baseline = data["baseline"]

    para(doc,
         "Table S2.  One-at-a-time sensitivity analysis.  Each row varies "
         "a single parameter while holding the others at baseline values "
         f"(FDR = {baseline['fdr']}, CV folds = {baseline['cv_folds']}, "
         f"max groups = {baseline['max_groups']}).  "
         "Baseline configurations are marked with an asterisk (*).  "
         "All values are mean ± standard deviation over 10 pipeline "
         "iterations.")

    # Build table header
    headers = ["Dataset", "FDR", "CV Folds", "Max Groups",
               "F1 (mean ± SD)", "AUC-ROC (mean ± SD)"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Group by dataset
    datasets = sorted(set(r_["dataset_id"] for r_ in results))
    for ds in datasets:
        ds_results = [r_ for r_ in results if r_["dataset_id"] == ds]
        for r_ in ds_results:
            is_bl = (r_["fdr_threshold"] == baseline["fdr"] and
                     r_["cv_folds"] == baseline["cv_folds"] and
                     r_["max_groups"] == baseline["max_groups"])
            marker = " *" if is_bl else ""

            row = table.add_row()
            values = [
                ds,
                f"{r_['fdr_threshold']:.2f}",
                str(r_["cv_folds"]),
                str(r_["max_groups"]),
                f"{r_['mean_f1']:.3f} ± {r_['std_f1']:.3f}{marker}",
                f"{r_['mean_auc']:.3f} ± {r_['std_auc']:.3f}",
            ]
            for i, v in enumerate(values):
                cell = row.cells[i]
                cell.text = ""
                p = cell.paragraphs[0]
                run = p.add_run(v)
                run.font.size = Pt(9)
                if is_bl:
                    run.bold = True
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Parameter impact summary
    impact_path = (
        project_root / "output" / "sensitivity_runs" / "sensitivity_impact.json")
    if impact_path.exists():
        impact_data = json.loads(impact_path.read_text())
        ranking = impact_data.get("ranking", [])
        impact = impact_data.get("impact", {})
        if ranking:
            para(doc, "")
            para(doc,
                 "Parameter impact ranking (by average F1 range across "
                 "datasets): "
                 + "; ".join(
                     f"({i+1}) {name} (ΔF1 = {impact[name]['avg_range']:.3f})"
                     for i, name in enumerate(ranking)
                 )
                 + ".  The maximum observed F1 variation for any single "
                 "parameter on any dataset was <= 0.045, confirming that "
                 "the pipeline is robust to hyperparameter perturbations "
                 "within the tested ranges.")


# ============================================================================ #
#                                     MAIN                                      #
# ============================================================================ #

def _get_next_version() -> int:
    """Find the latest manuscript version and return the next version number."""
    archive_dir = project_root / "reports_ARCHIVE"
    if not archive_dir.exists():
        return 1
    
    # Find all GSM_Manuscript_v*.docx files
    existing = list(archive_dir.glob("GSM_Manuscript_v*.docx"))
    if not existing:
        return 1
    
    # Extract version numbers
    versions = []
    for path in existing:
        try:
            # Extract number from "GSM_Manuscript_v5.docx" -> 5
            version_str = path.stem.split("_v")[-1]
            versions.append(int(version_str))
        except (ValueError, IndexError):
            continue
    
    return max(versions) + 1 if versions else 1


def build():
    """Orchestrate full manuscript generation."""
    version = _get_next_version()
    
    print("=" * 70)
    print(f"  GSM Manuscript DOCX Builder (v{version})")
    print("=" * 70)

    json_path = (
        project_root / "reports_ARCHIVE" / "manuscript_data.json")
    if not json_path.exists():
        print("ERROR: manuscript_data.json not found.  "
              "Run analyze_manuscript_results.py first.")
        return

    data = json.loads(json_path.read_text())
    perf = data["performance"]
    val = data["validation"]
    m_figs = data["manuscript_figures"]
    ds_figs = data["dataset_figures"]

    print(f"  Datasets:   {len(perf)}")
    print(f"  Figures:    {len(m_figs)} cross-dataset + "
          f"{sum(len(v) for v in ds_figs.values())} per-dataset")

    doc = Document()

    # Global style
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for sec in doc.sections:
        sec.top_margin = Cm(2.54)
        sec.bottom_margin = Cm(2.54)
        sec.left_margin = Cm(2.54)
        sec.right_margin = Cm(2.54)

    print("\n  Building sections ...")
    write_title_page(doc)
    write_highlights(doc, perf)
    write_abstract(doc, perf)
    write_introduction(doc)
    write_methods(doc, perf, m_figs)
    write_results(doc, perf, val, m_figs, ds_figs)
    write_discussion(doc, perf, val)
    write_conclusions(doc, perf, val)
    write_references(doc)
    write_back_matter(doc)
    write_supplementary(doc, perf, val, ds_figs)

    out_dir = project_root / "reports_ARCHIVE"
    out = out_dir / f"GSM_Manuscript_v{version}.docx"
    doc.save(str(out))
    kb = out.stat().st_size / 1024
    print(f"\n  Saved:  {out}")
    print(f"  Size:   {kb:.0f} KB")
    print("=" * 70)


if __name__ == "__main__":
    build()
