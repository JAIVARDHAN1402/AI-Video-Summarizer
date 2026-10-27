"""Build the Project-I report from the School's template.

The template file is opened, its instructional body is removed, and the
report is written back into it, so the page size, margins and styles are the
School's own rather than a reconstruction of them.

Every measured number quoted below is read from the system's recorded state
by docs/report_data.py, not typed in by hand, so the report cannot drift away
from what the code actually does.

Run from the project root:  python docs/make_report.py
"""

import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docs.report_data import FACTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(
    ROOT, "BCSE497J Project I Report - Template FINAlL Updated (2).docx")
OUT = os.path.join(ROOT, "BCSE497J Project-I Report - VideoSense.docx")
FIG = os.path.join(ROOT, "docs", "figures")
IP = os.path.join(ROOT, "ip")

BODY_FONT = "Times New Roman"
BODY_PT = 12

TITLE = ("Deadline-Aware Video Summarisation with an SLA-Constrained, "
         "Cost-Aware Scheduler")
TITLE_UPPER = TITLE.upper()


# --------------------------------------------------------------- scaffolding
def clear_body(doc):
    """Remove the template's instructions, keeping its section properties."""
    body = doc.element.body
    for child in list(body.iterchildren()):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def ensure_style(doc, name):
    """Return a paragraph style, creating it if the template lacks it.

    The School's template defines no Caption or List Bullet style, and a
    style python-docx cannot find is a KeyError rather than a fallback.
    """
    try:
        return doc.styles[name]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = doc.styles["Normal"]
        if name.startswith("Heading"):
            st.element.set(qn("w:styleId"), name.replace(" ", ""))
            ppr = st.element.get_or_add_pPr()
            outline = OxmlElement("w:outlineLvl")
            outline.set(qn("w:val"), str(int(name.split()[-1]) - 1))
            ppr.append(outline)
        return st


def style_base(doc):
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(BODY_PT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    pf = normal.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)

    for name, size, bold in (("Heading 1", 16, True), ("Heading 2", 13, True),
                             ("Heading 3", 12, True), ("Caption", 11, False)):
        st = ensure_style(doc, name)
        st.font.name = BODY_FONT
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.italic = (name == "Caption")
        st.paragraph_format.space_before = Pt(12 if name.startswith("H") else 4)
        st.paragraph_format.space_after = Pt(8 if name.startswith("H") else 10)
        st.paragraph_format.keep_with_next = True


def p(doc, text="", size=BODY_PT, bold=False, italic=False, align=None,
      space_after=6, style=None, line_spacing=1.15):
    par = doc.add_paragraph(style=style)
    if text:
        run = par.add_run(text)
        run.font.name = BODY_FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
    par.paragraph_format.space_after = Pt(space_after)
    par.paragraph_format.line_spacing = line_spacing
    if align is not None:
        par.alignment = align
    return par


def justified(doc, text):
    return p(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)


def bullet(doc, text, level=0):
    ensure_style(doc, "List Bullet")
    par = doc.add_paragraph(style="List Bullet")
    run = par.add_run(text)
    run.font.name = BODY_FONT
    run.font.size = Pt(BODY_PT)
    par.paragraph_format.left_indent = Inches(0.35 + 0.3 * level)
    par.paragraph_format.space_after = Pt(4)
    par.paragraph_format.line_spacing = 1.15
    return par


def chapter(doc, number, title):
    doc.add_page_break()
    p(doc, f"CHAPTER {number}", size=16, bold=True,
      align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    h = doc.add_heading(title.upper(), level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return h


def section(doc, text, level=2):
    return doc.add_heading(text, level=level)


def field(par, instruction):
    """Insert a Word field so Word fills the page numbers in on open."""
    r1 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    r1._r.append(fc)

    r2 = par.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instruction
    r2._r.append(it)

    r3 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "separate")
    r3._r.append(fc)

    r4 = par.add_run("Right-click and choose Update Field to build this list.")
    r4.font.name = BODY_FONT
    r4.font.size = Pt(BODY_PT)
    r4.italic = True

    r5 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "end")
    r5._r.append(fc)


_fig_n = [0]
_tab_n = [0]


def figure(doc, filename, caption, width=5.9):
    for base in (FIG, IP, os.path.join(IP, "views")):
        path = os.path.join(base, filename)
        if os.path.exists(path):
            break
    else:
        raise SystemExit(f"figure not found anywhere: {filename}")
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    w = min(width, 6.0)
    if ih / iw * w > 7.4:
        w = 7.4 * iw / ih
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(4)
    par.add_run().add_picture(path, width=Inches(w))

    _fig_n[0] += 1
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run("Figure ")
    r.font.name, r.font.size = BODY_FONT, Pt(11)
    field_seq(cap, "SEQ Figure \\* ARABIC")
    r = cap.add_run("  " + caption)
    r.font.name, r.font.size = BODY_FONT, Pt(11)


def field_seq(par, instruction):
    r1 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    r1._r.append(fc)
    r2 = par.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instruction
    r2._r.append(it)
    r3 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "separate")
    r3._r.append(fc)
    r4 = par.add_run(str(_fig_n[0]))
    r4.font.name, r4.font.size = BODY_FONT, Pt(11)
    r5 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "end")
    r5._r.append(fc)


def table(doc, caption, rows, widths=None, header=True, font=10.5):
    """Caption above the table, as the template requires."""
    _tab_n[0] += 1
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Table {_tab_n[0]}  {caption}")
    r.font.name, r.font.size = BODY_FONT, Pt(11)

    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for c, val in zip(cells, row):
            c.text = ""
            par = c.paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.line_spacing = 1.0
            run = par.add_run(str(val))
            run.font.name = BODY_FONT
            run.font.size = Pt(font)
            run.bold = (header and i == 0)
    if widths:
        for col, w in zip(t.columns, widths):
            for cell in col.cells:
                cell.width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def code_block(doc, text):
    for line in text.split("\n"):
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing = 1.0
        par.paragraph_format.left_indent = Inches(0.3)
        run = par.add_run(line if line else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(9)


def main():
    import docs.report_chapters as chapters
    import docs.report_front as front

    doc = Document(TEMPLATE)
    clear_body(doc)
    style_base(doc)

    front.build(sys.modules[__name__], doc)
    chapters.build(sys.modules[__name__], doc, FACTS)

    doc.save(OUT)
    print("wrote", OUT)
    print(f"  figures: {_fig_n[0]}   tables: {_tab_n[0]}")
    print("  open in Word, select all (Ctrl+A) and press F9 to build the "
          "contents and the lists of figures and tables.")


if __name__ == "__main__":
    main()
