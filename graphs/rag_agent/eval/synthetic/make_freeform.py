# -*- coding: utf-8 -*-
"""Три стандарта в свободной форме — как их пишут в жизни, без общего образца.

- WHX-BUL-031 — служебная записка (memo) по устьевому оборудованию: правила прозой в письме,
  а в конце поправка, которая отменяет часть пункта 1 (запятая в тысячах). Проверка, что
  применяется последняя редакция, а не первое прочитанное.
- PLX-DD-007 — словарь данных по трубопроводам и резервуарам: почти всё в таблицах, десятичная
  ЗАПЯТАЯ, текстовые значения строчными, цепочка «синоним → наименование → код».
- RTE-GN-450 — чек-лист SHALL / SHALL NOT по роторному и электрооборудованию: таблица
  «Wrong | Right | Why», ссылка на другой пункт, «приблизительное значение не трогать».

Значения таблиц не встречаются в тексте документов (скрипт проверяет и падает). Запуск:
    venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.make_freeform
"""
import json
import re
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PDF_DIR = ROOT / "graphs/structured_documents/data/input"
XLSX_DIR = ROOT / "graphs/rag_agent/data/input"
EXPECTED_DIR = Path(__file__).parent / "expected"

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; }
h1 { font-size: 13pt; margin: 6pt 0 3pt 0; }
h2 { font-size: 11pt; margin: 6pt 0 2pt 0; }
p { margin: 2pt 0 4pt 0; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 4pt 0; }
td, th { border: 1px solid #333; padding: 2pt 4pt; font-size: 8.5pt; }
"""


def table(head, rows):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    return "<table>" + h + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                                   for r in rows) + "</table>"


# ---------------------------------------------------------------- WHX-BUL-031 (memo)
WHX_PAGES = [
    """<p style='font-size:8pt;color:#555'>WHX Operations — Fictitious test corpus, not for engineering use</p>
<h1>INTERNAL MEMO — WHX-BUL-031</h1>
<p><b>To:</b> Materials catalogue team, all asset areas<br><b>From:</b> Wellhead &amp; Pressure Control Technical Authority<br>
<b>Subject:</b> How we want wellhead, tree and BOP data entered from now on<br><b>Date:</b> 14 March, rev. 2</p>
<p>Team, after the last audit we found the same wellhead equipment entered five different ways, which breaks our spare-parts
matching. Below is how each field is to be filled in. Where I don't mention a field, leave it the way the vendor wrote it.
Please read to the end — there is an amendment at the bottom that changes item 1.</p>
<p><b>1. Rated working pressure.</b> We always record this in psi, as a whole number, with a comma between the thousands, so
2,000 / 3,000 / 5,000 / 10,000 / 15,000 / 20,000 psi. Those six are the only ratings that exist for this kit, so if a vendor sheet
gives the pressure in bar or MPa, convert it (1 bar = 14.5038 psi, 1 MPa = 145.038 psi) and then take whichever of the six
ratings is nearest. Don't write "psig", "WP" or "RWP" after the number, and "K" for thousands (as in "5K") is not acceptable in
the catalogue either — spell the number out. Example: a sheet saying 1035 bar is entered as 15,000 psi.</p>
<p><b>2. Temperature class.</b> This is the API 6A letter pair — lower class letter, a hyphen, upper class letter, e.g. L-V.
Vendors also write it run together, with a slash or with "to" between the letters; it is the same class and goes in with the
hyphen, capital letters. If what you have is an actual temperature range in degrees rather than letters, do not try to work
out the class — copy it in as it is and flag it to us.</p>""",
    """<p style='font-size:8pt;color:#555'>WHX-BUL-031 rev. 2 — page 2</p>
<p><b>3. Material class.</b> Capitals, no spaces: AA, BB, CC, DD, EE, FF or HH. For sour service the class carries "-NL" when no
partial-pressure limit applies, so DD-NL, and so on. Lower case, a space or nothing in place of the hyphen all mean the same.</p>
<p><b>4. PSL.</b> Write "PSL", one space, the level digit, and a capital G straight after the digit if the equipment was gas
tested — PSL 2G is the format. "Level 3" on its own also means PSL 3, and a hyphen or no space between PSL and the digit is
just sloppy typing of the same thing.</p>
<p><b>5. Nominal bore.</b> Bores come in a fixed series: 1-13/16, 2-1/16, 2-9/16, 3-1/16, 4-1/16, 5-1/8, 7-1/16, 9, 11,
13-5/8, 16-3/4 and 18-3/4 inches. Write the fraction with a hyphen between the whole and the fraction, then a space and "in."
with the full stop, e.g. 2-9/16 in. A decimal inch figure from a datasheet is the nearest bore of the series, and a bore given
in millimetres is converted (25.4 mm per inch) and likewise taken to the nearest bore of the series.</p>
<p><b>6. End connections.</b> These go in as our connection codes, from the little table below. Use the code only when the
description matches the table; for anything else, type it in as received and send it to the data team — we will add new
codes, you must not pick the "closest" one.</p>"""
    + table(["Connection", "Code"], [["6B flange", "EC-6B"], ["6BX flange", "EC-6BX"], ["Studded", "EC-ST"],
                                     ["Clamp hub", "EC-HUB"], ["Threaded line pipe", "EC-LP"]])
    + """<p><b>AMENDMENT 1 (supersedes part of item 1).</b> The thousands comma in item 1 is causing import errors in the ERP
upload. With immediate effect, write the working pressure WITHOUT any thousands separator — 10000 psi, not 10,000 psi.
Everything else in item 1 (psi, whole number, nearest of the six ratings, no suffixes) still applies.</p>
<p>Thanks — Technical Authority, Wellhead &amp; Pressure Control.</p>""",
]

WHX_ROWS = [
    # Item, Attribute, Value, expected, review
    ("Wellhead WH-01", "Working pressure", "5000 psi", "5000 psi", False),
    ("Wellhead WH-01", "Working pressure", "345 bar", "5000 psi", False),
    ("Christmas tree XT-02", "Working pressure", "103.4 MPa", "15000 psi", False),
    ("Christmas tree XT-02", "Working pressure", "10K psi WP", "10000 psi", False),
    ("BOP stack BOP-03", "Working pressure", "3,000 psig", "3000 psi", False),
    ("Wellhead WH-01", "Temperature class", "LU", "L-U", False),
    ("Christmas tree XT-02", "Temperature class", "P/U", "P-U", False),
    ("BOP stack BOP-03", "Temperature class", "K to U", "K-U", False),
    ("Wellhead WH-01", "Temperature class", "-20 to 250 F", "-20 to 250 F", True),
    ("Wellhead WH-01", "Material class", "ee nl", "EE-NL", False),
    ("Christmas tree XT-02", "Material class", "ff", "FF", False),
    ("BOP stack BOP-03", "Material class", "HHNL", "HH-NL", False),
    ("Wellhead WH-01", "PSL", "psl-4", "PSL 4", False),
    ("Christmas tree XT-02", "PSL", "PSL3G", "PSL 3G", False),
    ("BOP stack BOP-03", "PSL", "level 1", "PSL 1", False),
    ("Wellhead WH-01", "Nominal bore", "3.06\"", "3-1/16 in.", False),
    ("Christmas tree XT-02", "Nominal bore", "179 mm", "7-1/16 in.", False),
    ("BOP stack BOP-03", "Nominal bore", "13 5/8", "13-5/8 in.", False),
    ("Wellhead WH-01", "Nominal bore", "11\"", "11 in.", False),
    ("Wellhead WH-01", "End connection", "6BX flange", "EC-6BX", False),
    ("Christmas tree XT-02", "End connection", "studded", "EC-ST", False),
    ("BOP stack BOP-03", "End connection", "Clamp hub", "EC-HUB", False),
    ("BOP stack BOP-03", "End connection", "Grayloc hub", {"re": r"^(?:(?:pending registration|ambiguous match): )?Grayloc hub$"}, True),
]

# ------------------------------------------------------- PLX-DD-007 (data dictionary)
PLX_PAGES = [
    """<p style='font-size:8pt;color:#555'>PLX Midstream — Fictitious test corpus, not for engineering use</p>
<h1>PLX-DD-007 Pipeline and Storage Data Dictionary</h1>
<p>Edition 4. This dictionary fixes the stored form of pipeline and tank attributes. Read Section 1 first; the per-attribute
rows in Table 2 override nothing in Section 1, they add to it.</p>
<h2>1 Conventions</h2>"""
    + table(["No.", "Convention"], [
        ["1.1", "The decimal separator is the comma (,). A point used as decimal separator is replaced by a comma. No thousands separator."],
        ["1.2", "The unit symbol given in Table 2 follows the number after one space."],
        ["1.3", "Text values are stored in lower case. Codes from Tables 3 and 4 are stored exactly as printed (upper case)."],
        ["1.4", "Converted values are rounded to the number of decimals given in Table 2 (half up)."],
        ["1.5", "A value that is not covered by Tables 3, 4 or 5 is kept as received and reported to the data owner."],
    ])
    + """<h2>2 Attributes</h2><p><b>Table 2 — Attribute formats</b></p>"""
    + table(["Attribute", "Unit", "Decimals", "Conversion from other units", "Notes"], [
        ["Outside diameter", "mm", "1", "in × 25,4", "—"],
        ["Wall thickness", "mm", "2", "in × 25,4", "—"],
        ["Design pressure", "barg", "1", "psig × 0,0689476; kPag ÷ 100", "gauge pressure only"],
        ["Design temperature", "°C", "0", "(°F − 32) ÷ 1,8", "—"],
        ["Nominal capacity", "m³", "0", "bbl × 0,158987", "tanks only"],
        ["Roof type", "—", "—", "—", "code from Table 3"],
        ["Coating system", "—", "—", "—", "code from Table 4"],
        ["Line pipe grade", "—", "—", "—", "API 5L, grade, PSL — see 2.1"],
        ["Service", "—", "—", "—", "free text, per 1.3"],
    ])
    + """<p><b>2.1</b> Line pipe grade is written "API 5L", the grade, and the PSL with no space between PSL and its digit,
single spaces between the three parts (pattern: API 5L &lt;grade&gt; PSL&lt;n&gt;). The words "Gr", "Gr." and "Grade" in front of
the grade are dropped.</p>""",
    """<p style='font-size:8pt;color:#555'>PLX-DD-007 Edition 4 — page 2</p>
<h2>3 Code tables</h2><p><b>Table 3 — Roof type codes</b></p>"""
    + table(["Roof type", "Code"], [["Fixed cone roof", "RF-CN"], ["Fixed dome roof", "RF-DM"],
                                    ["External floating roof", "RF-EF"], ["Internal floating roof", "RF-IF"]])
    + """<p><b>Table 4 — Coating system codes</b></p>"""
    + table(["Coating system", "Code"], [["Three-layer polyethylene", "CS-3LPE"], ["Fusion bonded epoxy", "CS-FBE"],
                                         ["Three-layer polypropylene", "CS-3LPP"], ["Concrete weight coating", "CS-CWC"]])
    + """<p><b>Table 5 — Recognised short names</b> (the short name stands for the full name, which is then coded per Table 3 or 4)</p>"""
    + table(["Short names", "Full name"], [["EFR; floating roof (external)", "External floating roof"],
                                           ["IFR; floating roof (internal)", "Internal floating roof"],
                                           ["FBE", "Fusion bonded epoxy"], ["3LPE; 3-layer PE", "Three-layer polyethylene"],
                                           ["3LPP; 3-layer PP", "Three-layer polypropylene"]]),
]

PLX_ROWS = [
    ("Pipeline PL-11", "Outside diameter", "16 in", "406,4 mm", False),
    ("Pipeline PL-11", "Outside diameter", "8.625\"", "219,1 mm", False),
    ("Pipeline PL-11", "Outside diameter", "508 mm", "508,0 mm", False),
    ("Pipeline PL-11", "Wall thickness", "0.500 in", "12,70 mm", False),
    ("Pipeline PL-11", "Wall thickness", "6.4 mm", "6,40 mm", False),
    ("Pipeline PL-11", "Wall thickness", "11,13mm", "11,13 mm", False),
    ("Pipeline PL-11", "Design pressure", "1440 psig", "99,3 barg", False),
    ("Pipeline PL-11", "Design pressure", "85 barg", "85,0 barg", False),
    ("Storage tank TK-12", "Design pressure", "4500 kPag", "45,0 barg", False),
    ("Pipeline PL-11", "Design temperature", "150 F", "66 °C", False),
    ("Storage tank TK-12", "Design temperature", "-29 °C", "-29 °C", False),
    ("Pipeline PL-11", "Design temperature", "200 deg F", "93 °C", False),
    ("Storage tank TK-12", "Nominal capacity", "50000 bbl", "7949 m³", False),
    ("Storage tank TK-12", "Nominal capacity", "12 000 m3", "12000 m³", False),
    ("Storage tank TK-12", "Roof type", "External floating roof", "RF-EF", False),
    ("Storage tank TK-12", "Roof type", "IFR", "RF-IF", False),
    ("Storage tank TK-12", "Roof type", "fixed cone roof", "RF-CN", False),
    ("Storage tank TK-12", "Roof type", "Geodesic dome", {"re": r"^(?:(?:pending registration|ambiguous match): )?(?:Geodesic dome|geodesic dome)$"}, True),
    ("Pipeline PL-11", "Coating system", "FBE", "CS-FBE", False),
    ("Pipeline PL-11", "Coating system", "Three-layer polypropylene", "CS-3LPP", False),
    ("Pipeline PL-11", "Coating system", "3-layer PE", "CS-3LPE", False),
    ("Pipeline PL-11", "Coating system", "Coal tar enamel", {"re": r"^(?:(?:pending registration|ambiguous match): )?(?:Coal tar enamel|coal tar enamel)$"}, True),
    ("Pipeline PL-11", "Line pipe grade", "X65 PSL 2", "API 5L X65 PSL2", False),
    ("Pipeline PL-11", "Line pipe grade", "API5L-X52 PSL1", "API 5L X52 PSL1", False),
    ("Pipeline PL-11", "Line pipe grade", "API 5L Grade B PSL 2", "API 5L B PSL2", False),
    ("Pipeline PL-11", "Service", "Sour Gas", "sour gas", False),
    ("Storage tank TK-12", "Service", "CRUDE OIL", "crude oil", False),
]

# ------------------------------------------------- RTE-GN-450 (checklist SHALL / SHALL NOT)
RTE_PAGES = [
    """<p style='font-size:8pt;color:#555'>RTE Engineering — Fictitious test corpus, not for engineering use</p>
<h1>RTE-GN-450 Rotating and Electrical Equipment — Data Quality Guide</h1>
<p>This guide is a checklist. Every line is either a SHALL or a SHALL NOT. Where a line refers to another line, both apply.</p>
<h2>A. General</h2>
<ul>
<li><b>A1</b> SHALL put one space between a number and its unit.</li>
<li><b>A2</b> SHALL NOT normalize a value marked as approximate ("approx.", "ca.", "~"): copy it exactly as received and flag it.</li>
<li><b>A3</b> SHALL NOT add to a value anything the source did not give (no default protection level, no default gas group).</li>
</ul>
<h2>B. Ratings</h2>
<ul>
<li><b>B1 Power</b> SHALL be in kW, whole number. hp SHALL be converted with 1 hp = 0.7457 kW; MW SHALL be converted to kW.</li>
<li><b>B2 Voltage</b> from 1000 V upward SHALL be in kV with exactly one decimal; below 1000 V SHALL be in V, whole number.
SHALL NOT carry "AC"/"VAC" — the unit is V or kV only.</li>
<li><b>B3 Speed</b> SHALL be a whole number followed by "rpm" in lower case. "r/min", "min-1" and "RPM" are the same unit.</li>
<li><b>B4 Gas flow</b> of compressors SHALL be in Nm3/h (plain digit 3, capital N, lower-case m and h), whole number.
Flow in MMSCFD SHALL be converted with 1 MMSCFD = 1177 Nm3/h (company factor — use this one).</li>
</ul>
<h2>C. Markings</h2>
<ul>
<li><b>C1 Ex marking</b> SHALL read "Ex", then the protection type in lower case, the gas group, the temperature class and,
if given, the equipment protection level written capital-then-lower-case — each separated by one space. Hyphens between parts
SHALL be treated as spaces. The obsolete prefix "EEx" SHALL be written "Ex". Per A3 nothing missing is added.</li>
<li><b>C2 Ingress protection</b> SHALL be "IP" in capitals immediately followed by the two digits.</li>
</ul>""",
    """<p style='font-size:8pt;color:#555'>RTE-GN-450 — page 2</p>
<h2>D. Driver type</h2>
<ul><li><b>D1</b> SHALL be the code from Table D. A description not in Table D SHALL NOT be coded; per A2-style handling it is
copied as received and flagged.</li></ul>
<p><b>Table D — Driver codes</b></p>"""
    + table(["Driver", "Code"], [["Induction motor", "ROT-IM"], ["Synchronous motor", "ROT-SM"], ["Gas turbine", "ROT-GT"],
                                 ["Steam turbine", "ROT-ST"], ["Gas engine", "ROT-GE"]])
    + """<h2>E. Common mistakes</h2>"""
    + table(["Wrong", "Right", "Why"], [
        ["315KW", "315 kW", "A1, B1"],
        ["11000 V", "11.0 kV", "B2"],
        ["1485 RPM", "1485 rpm", "B3"],
        ["Ex-d-IIA-T3", "Ex d IIA T3", "C1, A3 (no level added)"],
        ["IP-54", "IP54", "C2"],
        ["~250 kW", "~250 kW", "A2"],
    ]),
]

RTE_ROWS = [
    ("Compressor K-21", "Rated power", "180 hp", "134 kW", False),
    ("Motor M-22", "Rated power", "1.2 MW", "1200 kW", False),
    ("Pump P-23", "Rated power", "75kW", "75 kW", False),
    ("Compressor K-21", "Rated power", "approx. 300 kW", "approx. 300 kW", True),
    ("Motor M-22", "Rated voltage", "6600 V", "6.6 kV", False),
    ("Pump P-23", "Rated voltage", "415 VAC", "415 V", False),
    ("Motor M-22", "Rated voltage", "13.8kV", "13.8 kV", False),
    ("Motor M-22", "Rated voltage", "3.3 kV", "3.3 kV", False),
    ("Pump P-23", "Rated voltage", "690V", "690 V", False),
    ("Motor M-22", "Speed", "2980 RPM", "2980 rpm", False),
    ("Pump P-23", "Speed", "1500 r/min", "1500 rpm", False),
    ("Compressor K-21", "Speed", "3600 min-1", "3600 rpm", False),
    ("Motor M-22", "Ex marking", "Ex-db-IIC-T3-Gb", "Ex db IIC T3 Gb", False),
    ("Motor M-22", "Ex marking", "EEx e II T3", "Ex e II T3", False),
    ("Pump P-23", "Ex marking", "Ex DB IIB T4 GB", "Ex db IIB T4 Gb", False),
    ("Motor M-22", "Ingress protection", "IP 56", "IP56", False),
    ("Pump P-23", "Ingress protection", "ip65", "IP65", False),
    ("Compressor K-21", "Driver type", "Gas turbine", "ROT-GT", False),
    ("Pump P-23", "Driver type", "induction motor", "ROT-IM", False),
    ("Compressor K-21", "Driver type", "Steam turbine", "ROT-ST", False),
    ("Pump P-23", "Driver type", "Electric motor", {"re": r"^(?:(?:pending registration|ambiguous match): )?Electric motor$"}, True),
    ("Compressor K-21", "Gas flow", "25 MMSCFD", "29425 Nm3/h", False),
    ("Compressor K-21", "Gas flow", "12500 Nm³/h", "12500 Nm3/h", False),
    ("Compressor K-21", "Gas flow", "8000 NM3/H", "8000 Nm3/h", False),
]

# Атрибуты, чьё значение по замыслу берётся из таблицы кодов документа: запись и должна
# совпадать с пунктом таблицы, поэтому проверка «нет в тексте» к ним не относится.
CODED = {"End connection", "Roof type", "Coating system", "Driver type"}

DOCS = [
    ("WHX-BUL-031_wellhead_data_entry_bulletin", "whx_freeform", WHX_PAGES, WHX_ROWS),
    ("PLX-DD-007_pipeline_storage_data_dictionary", "plx_freeform", PLX_PAGES, PLX_ROWS),
    ("RTE-GN-450_rotating_electrical_data_quality_guide", "rte_freeform", RTE_PAGES, RTE_ROWS),
]


def build(stem: str, key: str, pages: list[str], rows: list[tuple]) -> None:
    pdf = PDF_DIR / f"{stem}.pdf"
    doc = fitz.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(fitz.Rect(50, 40, 545, 800), body, css=CSS)
    doc.save(pdf)
    text = re.sub(r"\s+", " ", "".join(p.get_text() for p in fitz.open(pdf))).lower()
    leaked = [r[2] for r in rows if r[1] not in CODED and r[2].lower() in text and len(r[2]) > 3]
    if leaked:
        raise SystemExit(f"{stem}: значения из таблицы есть в тексте документа: {leaked}")
    pd.DataFrame([r[:3] for r in rows], columns=["Item", "Attribute", "Value"]).to_excel(
        XLSX_DIR / f"{key}.xlsx", index=False)
    (EXPECTED_DIR / f"{key}.json").write_text(json.dumps(
        [{"expected": r[3], "review": r[4]} for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{stem}: {len(pages)} стр., строк {len(rows)}")


if __name__ == "__main__":
    for args in DOCS:
        build(*args)
