# -*- coding: utf-8 -*-
"""Синтетический стандарт PKX-STD 330 — потери именно на этапе группировки.

  P1  правило задано только примерами, и примеры стоят в начале следующей страницы (7.1, 6.4);
  P2  правило существует только в образце датащита «до/после» в приложении (7.2 → Appendix C);
  P3  реестр кодов разорван между страницами, продолжение — без заголовка и без подписи (7.3);
  P4  ссылка на свой пункт с переопределением («per 6.2, except …», 8.1);
  P5  подпись атрибута в датащите не синоним имени правила («Duty», «Capacity»).
"""
import json
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория
PDF = ROOT / "graphs/structured_documents/data/input/PKX-STD-330_package_compressor_normalization_rules.pdf"
XLSX = ROOT / "graphs/rag_agent/data/input/pkx_compressor_test_corpus.xlsx"
EXPECTED = Path(__file__).parent / "expected" / "pkx_expected.json"

HEAD = ("<p style='font-size:8pt;color:#555'>PKX-STD 330 — Fictitious / Internal Test Corpus Only<br>"
        "Package Compressors and Air Treatment, Edition 1<br>Page {n} of {total}</p>")
CSS = """
body { font-family: sans-serif; font-size: 9.5pt; }
h1 { font-size: 13pt; margin: 6pt 0 3pt 0; }
p { margin: 2pt 0 3pt 0; }
.ex { margin-left: 18pt; }
table { border-collapse: collapse; margin: 4pt 0; }
td, th { border: 1px solid #333; padding: 2pt 4pt; font-size: 9pt; }
"""


def ex(raw, norm):
    return f"<p class='ex'>EXAMPLE: \"{raw}\"  →  \"{norm}\"</p>"


def table(head, rows):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>" if head else ""
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{h}{b}</table>"


PAGES = [
    """<h1>PKX-STD 330</h1>
<p><b>Attribute Normalization Rules for Package Compressors and Air Treatment Equipment</b></p>
<h1>1  Scope</h1>
<p>1.1  This document specifies presentation-normalization rules for attribute values recorded on data sheets for
rotary screw compressors, reciprocating compressors, rotary lobe blowers, compressed air receivers and refrigerated
air dryers.</p>
<p>1.2  Normalization changes presentation only and never the technical substance of a value.</p>
<h1>6  Units of Measurement</h1>
<p>6.1  Free air delivery shall be expressed in cubic metres per minute ("m3/min") to two decimal places, preceded
by a single space.</p>"""
    + ex("12.5 m3/min", "12.50 m3/min")
    + """<p>6.2  Gauge pressure (working, maximum or set pressure) shall be expressed in bar gauge to one decimal place,
written with a single space and the unit "bar(g)".</p>"""
    + ex("7 barg", "7.0 bar(g)")
    + """<p>6.3  Receiver volume shall be expressed in litres as a whole number followed by a single space and the unit
symbol written as an upper-case "L".</p>
<p>6.4  Pressure dew point shall be written as a whole number with its sign always shown, a single space, and "°C",
as in the following examples:</p>""",
    # страница 2 начинается с примеров пункта 6.4 (P1)
    ex("3C", "+3 °C")
    + ex("-40 C", "-40 °C")
    + """<h1>7  Wording and Codes</h1>
<p>7.1  Cooling method shall be written as shown in the following examples:</p>""",
    # страница 3: примеры 7.1 (P1), 7.2 (P2), начало реестра (P3)
    ex("Cooled, air", "Air-cooled")
    + ex("Cooled, water", "Water-cooled")
    + """<p>7.2  Mounting arrangement values shall be written in the form shown in the sample data sheet of Appendix C.</p>
<p>7.3  Compressor type shall be recorded only as a code of Table 2. A type not listed in Table 2 shall be
reproduced unchanged and flagged for review.</p>
<p><b>Table 2 — Compressor type codes</b></p>"""
    + table(["Compressor type", "Code"],
            [["Oil-injected rotary screw", "CMP-OIS"], ["Oil-free rotary screw", "CMP-OFS"]]),
    # страница 4: продолжение таблицы без заголовка (P3), ссылка с переопределением (P4)
    table(None, [["Rotary lobe", "CMP-RL"], ["Reciprocating, single stage", "CMP-R1"],
                 ["Reciprocating, two stage", "CMP-R2"]])
    + """<h1>8  Special Cases</h1>
<p>8.1  Blower discharge pressure shall be normalized per 6.2, except that it shall be expressed in millibar gauge
as a whole number with the unit "mbar(g)".</p>"""
    + ex("0.8 barg", "800 mbar(g)")
    + """<p>8.2  Where a value carries a unit this document does not list, it shall be reproduced unchanged and flagged
for manual review.</p>
<h1>9  Attributes Excluded from Reformatting</h1>
<p>• Serial numbers and purchaser tag numbers.</p>""",
    # страница 5: приложение C — единственное место, где задана форма монтажа (P2)
    """<h1>Appendix C  Sample Data Sheet, Before and After Normalization</h1>
<p>(Normative.) The normalized column is the required presentation form.</p>"""
    + table(["Attribute", "Recorded value", "Normalized value"],
            [["Duty", "12.5 m3/min", "12.50 m3/min"],
             ["Mounting arrangement", "Mounted on skid", "Skid-mounted"],
             ["Mounting arrangement", "Mounted on base frame", "Base-frame-mounted"],
             ["Mounting arrangement", "Mounted on wall", "Wall-mounted"],
             ["Cooling", "Cooled, air", "Air-cooled"]]),
]

ROWS = [
    ("Rotary Screw Compressor SC-01", "Free air delivery", "12.5 m3/min", "12.50 m3/min", False),
    ("Rotary Screw Compressor SC-01", "Duty", "12.50m3/min", "12.50 m3/min", False),
    ("Rotary Screw Compressor SC-01", "Working pressure", "7 barg", "7.0 bar(g)", False),
    ("Rotary Screw Compressor SC-01", "Working pressure", "7.0 bar g", "7.0 bar(g)", False),
    ("Rotary Screw Compressor SC-01", "Cooling method", "Cooled, air", "Air-cooled", False),
    ("Rotary Screw Compressor SC-01", "Compressor type", "Oil-injected rotary screw", "CMP-OIS", False),
    ("Rotary Screw Compressor SC-01", "Mounting arrangement", "Mounted on skid", "Skid-mounted", False),
    ("Reciprocating Compressor RC-01", "Compressor type", "Reciprocating, two stage", "CMP-R2", False),
    ("Reciprocating Compressor RC-01", "Capacity", "3.2 m3/min", "3.20 m3/min", False),
    ("Reciprocating Compressor RC-01", "Cooling method", "Cooled, water", "Water-cooled", False),
    ("Reciprocating Compressor RC-01", "Set pressure", "10 barg", "10.0 bar(g)", False),
    ("Rotary Lobe Blower BL-01", "Compressor type", "Rotary lobe", "CMP-RL", False),
    ("Rotary Lobe Blower BL-01", "Discharge pressure", "0.8 barg", "800 mbar(g)", False),
    ("Rotary Lobe Blower BL-01", "Discharge pressure", "0.6 bar(g)", "600 mbar(g)", False),
    ("Rotary Lobe Blower BL-01", "Mounting arrangement", "Mounted on base frame", "Base-frame-mounted", False),
    ("Rotary Lobe Blower BL-01", "Duty", "25 m3/min", "25.00 m3/min", False),
    ("Air Receiver AR-01", "Volume", "2000 l", "2000 L", False),
    ("Air Receiver AR-01", "Volume", "2000L", "2000 L", False),
    ("Air Receiver AR-01", "Maximum pressure", "11 barg", "11.0 bar(g)", False),
    ("Air Receiver AR-01", "Mounting arrangement", "Mounted on wall", "Wall-mounted", False),
    ("Refrigerated Dryer RD-01", "Pressure dew point", "3C", "+3 °C", False),
    ("Refrigerated Dryer RD-01", "Pressure dew point", "+5 °C", "+5 °C", False),
    ("Refrigerated Dryer RD-01", "Pressure dew point", "5 C", "+5 °C", False),
    ("Refrigerated Dryer RD-01", "Cooling method", "Cooled, air", "Air-cooled", False),
]


def main():
    doc = fitz.open()
    for n, body in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(fitz.Rect(50, 40, 545, 800), HEAD.format(n=n, total=len(PAGES)) + body, css=CSS)
    doc.save(PDF)
    pd.DataFrame([r[:3] for r in ROWS], columns=["Item", "Attribute", "Value"]).to_excel(XLSX, index=False)
    EXPECTED.write_text(json.dumps([{"expected": r[3], "review": r[4]} for r in ROWS], ensure_ascii=False,
                                   indent=1), encoding="utf-8")
    print(PDF, XLSX, len(ROWS))


if __name__ == "__main__":
    main()
