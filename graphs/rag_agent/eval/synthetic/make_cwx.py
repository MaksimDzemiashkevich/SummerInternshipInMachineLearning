# -*- coding: utf-8 -*-
"""Синтетический стандарт CWX-STD 221 + тестовый датащит с эталоном.

Стресс-точки (каждая — отдельный способ, которым правило не доходит до строки):
  S1  правило общее, а имя атрибута в датащите с именем правила не совпадает
      («Tank dimensions», «Plate size», «Dimensions (LxWxH)» против «multi-dimension value»);
  S2  связь «атрибут → правило» есть ТОЛЬКО в worked example / edge cases (precharge = давление,
      approach = разность температур в K);
  S3  десятичные знаки зависят от класса оборудования (Table 1) — правило про предмет;
  S4  ссылка на пункт другого, загруженного документа (6.2 OGX-STD 410 — температура);
  S5  ссылка на документ, которого нет в базе (ZZX-STD 000) — обязана стать пометкой;
  S6  форма числа задана только в приложении шаблоном «<n.n> kW»;
  S7  EXAMPLE правила оказывается в начале следующей страницы;
  S8  матрица идентичности называет те же атрибуты и отъедает слоты ретрива;
  S9  одна единица (kW) — два разных правила (мощность <n.n>, холодопроизводительность <n>).
"""
import json
import sys
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория
PDF = ROOT / "graphs/structured_documents/data/input/CWX-STD-221_chilled_water_pumping_normalization_rules.pdf"
XLSX = ROOT / "graphs/rag_agent/data/input/cwx_chilled_water_test_corpus.xlsx"
EXPECTED = Path(__file__).parent / "expected" / "cwx_expected.json"

HEAD = ("<p style='font-size:8pt;color:#555'>CWX-STD 221 — Fictitious / Internal Test Corpus Only<br>"
        "Chilled Water &amp; Pumping Equipment, Edition 1<br>Not for engineering use<br>Page {n} of {total}</p>")

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; }
h1 { font-size: 13pt; margin: 6pt 0 3pt 0; }
h2 { font-size: 11pt; margin: 5pt 0 2pt 0; }
p { margin: 2pt 0 3pt 0; }
.ex { margin-left: 18pt; }
table { border-collapse: collapse; margin: 4pt 0; }
td, th { border: 1px solid #333; padding: 2pt 4pt; font-size: 9pt; }
"""


def ex(raw, norm):
    return f"<p class='ex'>EXAMPLE: \"{raw}\"  →  \"{norm}\"</p>"


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


PAGES = [
    # 1 — title
    """<h1>CWX-STD 221</h1>
<p><b>Attribute Normalization Rules for Chilled Water and Pumping Equipment</b></p>
<p>Centrifugal Pumps, Chillers, Cooling Towers, Plate Heat Exchangers, and Expansion Tanks</p>
<p>Issued by the Internal Data Standards Working Group — Fictitious Test Corpus Series. Edition 1 — Not for engineering use.</p>
<p>This document is a wholly fictitious, internal working paper prepared solely as a synthetic test artifact for
software that applies attribute-presentation normalization rules to chilled water equipment records. It does not
correspond to any published standard and shall not be relied upon for any engineering purpose.</p>""",
    # 2 — scope, general
    """<h1>1  Scope</h1>
<p>1.1  This document specifies presentation-normalization rules for attribute values recorded on data sheets for the
following equipment classes: centrifugal pumps; water chillers; cooling towers; plate heat exchangers; and
expansion tanks.</p>
<p>1.2  Normalization performed under this document changes presentation only. It does not alter the technical
substance of a recorded value.</p>
<h1>2  Normative References</h1>
<p>OGX-STD 410 (multi-equipment normalization rules) and ZZX-STD 000 (acoustic data presentation). Both are cited
for internal consistency of the test corpus only.</p>
<h1>4  General Principles</h1>
<p>4.1  Every rule shall be applied deterministically and idempotently: re-applying a rule to an already-normalized
value leaves it unchanged.</p>
<p>4.2  Where an attribute value is missing or marked "N/A", no rule of this document shall be applied.</p>
<h1>5  Item Identity</h1>
<p>5.1  Two records describe the same item only where every key attribute listed for the equipment class in
Clause 9 is equal in technical substance. Consistent normalization of shared attributes is never evidence of
identity.</p>""",
    # 3 — units part 1; the EXAMPLE of 6.2 is pushed to next page (S7)
    """<h1>6  Units of Measurement</h1>
<p>6.1  Volumetric water flow shall be expressed in cubic metres per hour ("m3/h"), preceded by a single space and
rounded to the number of decimal places given for the equipment class in Table 1.</p>
<p><b>Table 1 — Decimal places for water flow by equipment class</b></p>"""
    + table(["Equipment class", "Decimal places"],
            [["Centrifugal pump", "1"], ["Water chiller", "0"], ["Cooling tower", "0"],
             ["Plate heat exchanger", "1"]])
    + ex("45 m3/h", "45.0 m3/h")
    + """<p>6.2  Pressure, head and pressure drop shall be expressed in kilopascals as a whole number followed by a
single space and "kPa". A value recorded in bar shall first be converted using 1 bar = 100 kPa.</p>""",
    # 4 — rest of units
    ex("2.5 bar", "250 kPa")
    + """<p>6.3  Temperature attributes shall be normalized per 6.2 of OGX-STD 410, which this document adopts by
reference for temperatures only.</p>
<p>6.4  Electrical power (motor power, fan power, compressor input power) shall be written with the unit "kW"
preceded by a single space, in the form given in Appendix A.</p>
<p>6.5  Cooling capacity shall be expressed in kilowatts as a whole number followed by a single space and "kW". A
capacity recorded in refrigeration tons shall be converted using 1 TR = 3.517 kW.</p>"""
    + ex("142 TR", "499 kW")
    + """<p>6.6  Where a value gives two or more dimensions of the same object, the numbers shall be joined by an
unspaced lowercase "x", followed by a single space and the unit "mm".</p>"""
    + ex("4200 x 2200 x 2400mm", "4200x2200x2400 mm"),
    # 5 — designations and wording
    """<h1>7  Designation Format</h1>
<p>7.1  A nominal connection size shall be written with the keyword "DN" first, a single space, then the numeral.</p>"""
    + ex("150 DN", "DN 150")
    + """<p>7.2  A pressure rating shall be written with the keyword "PN" first, a single space, then the numeral.</p>"""
    + ex("PN16", "PN 16")
    + """<p>7.3  A refrigerant designation shall be written with an upper-case "R", a hyphen, and the number with its
letter suffix unchanged.</p>"""
    + ex("R134a", "R-134a")
    + """<p>7.4  A motor efficiency class shall be written in upper case, unspaced and unhyphenated.</p>"""
    + ex("IE-3", "IE3")
    + """<h1>8  Wording Format</h1>
<p>8.1  A material name recorded as "noun, qualifier" shall be reordered to "qualifier noun".</p>"""
    + ex("Iron, cast", "Cast iron")
    + """<p>8.2  Drive arrangement shall be recorded only as a code of Table 2. A value not listed in Table 2 shall be
reproduced unchanged and flagged for review.</p>
<p><b>Table 2 — Drive arrangement codes</b></p>"""
    + table(["Drive arrangement", "Code"],
            [["Close-coupled", "DRV-CC"], ["Direct drive, flexible coupling", "DRV-DF"],
             ["Belt drive", "DRV-B"]]),
    # 6 — identity matrix (S8)
    """<h1>9  Item-Identity Key Attributes</h1>
<p>The following table lists the key attributes of each equipment class. It governs identity only and contains no
presentation rule.</p>"""
    + table(["Equipment class", "Key attributes", "Display attributes"],
            [["Centrifugal pump", "Design flow; pump head; motor power", "Casing material, drive arrangement"],
             ["Water chiller", "Cooling capacity; refrigerant; chilled water flow", "Dimensions"],
             ["Cooling tower", "Water flow; approach; fan power", "Basin material"],
             ["Plate heat exchanger", "Primary flow; pressure drop; plate size", "Plate material"],
             ["Expansion tank", "Tank dimensions; precharge", "Shell material, flange rating"]])
    + """<h1>10  Attributes Excluded from Reformatting</h1>
<p>• Purchaser tag numbers and serial numbers.</p><p>• Free-text remarks.</p>""",
    # 7 — worked examples + edge cases (S2)
    """<h1>11  Worked Examples</h1>
<h2>11.1  Two Expansion Tanks</h2>
<p>Raw record 1: Tank dimensions 600 x 600 x 900mm; Precharge 1.5 bar; Shell Steel, carbon.</p>
<p>Normalized record 1: 600x600x900 mm; 150 kPa; Carbon steel.</p>
<p>The records remain distinct items where their key attributes differ.</p>
<h1>12  Implementation Guidance and Edge Cases</h1>
<p>12.1  The approach of a cooling tower, and any other temperature difference, is not a temperature: it shall be
expressed in kelvin as a whole number followed by a single space and "K", and is not subject to 6.3.</p>"""
    + ex("5 °C", "5 K")
    + """<p>12.2  Sound pressure level shall be presented per Clause 4.2 of ZZX-STD 000.</p>
<p>12.3  Where a numeric value carries a unit this document does not list, pass it through unchanged and flag it for
manual review.</p>""",
    # 8 — appendix A (S6)
    """<h1>Appendix A  Presentation Forms</h1>
<p>(Normative.) Consolidates the presentation forms defined in Clauses 6 and 7.</p>"""
    + table(["Category", "Normalized form"],
            [["Water flow", "&lt;n.n&gt; m3/h (decimals per Table 1)"], ["Pressure, head, pressure drop", "&lt;n&gt; kPa"],
             ["Temperature", "per OGX-STD 410, 6.2"], ["Electrical power", "&lt;n.n&gt; kW"],
             ["Cooling capacity", "&lt;n&gt; kW"], ["Temperature difference", "&lt;n&gt; K"],
             ["Dimensions", "&lt;n&gt;x&lt;n&gt;[x&lt;n&gt;] mm"], ["Connection size", "DN &lt;n&gt;"],
             ["Pressure rating", "PN &lt;n&gt;"], ["Refrigerant", "R-&lt;number&gt;&lt;suffix&gt;"],
             ["Motor efficiency class", "IE&lt;n&gt;"]]),
]


# (Item, Attribute, Value, Expected, needs_review)
ROWS = [
    ("Chilled Water Pump CHWP-01", "Design flow, m3/h", "45 m3/h", "45.0 m3/h", False),
    ("Chilled Water Pump CHWP-01", "Design flow, m3/h", "45.0m3/h", "45.0 m3/h", False),
    ("Chilled Water Pump CHWP-01", "Pump head", "2.5 bar", "250 kPa", False),
    ("Chilled Water Pump CHWP-01", "Pump head", "250kPa", "250 kPa", False),
    ("Chilled Water Pump CHWP-01", "Motor power", "11 kW", "11.0 kW", False),
    ("Chilled Water Pump CHWP-01", "Motor power", "11kW", "11.0 kW", False),
    ("Chilled Water Pump CHWP-01", "Motor efficiency", "IE-3", "IE3", False),
    ("Chilled Water Pump CHWP-01", "Motor efficiency", "ie 3", "IE3", False),
    ("Chilled Water Pump CHWP-01", "Suction connection", "150 DN", "DN 150", False),
    ("Chilled Water Pump CHWP-01", "Suction connection", "DN150", "DN 150", False),
    ("Chilled Water Pump CHWP-01", "Casing material", "Iron, cast", "Cast iron", False),
    ("Chilled Water Pump CHWP-01", "Drive arrangement", "Close-coupled", "DRV-CC", False),
    ("Chilled Water Pump CHWP-01", "Fluid temperature", "7 °C", "7°C", False),
    ("Chilled Water Pump CHWP-01", "Fluid temperature", "44.6 F", "7°C", False),
    ("Air-Cooled Chiller CH-01", "Chilled water flow, m3/h", "86 m3/h", "86 m3/h", False),
    ("Air-Cooled Chiller CH-01", "Chilled water flow, m3/h", "86.0 m3/h", "86 m3/h", False),
    ("Air-Cooled Chiller CH-01", "Cooling capacity", "500 kW", "500 kW", False),
    ("Air-Cooled Chiller CH-01", "Cooling capacity", "142 TR", "499 kW", False),
    ("Air-Cooled Chiller CH-01", "Compressor input power", "118 kW", "118.0 kW", False),
    ("Air-Cooled Chiller CH-01", "Refrigerant", "R134a", "R-134a", False),
    ("Air-Cooled Chiller CH-01", "Refrigerant", "r 1234ze", "R-1234ze", False),
    ("Air-Cooled Chiller CH-01", "Leaving water temperature", "6 °C", "6°C", False),
    ("Air-Cooled Chiller CH-01", "Leaving water temperature", "42.8 F", "6°C", False),
    ("Air-Cooled Chiller CH-01", "Dimensions (LxWxH)", "4200 x 2200 x 2400mm", "4200x2200x2400 mm", False),
    ("Cooling Tower CT-01", "Water flow, m3/h", "120.0 m3/h", "120 m3/h", False),
    ("Cooling Tower CT-01", "Approach", "5 °C", "5 K", False),
    ("Cooling Tower CT-01", "Fan power", "7.5kW", "7.5 kW", False),
    ("Cooling Tower CT-01", "Sound pressure level", "78dB(A)", "78dB(A)", True),
    ("Cooling Tower CT-01", "Basin material", "Steel, stainless", "Stainless steel", False),
    ("Plate Heat Exchanger PHE-01", "Primary flow, m3/h", "60 m3/h", "60.0 m3/h", False),
    ("Plate Heat Exchanger PHE-01", "Pressure drop", "0.45 bar", "45 kPa", False),
    ("Plate Heat Exchanger PHE-01", "Plate material", "Steel, stainless", "Stainless steel", False),
    ("Plate Heat Exchanger PHE-01", "Plate size", "500 x 1200mm", "500x1200 mm", False),
    ("Plate Heat Exchanger PHE-01", "Pressure rating", "16 PN", "PN 16", False),
    ("Expansion Tank ET-01", "Precharge", "1.5 bar", "150 kPa", False),
    ("Expansion Tank ET-01", "Tank dimensions", "600 x 600 x 900mm", "600x600x900 mm", False),
    ("Expansion Tank ET-01", "Flange rating", "PN16", "PN 16", False),
    ("Expansion Tank ET-01", "Shell material", "Steel, carbon", "Carbon steel", False),
]


def main():
    doc = fitz.open()
    total = len(PAGES)
    for n, body in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        html = HEAD.format(n=n, total=total) + body
        page.insert_htmlbox(fitz.Rect(50, 40, 545, 800), html, css=CSS)
    doc.save(PDF)
    print("pdf:", PDF)

    frame = pd.DataFrame([r[:3] for r in ROWS], columns=["Item", "Attribute", "Value"])
    frame.to_excel(XLSX, index=False)
    EXPECTED.write_text(json.dumps([{"expected": r[3], "review": r[4]} for r in ROWS],
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("xlsx:", XLSX, len(ROWS), "rows")


if __name__ == "__main__":
    main()
