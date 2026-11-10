# -*- coding: utf-8 -*-
"""BHX-STD 077 — правила БЕЗ примеров «было → стало»: только описание правильного вида.

Документ говорит, в какой единице, с каким числом знаков, в каком порядке и регистре писать
значение, и сводит это в таблицу и приложение с формами «<n.n> mm». Ни одной пары X → Y.
Датащит требует пересчёта единиц (in, hp, ft/min, L), узнавания синонимов (tph, deg, m³),
перестановки слов и приведения регистра — всего того, что пример обычно «показывает».
"""
import json
import re
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория
PDF = ROOT / "graphs/structured_documents/data/input/BHX-STD-077_conveyor_bulk_handling_normalization_rules.pdf"
XLSX = ROOT / "graphs/rag_agent/data/input/bhx_conveyor_test_corpus.xlsx"
EXPECTED = Path(__file__).parent / "expected" / "bhx_expected.json"

HEAD = ("<p style='font-size:8pt;color:#555'>BHX-STD 077 — Fictitious / Internal Test Corpus Only<br>"
        "Conveyors and Bulk Handling Equipment, Edition 1<br>Page {n} of {total}</p>")
CSS = """
body { font-family: sans-serif; font-size: 9.5pt; }
h1 { font-size: 13pt; margin: 6pt 0 3pt 0; }
p { margin: 2pt 0 3pt 0; }
table { border-collapse: collapse; margin: 4pt 0; }
td, th { border: 1px solid #333; padding: 2pt 4pt; font-size: 9pt; }
"""


def table(head, rows):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    return "<table>" + h + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


PAGES = [
    """<h1>BHX-STD 077</h1>
<p><b>Attribute Normalization Rules for Conveyors and Bulk Handling Equipment</b></p>
<p>Issued by the Internal Data Standards Working Group — Fictitious Test Corpus Series. Edition 1 — Not for engineering use.</p>
<h1>1  Scope</h1>
<p>1.1  This document specifies presentation-normalization rules for attribute values recorded on data sheets for
belt conveyors, screw conveyors, bucket elevators, vibrating screens and storage hoppers.</p>
<p>1.2  Normalization changes presentation only. It never alters the technical substance of a recorded value; a
unit conversion required by this document is a change of presentation, not of substance.</p>
<h1>4  General Principles</h1>
<p>4.1  Every rule shall be applied deterministically and idempotently: a value already in the required form is left
unchanged.</p>
<p>4.2  Where a value is missing or marked "N/A", no rule shall be applied.</p>
<p>4.3  Where a numeric value carries a unit this document does not list for that quantity and gives no conversion
for, it shall be reproduced unchanged and flagged for manual review.</p>""",
    """<h1>6  Units of Measurement</h1>
<p>6.1  Belt width shall be expressed in millimetres as a whole number, followed by a single space and the unit
"mm". A width recorded in inches shall be converted using 1 in = 25.4 mm before rounding.</p>
<p>6.2  Screw diameter and pulley diameter shall be expressed in millimetres as a whole number, followed by a single
space and the unit "mm", converting from inches in the same way as 6.1.</p>
<p>6.3  Belt speed and bucket speed shall be expressed in metres per second to two decimal places, followed by a
single space and the unit "m/s". A speed recorded in feet per minute shall be converted using
1 ft/min = 0.00508 m/s.</p>
<p>6.4  Throughput capacity shall be expressed in tonnes per hour as a whole number, followed by a single space and
the unit written as "t/h". The abbreviations "tph" and "t/hr" denote the same unit.</p>
<p>6.5  Motor power shall be expressed in kilowatts to one decimal place, followed by a single space and the unit
"kW" with an upper-case W. A power recorded in horsepower shall be converted using 1 hp = 0.7457 kW.</p>
<p>6.6  Screen aperture shall be expressed in millimetres to one decimal place, followed by a single space and "mm".</p>
<p>6.7  Inclination angle shall be expressed as a whole number of degrees immediately followed by the degree sign,
with no space and no word "degrees" or "deg".</p>
<p>6.8  Hopper volume shall be expressed in cubic metres to one decimal place, followed by a single space and the unit
written as "m3" (no superscript). A volume recorded in litres shall be converted using 1 m3 = 1000 L.</p>""",
    """<p><b>Table 1 — Summary of units and decimal places</b></p>"""
    + table(["Quantity", "Unit", "Decimal places"],
            [["Belt width", "mm", "0"], ["Screw or pulley diameter", "mm", "0"], ["Belt or bucket speed", "m/s", "2"],
             ["Throughput capacity", "t/h", "0"], ["Motor power", "kW", "1"], ["Screen aperture", "mm", "1"],
             ["Inclination angle", "°", "0"], ["Hopper volume", "m3", "1"]])
    + """<h1>7  Designation Format</h1>
<p>7.1  A belt carcass rating shall be written in upper case as the fabric code, the tensile rating number, a forward
slash and the ply count, with no spaces between any of these elements.</p>
<p>7.2  A screen deck count shall be written as the numeral, a single space, and the word "deck" when the count is one
or "decks" when it is more than one. A count written as a word shall be written as its numeral.</p>
<h1>8  Wording Format</h1>
<p>8.1  A drive position shall be written as the position word followed by the word "drive", with only the first letter
of the value in upper case and no comma.</p>
<p>8.2  A material name recorded as a noun followed by one or more qualifiers after a comma shall be written as the
qualifiers followed by the noun, with no comma, the first letter of the value in upper case and the remaining words
as recorded, except that an abbreviation written in upper case keeps its case.</p>
<h1>9  Attributes Excluded from Reformatting</h1>
<p>• Serial numbers, purchaser tag numbers and free-text remarks.</p>""",
    """<h1>Appendix A  Presentation Forms</h1>
<p>(Normative.) Consolidates the forms required by Clauses 6 to 8.</p>"""
    + table(["Quantity", "Required form"],
            [["Belt width", "&lt;n&gt; mm"], ["Screw or pulley diameter", "&lt;n&gt; mm"],
             ["Belt or bucket speed", "&lt;n.nn&gt; m/s"], ["Throughput capacity", "&lt;n&gt; t/h"],
             ["Motor power", "&lt;n.n&gt; kW"], ["Screen aperture", "&lt;n.n&gt; mm"],
             ["Inclination angle", "&lt;n&gt;°"], ["Hopper volume", "&lt;n.n&gt; m3"],
             ["Belt carcass rating", "&lt;CODE&gt;&lt;rating&gt;/&lt;plies&gt;"],
             ["Screen deck count", "&lt;n&gt; deck(s)"], ["Drive position", "&lt;Position&gt; drive"]]),
]

ROWS = [  # (Item, Attribute, Value, Expected)
    ("Belt Conveyor BC-04", "Belt width", "1200mm", "1200 mm"),
    ("Belt Conveyor BC-04", "Belt width", "48 in", "1219 mm"),
    ("Belt Conveyor BC-04", "Belt speed", "2.5 m/s", "2.50 m/s"),
    ("Belt Conveyor BC-04", "Belt speed", "500 ft/min", "2.54 m/s"),
    ("Belt Conveyor BC-04", "Capacity", "850 t/h", "850 t/h"),
    ("Belt Conveyor BC-04", "Capacity", "850.0 tph", "850 t/h"),
    ("Belt Conveyor BC-04", "Motor power", "75kW", "75.0 kW"),
    ("Belt Conveyor BC-04", "Motor power", "100 hp", "74.6 kW"),
    ("Belt Conveyor BC-04", "Inclination", "12 deg", "12°"),
    ("Belt Conveyor BC-04", "Inclination", "15 °", "15°"),
    ("Belt Conveyor BC-04", "Belt rating", "ep 630 / 4", "EP630/4"),
    ("Belt Conveyor BC-04", "Belt rating", "EP 800/5", "EP800/5"),
    ("Belt Conveyor BC-04", "Drive position", "Drive, head", "Head drive"),
    ("Belt Conveyor BC-04", "Drive position", "TAIL DRIVE", "Tail drive"),
    ("Belt Conveyor BC-04", "Pulley diameter", "630.4 mm", "630 mm"),
    ("Belt Conveyor BC-04", "Idler material", "Steel, abrasion-resistant", ["Abrasion-resistant Steel", "Abrasion-resistant steel"]),
    ("Screw Conveyor SC-02", "Screw diameter", "300 mm", "300 mm"),
    ("Screw Conveyor SC-02", "Screw diameter", "12 in", "305 mm"),
    ("Screw Conveyor SC-02", "Capacity", "40 t/hr", "40 t/h"),
    ("Screw Conveyor SC-02", "Motor power", "15 kW", "15.0 kW"),
    ("Screw Conveyor SC-02", "Motor power", "20 HP", "14.9 kW"),
    ("Screw Conveyor SC-02", "Trough material", "Steel, stainless", ["Stainless Steel", "Stainless steel"]),
    ("Bucket Elevator BE-03", "Capacity", "120.4 t/h", "120 t/h"),
    ("Bucket Elevator BE-03", "Bucket speed", "1.6 m/s", "1.60 m/s"),
    ("Bucket Elevator BE-03", "Motor power", "30kw", "30.0 kW"),
    ("Bucket Elevator BE-03", "Drive position", "Drive, top", "Top drive"),
    ("Vibrating Screen VS-05", "Screen aperture", "5 mm", "5.0 mm"),
    ("Vibrating Screen VS-05", "Screen aperture", "0.5 in", "12.7 mm"),
    ("Vibrating Screen VS-05", "Deck count", "2", "2 decks"),
    ("Vibrating Screen VS-05", "Deck count", "Double deck", "2 decks"),
    ("Vibrating Screen VS-05", "Deck count", "1 decks", "1 deck"),
    ("Vibrating Screen VS-05", "Inclination", "20 degrees", "20°"),
    ("Vibrating Screen VS-05", "Screen media material", "Polyurethane, UV-resistant", ["UV-resistant Polyurethane", "UV-resistant polyurethane"]),
    # материалы: по 8.2 остальные слова «как записаны» (Steel), принимаются оба прочтения
    ("Storage Hopper HP-02", "Volume", "25 m³", "25.0 m3"),
    ("Storage Hopper HP-02", "Volume", "25000 L", "25.0 m3"),
    ("Storage Hopper HP-02", "Volume", "12.34 m3", "12.3 m3"),
    ("Storage Hopper HP-02", "Shell material", "Steel, mild", ["Mild Steel", "Mild steel"]),
    ("Storage Hopper HP-02", "Liner material", "Polyethylene, UHMW", ["UHMW Polyethylene", "UHMW polyethylene"]),
    ("Storage Hopper HP-02", "Discharge temperature", "60 °C", {"review": "60 °C"}),
]


def main():
    doc = fitz.open()
    for n, body in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(fitz.Rect(50, 40, 545, 800), HEAD.format(n=n, total=len(PAGES)) + body, css=CSS)
    doc.save(PDF)
    text = re.sub(r"\s+", " ", "".join(p.get_text() for p in fitz.open(PDF))).lower()
    arrows = re.findall(r"→|->|example", text)
    leaked = [r[2] for r in ROWS if re.search(r"(?<![\w-])" + re.escape(r[2].lower()) + r"(?![\w-])", text)]
    print("стрелок/EXAMPLE в документе:", len(arrows), "| значений датащита в тексте документа:", leaked)
    pd.DataFrame([r[:3] for r in ROWS], columns=["Item", "Attribute", "Value"]).to_excel(XLSX, index=False)
    out = []
    for r in ROWS:
        if isinstance(r[3], dict):  # единица, которой документ не знает: как записано + пометка
            out.append({"expected": r[3]["review"], "review": True})
        else:
            out.append({"expected": r[3], "review": None})
    EXPECTED.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(PDF.name, XLSX.name, len(ROWS), "строк")


if __name__ == "__main__":
    main()
