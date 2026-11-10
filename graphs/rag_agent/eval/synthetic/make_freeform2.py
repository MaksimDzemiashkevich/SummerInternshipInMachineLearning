# -*- coding: utf-8 -*-
"""Ещё два стандарта в свободной форме, непохожие на прежние и друг на друга.

- СТО-НГК-112 — стандарт организации НА РУССКОМ языке в стиле ГОСТ (арматура и сосуды): «Ду» →
  «DN», МПа → PN, марки стали кириллицей (латинские X/H во входе — ловушка), климатическое
  исполнение, кириллическая «С» в «°С», десятичная запятая, буквенные коды присоединения.
- ICS-SPEC-260 — контрактная спецификация КИП в две колонки: многоуровневые пункты, сноски,
  Приложение D, которое «prevails» над разделом 5 (EPL в Ex-маркировке), Kv → Cv, материал
  записывается по-разному у датчика и у клапана.

Значения таблиц не встречаются в тексте документов (кроме атрибутов-кодов). Запуск:
    venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.make_freeform2
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
body { font-family: sans-serif; font-size: 9pt; }
h1 { font-size: 12.5pt; margin: 6pt 0 3pt 0; }
h2 { font-size: 10.5pt; margin: 5pt 0 2pt 0; }
p { margin: 2pt 0 3pt 0; }
table { border-collapse: collapse; margin: 3pt 0; }
td, th { border: 1px solid #333; padding: 2pt 3pt; font-size: 8.5pt; vertical-align: top; }
td.col { border: none; padding: 0 6pt 0 0; width: 50%; font-size: 9pt; }
sup { font-size: 6pt; }
"""


def table(head, rows):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    return "<table>" + h + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                                   for r in rows) + "</table>"


def two_columns(left: str, right: str) -> str:
    return f"<table><tr><td class='col'>{left}</td><td class='col'>{right}</td></tr></table>"


# ------------------------------------------------------------ СТО-НГК-112 (русский, стиль ГОСТ)
STO_PAGES = [
    """<p style='font-size:8pt;color:#555'>ООО «НефтеГазКомплект» — учебный документ, не для инженерного применения</p>
<h1>СТАНДАРТ ОРГАНИЗАЦИИ СТО-НГК-112-2026</h1>
<p><b>Правила записи характеристик трубопроводной арматуры и сосудов в справочнике НСИ</b></p>
<h2>1 Область применения</h2>
<p>Настоящий стандарт устанавливает форму записи значений характеристик запорной арматуры (задвижки, краны шаровые,
клапаны) и сосудов, работающих под давлением, в справочнике нормативно-справочной информации (НСИ).</p>
<h2>4 Общие требования</h2>
<p>4.1 Десятичный разделитель — запятая. Между числом и обозначением единицы ставится один пробел.</p>
<p>4.2 Обозначение градуса Цельсия записывается как «°C» с латинской буквой C. Знак «+» перед положительным
значением не пишется, отрицательное значение записывается со знаком «-» без пробела; слово «минус» заменяется знаком.</p>
<p>4.3 Значение, которое не удаётся привести к форме настоящего стандарта, вносится в том виде, в каком получено,
и передаётся в группу НСИ на рассмотрение.</p>
<h2>5 Требования к отдельным характеристикам</h2>
<p>5.1 <b>Номинальный диаметр</b> записывают как «DN», пробел и число (ГОСТ 28338). Обозначения «Ду», «Dу»,
«условный проход» и значение в миллиметрах без обозначения считаются тем же номинальным диаметром. Размер в дюймах
переводят по таблице 1.</p>"""
    + "<p><b>Таблица 1 — Соответствие размеров в дюймах номинальному диаметру</b></p>"
    + table(["Размер, дюйм", "Номинальный диаметр"], [["1/2", "DN 15"], ["3/4", "DN 20"], ["1", "DN 25"],
                                                      ["2", "DN 50"], ["3", "DN 80"], ["4", "DN 100"],
                                                      ["6", "DN 150"], ["8", "DN 200"]])
    + """<p>5.2 <b>Номинальное давление</b> арматуры записывают как «PN», пробел и число. Обозначение «Ру» означает то же.
Давление, указанное в МПа, переводят: PN = значение в МПа × 10.</p>
<p>5.3 <b>Расчётное давление</b> сосуда записывают в МПа с двумя знаками после запятой. Значения в бар делят на 10,
в кПа — на 1000.</p>""",
    """<p style='font-size:8pt;color:#555'>СТО-НГК-112-2026 — с. 2</p>
<p>5.4 <b>Марку стали</b> записывают по ГОСТ: прописными кириллическими буквами, без пробелов и точек между
элементами обозначения. Латинские буквы, совпадающие по начертанию с кириллическими (X, H, C, B, T и др.),
заменяются кириллическими. Углеродистую сталь обыкновенного качества и качественную записывают словом «Сталь», пробел
и номер марки: «Сталь 20», «Сталь 10» (допускаются входные «Ст20», «Ст.20», «сталь20»).</p>
<p>5.5 <b>Климатическое исполнение</b> по ГОСТ 15150 записывают прописными буквами и цифрой категории размещения
слитно: У1, УХЛ1, ХЛ1, Т1.</p>
<p>5.6 <b>Температуру рабочей среды</b> записывают в °C целым числом. Значения в °F переводят: (°F − 32) / 1,8.</p>
<p>5.7 <b>Тип присоединения</b> записывают буквенным кодом по таблице 2. Тип, отсутствующий в таблице 2, вносят
как получен по 4.3.</p>"""
    + "<p><b>Таблица 2 — Коды типов присоединения</b></p>"
    + table(["Тип присоединения", "Код"], [["Фланцевое", "Ф"], ["Под приварку", "П"], ["Муфтовое", "М"],
                                           ["Штуцерное", "Ш"], ["Цапковое", "Ц"]])
    + """<p>5.8 <b>Массу</b> записывают в килограммах с одним знаком после запятой. Масса в тоннах умножается на 1000,
в фунтах — на 0,4536.</p>
<p>5.9 <b>Объём</b> сосуда записывают в м³ с одним знаком после запятой; объём в литрах делят на 1000.</p>
<h2>Приложение А (справочное)</h2>
<p>А.1 Примеры: «Ду 100» → «DN 100»; «Ру 63» → «PN 63»; «ст.10» → «Сталь 10»; «у 1» → «У1».</p>""",
]

STO_ROWS = [
    ("Задвижка ЗКС-01", "Номинальный диаметр", "Ду150", "DN 150", False),
    ("Задвижка ЗКС-01", "Номинальный диаметр", "6 дюймов", "DN 150", False),
    ("Кран шаровый КШ-02", "Номинальный диаметр", "условный проход 80 мм", "DN 80", False),
    ("Кран шаровый КШ-02", "Номинальный диаметр", "2 in", "DN 50", False),
    ("Задвижка ЗКС-01", "Номинальное давление", "Ру 16", "PN 16", False),
    ("Кран шаровый КШ-02", "Номинальное давление", "4,0 МПа", "PN 40", False),
    ("Сосуд В-03", "Расчётное давление", "2,5 МПа", "2,50 МПа", False),
    ("Сосуд В-03", "Расчётное давление", "16 бар", "1,60 МПа", False),
    ("Сосуд В-03", "Расчётное давление", "250 кПа", "0,25 МПа", False),
    ("Задвижка ЗКС-01", "Марка стали корпуса", "09г2с", "09Г2С", False),
    ("Кран шаровый КШ-02", "Марка стали корпуса", "12X18H10T", "12Х18Н10Т", False),
    ("Сосуд В-03", "Марка стали корпуса", "ст 20", "Сталь 20", False),
    ("Задвижка ЗКС-01", "Климатическое исполнение", "ухл 1", "УХЛ1", False),
    ("Кран шаровый КШ-02", "Климатическое исполнение", "ХЛ-1", "ХЛ1", False),
    ("Задвижка ЗКС-01", "Температура рабочей среды, °С", "минус 45 °С", "-45 °C", False),
    ("Кран шаровый КШ-02", "Температура рабочей среды, °С", "+120", "120 °C", False),
    ("Сосуд В-03", "Температура рабочей среды, °С", "572 °F", "300 °C", False),
    ("Задвижка ЗКС-01", "Тип присоединения", "фланцевое", "Ф", False),
    ("Кран шаровый КШ-02", "Тип присоединения", "под приварку", "П", False),
    ("Сосуд В-03", "Тип присоединения", "раструбное", {"re": r"^(?:(?:pending registration|ambiguous match): )?[Рр]аструбное$"}, True),
    ("Задвижка ЗКС-01", "Масса", "1,25 т", "1250,0 кг", False),
    ("Сосуд В-03", "Масса", "830 кг", "830,0 кг", False),
    ("Кран шаровый КШ-02", "Масса", "2200 lb", "997,9 кг", False),
    ("Сосуд В-03", "Объём", "4000 л", "4,0 м³", False),
]
STO_CODED = {"Тип присоединения", "Марка стали корпуса"}

# --------------------------------------------- ICS-SPEC-260 (контрактная спецификация, две колонки)
ICS_LEFT_1 = """<h2>5 Data sheet population</h2>
<p><b>5.1 General.</b> 5.1.1 Values are written in SI units unless 5.2–5.6 state otherwise. 5.1.2 Numbers are rounded to
one decimal; a whole result is written without decimals. 5.1.3 One space separates a number from its unit, except
before "%" where the rule of the attribute applies.</p>
<p><b>5.2 Calibrated range.</b> 5.2.1 Written "LRV to URV" followed by the unit once. 5.2.2 Pressure instruments: kPa(g)
for gauge pressure. 5.2.3 Temperature instruments: °C. 5.2.4 Level instruments: mm, whole numbers.</p>
<p><b>5.3 Output signal.</b> Coded per Annex D, Table D.1. Signals not listed in Table D.1 are entered as received and
returned to the vendor<sup>1</sup>.</p>
<p><b>5.4 Hazardous area.</b> The marking is written "Ex", protection, gas group, temperature class, separated by single
spaces; the equipment protection level (EPL) is omitted<sup>2</sup>. "EEx" is written "Ex".</p>"""

ICS_RIGHT_1 = """<p><b>5.5 Accuracy.</b> 5.5.1 Written with the plus-minus sign "±" immediately followed by the number,
a space, "%", and "of span": e.g. a value of half a percent of span is ±0.5 % of span. 5.5.2 "+/-" and "+-" mean ±.
5.5.3 A value with no basis, or given as an absolute quantity (mm, °C), is entered as received and returned to the
vendor<sup>1</sup>.</p>
<p><b>5.6 Process connection.</b> Size as an inch fraction, a space, "in", a space, "NPT", a space and the gender in
brackets: (F) female, (M) male. "FNPT", "NPT-F", "NPTF" and "female" denote (F); "MNPT", "NPT-M", "NPTM" and "male"
denote (M).</p>
<p><b>5.7 Enclosure.</b> IP code: "IP" immediately followed by two digits. NEMA ratings are written as received and
are not converted to IP<sup>3</sup>.</p>
<p><b>5.8 Flow coefficient.</b> Control valves only. The Cv value as a number, without the letters "Cv". A Kv value
is converted: Cv = 1.156 × Kv.</p>
<p><b>5.9 Materials.</b> See Annex D, Table D.2 (instruments) and Table D.3 (valve bodies).</p>
<p style='font-size:7.5pt'>¹ Returned values are kept in the data sheet unchanged until the vendor resubmits.<br>
² Superseded for hazardous-area markings by Annex D, note D.3.<br>
³ The two systems are not equivalent; no conversion table is approved.</p>"""

ICS_PAGES = [
    ("""<p style='font-size:8pt;color:#555'>ICS Engineering — Fictitious test corpus, not for engineering use</p>
<h1>ICS-SPEC-260 Instrument Data Sheet Population — Contractor Specification</h1>
<p>Rev. C. Applies to pressure, temperature and level transmitters and to control valves supplied under this contract.
Where Annex D conflicts with Section 5, Annex D prevails.</p>""", ICS_LEFT_1, ICS_RIGHT_1),
    """<p style='font-size:8pt;color:#555'>ICS-SPEC-260 Rev. C — page 2</p>
<h2>Annex D (normative) — Codes and material designations</h2>
<p><b>Table D.1 — Output signal codes</b></p>"""
    + table(["Signal", "Also received as", "Code"], [
        ["4-20 mA", "4…20 mA; 4 to 20 mA; analog", "AI"],
        ["4-20 mA with HART", "4-20 mA + HART; HART", "AI-H"],
        ["Foundation Fieldbus H1", "FOUNDATION fieldbus; FF", "FF"],
        ["Profibus PA", "PB-PA", "PA"],
        ["Modbus RTU", "Modbus", "MB"]])
    + """<p><b>Table D.2 — Wetted material designations (instruments)</b></p>"""
    + table(["Written as", "Also received as"], [["SS316L", "316L stainless; AISI 316L"], ["C-276", "Hastelloy C; Alloy C-276"],
                                                 ["Monel 400", "Monel; Alloy 400"], ["Tantalum", "Ta"]])
    + """<p><b>Table D.3 — Valve body materials</b> (written with the ASTM specification: spec number, a space, grade)</p>"""
    + table(["Written as", "Also received as"], [["A216 WCB", "carbon steel casting"], ["A351 CF8M", "cast 316 stainless"],
                                                 ["A352 LCC", "low-temperature carbon steel casting"]])
    + """<p><b>D.3</b> For hazardous-area markings the equipment protection level (Ga, Gb, Gc) is retained after the
temperature class, separated by a single space. This note prevails over 5.4.</p>
<p><b>D.4</b> Material designations not listed in Tables D.2 and D.3 are entered as received.</p>""",
]

ICS_ROWS = [
    ("Pressure transmitter PT-301", "Calibrated range", "0-10 bar", "0 to 1000 kPa(g)", False),
    ("Pressure transmitter PT-301", "Calibrated range", "0...25 psig", "0 to 172.4 kPa(g)", False),
    ("Temperature transmitter TT-302", "Calibrated range", "32-392 F", "0 to 200 °C", False),
    ("Temperature transmitter TT-302", "Calibrated range", "-50...150 degC", "-50 to 150 °C", False),
    ("Level transmitter LT-304", "Calibrated range", "0-2.5 m", "0 to 2500 mm", False),
    ("Pressure transmitter PT-301", "Output signal", "4-20mA/HART", "AI-H", False),
    ("Temperature transmitter TT-302", "Output signal", "4..20 mA", "AI", False),
    ("Level transmitter LT-304", "Output signal", "FF H1", "FF", False),
    ("Pressure transmitter PT-301", "Output signal", "WirelessHART", "WirelessHART", True),
    ("Pressure transmitter PT-301", "Accuracy", "0.075% of span", "±0.075 % of span", False),
    ("Temperature transmitter TT-302", "Accuracy", "+/-0.1 % span", "±0.1 % of span", False),
    ("Level transmitter LT-304", "Accuracy", "±3 mm", "±3 mm", True),
    ("Pressure transmitter PT-301", "Accuracy", "0.04%", "0.04%", True),
    ("Pressure transmitter PT-301", "Process connection", "1/2\" FNPT", "1/2 in NPT (F)", False),
    ("Temperature transmitter TT-302", "Process connection", "3/4 NPT-M", "3/4 in NPT (M)", False),
    ("Pressure transmitter PT-301", "Enclosure", "IP 67", "IP67", False),
    ("Level transmitter LT-304", "Enclosure", "NEMA 4X", "NEMA 4X", False),
    ("Pressure transmitter PT-301", "Hazardous area", "Ex-d-IIC-T4-Gb", "Ex d IIC T4 Gb", False),
    ("Temperature transmitter TT-302", "Hazardous area", "EEx ia IIC T6 Ga", "Ex ia IIC T6 Ga", False),
    ("Control valve FV-303", "Flow coefficient", "Kv 100", "115.6", False),
    ("Control valve FV-303", "Flow coefficient", "Cv 240", "240", False),
    ("Control valve FV-303", "Body material", "WCB", "A216 WCB", False),
    ("Pressure transmitter PT-301", "Wetted material", "316L SS", "SS316L", False),
    ("Level transmitter LT-304", "Wetted material", "Hastelloy C276", "C-276", False),
]
ICS_CODED = {"Output signal", "Wetted material", "Body material", "Enclosure"}

DOCS = [
    ("STO-NGK-112_armatura_sosudy_NSI", "sto_freeform", STO_PAGES, STO_ROWS, STO_CODED),
    ("ICS-SPEC-260_instrument_datasheet_population", "ics_freeform", ICS_PAGES, ICS_ROWS, ICS_CODED),
]


def build(stem: str, key: str, pages: list[str], rows: list[tuple], coded: set) -> None:
    pdf = PDF_DIR / f"{stem}.pdf"
    doc = fitz.open()
    for body in pages:
        page = doc.new_page(width=595, height=842)
        # страница в две колонки — (шапка, левая колонка, правая колонка) в отдельных рамках
        boxes = ([(fitz.Rect(40, 36, 555, 806), body)] if isinstance(body, str) else
                 [(fitz.Rect(40, 36, 555, 110), body[0]), (fitz.Rect(40, 112, 292, 806), body[1]),
                  (fitz.Rect(303, 112, 555, 806), body[2])])
        for rect, html in boxes:
            spare, _ = page.insert_htmlbox(rect, html, css=CSS)
            if spare < 0:
                raise SystemExit(f"{stem}: текст не поместился в рамку")
    doc.save(pdf)
    text = re.sub(r"\s+", " ", "".join(p.get_text() for p in fitz.open(pdf))).lower()
    leaked = [r[2] for r in rows if r[1] not in coded and r[2].lower() in text and len(r[2]) > 3]
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
