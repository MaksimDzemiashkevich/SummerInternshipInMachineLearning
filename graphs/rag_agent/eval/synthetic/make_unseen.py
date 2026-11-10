# -*- coding: utf-8 -*-
"""Датащиты из значений, которых НЕТ в документах (ни в примерах, ни в таблицах).

Исключение — строки с пометкой REG: запись закрытого реестра обязана совпадать с таблицей
дословно (иначе код не присваивается по правилам самих документов), их проверка — отдельная.
Сборщик падает, если значение без пометки встречается в тексте своего документа.
"""
import json
import re
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория
SP = Path(__file__).parent / "expected"
PENDING = {"re": r"^pending registration: "}

HVAC = [  # (Item, Attribute, Value, Expected)
    ("Air Handling Unit AHU-07 4250m3h", "Airflow rate, m3/h", "4250m3/h", "4250.0 m3/h"),
    ("Air Handling Unit AHU-07 4250m3h", "Airflow rate, m3/h", "4250.00 m3/h", "4250.0 m3/h"),
    ("Air Handling Unit AHU-07 4250m3h", "External static pressure, Pa", "475pa", "475 Pa"),
    ("Air Handling Unit AHU-07 4250m3h", "External static pressure, Pa", "475 PA", "475 Pa"),
    ("Air Handling Unit AHU-07 4250m3h", "Fan motor power, kW", "18.5kW", "18.5 kW"),
    ("Air Handling Unit AHU-07 4250m3h", "Motor rating", "22 kW", "22.0 kW"),
    ("Air Handling Unit AHU-07 4250m3h", "Filter class", "G-4", "G4"),
    ("Air Handling Unit AHU-07 4250m3h", "Filter class", "M 5", "M5"),
    ("Air Handling Unit AHU-07 4250m3h", "Casing material", "Steel, painted", "Painted steel"),
    ("Air Handling Unit AHU-07 4250m3h", "Casing material", "Aluminium, anodized", "Anodized aluminium"),
    ("Extract Fan EF-03 2200m3h", "Rotational speed", "960 Rpm", "960 rpm"),
    ("Extract Fan EF-03 2200m3h", "Speed, rpm", "1475RPM", "1475 rpm"),
    ("Extract Fan EF-03 2200m3h", "Static pressure, Pa", "820PA", "820 Pa"),
    ("Extract Fan EF-03 2200m3h", "Impeller material", "Cu", "Copper"),
    ("Extract Fan EF-03 2200m3h", "Impeller material", "Ti", "Titanium"),
    ("Extract Fan EF-03 2200m3h", "Airflow rate, m3/h", "2200 m3/h", "2200.0 m3/h"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Nominal size", "1000 x 500mm", "1000x500 mm"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Duct connection size", "400 x 250 mm", "400x250 mm"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Fire rating, min", "90MIN", "90 min"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Fire rating, min", "90 Min", "90 min"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Actuation type", "Actuator, pneumatic", "Pneumatic actuator"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Actuation type", "Actuator, spring-return", "Spring-return actuator"),
    ("Fire Smoke Damper FSD-11 1000x500 90min", "Frame material", "Steel, zinc-coated", "Zinc-coated steel"),
    ("Duct Heater DH-05 4.5kW", "Rated power, kW", "4.5kW", "4.5 kW"),
    ("Duct Heater DH-05 4.5kW", "Rated power, kW", "12 kW", "12.0 kW"),
    ("Duct Heater DH-05 4.5kW", "Supply voltage", "230V", "230 V"),
    ("Duct Heater DH-05 4.5kW", "Sheath material", "Alloy, nickel", "Nickel alloy"),
    ("Panel Filter FL-09 F9 592x287", "Face size", "592 x 287mm", "592x287 mm"),
    ("Panel Filter FL-09 F9 592x287", "Rated airflow, m3/h", "1700 m3/h", "1700.0 m3/h"),
    ("Panel Filter FL-09 F9 592x287", "Filter class", "F-9", "F9"),
    ("VAV Box VAV-14 DN250 300-900m3h", "Nominal size", "250 DN", "DN 250"),
    ("VAV Box VAV-14 DN250 300-900m3h", "Nominal size", "160 DN", "DN 160"),
    ("VAV Box VAV-14 DN250 300-900m3h", "Airflow range, m3/h", "300 to 900m3/h", "300-900 m3/h"),
    ("VAV Box VAV-14 DN250 300-900m3h", "Airflow range, m3/h", "150 to 450 m3/h", "150-450 m3/h"),
    ("VAV Box VAV-14 DN250 300-900m3h", "Control type", "Dependent, pressure", "Pressure dependent"),
]

CWX = [
    ("Condenser Water Pump CWP-04", "Design flow, m3/h", "72 m3/h", "72.0 m3/h"),
    ("Condenser Water Pump CWP-04", "Design flow, m3/h", "18.26 m3/h", "18.3 m3/h"),
    ("Condenser Water Pump CWP-04", "Pump head", "3.2 bar", "320 kPa"),
    ("Condenser Water Pump CWP-04", "Pump head", "185kPa", "185 kPa"),
    ("Condenser Water Pump CWP-04", "Motor power", "30 kW", "30.0 kW"),
    ("Condenser Water Pump CWP-04", "Motor power", "5.5kW", "5.5 kW"),
    ("Condenser Water Pump CWP-04", "Motor efficiency", "ie-4", "IE4"),
    ("Condenser Water Pump CWP-04", "Motor efficiency", "IE 2", "IE2"),
    ("Condenser Water Pump CWP-04", "Discharge connection", "100 DN", "DN 100"),
    ("Condenser Water Pump CWP-04", "Discharge connection", "DN125", "DN 125"),
    ("Condenser Water Pump CWP-04", "Impeller material", "Bronze, aluminium", "Aluminium bronze"),
    ("Condenser Water Pump CWP-04", "Casing material", "Iron, ductile", "Ductile iron"),
    ("Condenser Water Pump CWP-04", "Drive arrangement", "Direct drive, rigid coupling", PENDING),
    ("Condenser Water Pump CWP-04", "Fluid temperature", "12 °C", "12°C"),
    ("Condenser Water Pump CWP-04", "Fluid temperature", "53.6 F", "12°C"),
    ("Water-Cooled Chiller CH-03", "Chilled water flow, m3/h", "143.6 m3/h", "144 m3/h"),
    ("Water-Cooled Chiller CH-03", "Cooling capacity", "250 TR", "879 kW"),
    ("Water-Cooled Chiller CH-03", "Cooling capacity", "1200kW", "1200 kW"),
    ("Water-Cooled Chiller CH-03", "Compressor input power", "96kW", "96.0 kW"),
    ("Water-Cooled Chiller CH-03", "Refrigerant", "R410A", "R-410A"),
    ("Water-Cooled Chiller CH-03", "Refrigerant", "r-513a", "R-513a"),
    ("Water-Cooled Chiller CH-03", "Leaving water temperature", "8 °C", "8°C"),
    ("Water-Cooled Chiller CH-03", "Leaving water temperature", "46.4 F", "8°C"),
    ("Water-Cooled Chiller CH-03", "Dimensions (LxWxH)", "3100 x 1800 x 2050mm", "3100x1800x2050 mm"),
    ("Cooling Tower CT-02", "Water flow, m3/h", "310.4 m3/h", "310 m3/h"),
    ("Cooling Tower CT-02", "Approach", "4 °C", "4 K"),
    ("Cooling Tower CT-02", "Cooling range", "6 °C", "6 K"),
    ("Cooling Tower CT-02", "Fan power", "15kW", "15.0 kW"),
    ("Cooling Tower CT-02", "Basin material", "Plastic, fibre-reinforced", "Fibre-reinforced plastic"),
    ("Cooling Tower CT-02", "Sound pressure level", "72 dB(A)", "72 dB(A)"),
    ("Plate Heat Exchanger PHE-06", "Secondary flow, m3/h", "41.37 m3/h", "41.4 m3/h"),
    ("Plate Heat Exchanger PHE-06", "Pressure drop", "0.3 bar", "30 kPa"),
    ("Plate Heat Exchanger PHE-06", "Pressure drop", "38kPa", "38 kPa"),
    ("Plate Heat Exchanger PHE-06", "Plate size", "320 x 880mm", "320x880 mm"),
    ("Plate Heat Exchanger PHE-06", "Pressure rating", "PN25", "PN 25"),
    ("Plate Heat Exchanger PHE-06", "Pressure rating", "10 PN", "PN 10"),
    ("Plate Heat Exchanger PHE-06", "Gasket material", "Rubber, nitrile", "Nitrile rubber"),
    ("Expansion Tank ET-05", "Precharge", "0.8 bar", "80 kPa"),
    ("Expansion Tank ET-05", "Tank dimensions", "480 x 480 x 710mm", "480x480x710 mm"),
    ("Expansion Tank ET-05", "Flange rating", "PN6", "PN 6"),
    ("Expansion Tank ET-05", "Shell material", "Steel, galvanized", "Galvanized steel"),
]
CWX_REVIEW = {"Sound pressure level", "Drive arrangement"}

PKX = [
    ("Rotary Screw Compressor SC-09", "Free air delivery", "8.4 m3/min", "8.40 m3/min"),
    ("Rotary Screw Compressor SC-09", "Duty", "31.126 m3/min", "31.13 m3/min"),
    ("Rotary Screw Compressor SC-09", "Working pressure", "8.5 barg", "8.5 bar(g)"),
    ("Rotary Screw Compressor SC-09", "Working pressure", "13 bar g", "13.0 bar(g)"),
    ("Rotary Screw Compressor SC-09", "Cooling method", "Cooled, oil", "Oil-cooled"),
    ("Rotary Screw Compressor SC-09", "Mounting arrangement", "Mounted on trailer", "Trailer-mounted"),
    ("Rotary Screw Compressor SC-09", "Compressor type", "Oil-free rotary screw", "CMP-OFS"),  # REG
    ("Reciprocating Compressor RC-04", "Compressor type", "Reciprocating, single stage", "CMP-R1"),  # REG
    ("Reciprocating Compressor RC-04", "Capacity, m3/min", "4 m3/min", "4.00 m3/min"),
    ("Reciprocating Compressor RC-04", "Cooling method", "Cooled, liquid", "Liquid-cooled"),
    ("Reciprocating Compressor RC-04", "Set pressure", "9.26 barg", "9.3 bar(g)"),
    ("Reciprocating Compressor RC-04", "Mounting arrangement", "Mounted on floor", "Floor-mounted"),
    ("Rotary Screw Compressor SC-11", "Compressor type", "Oil-flooded rotary screw", PENDING),
    ("Rotary Lobe Blower BL-06", "Discharge pressure", "0.45 barg", "450 mbar(g)"),
    ("Rotary Lobe Blower BL-06", "Discharge pressure", "1.2 bar(g)", "1200 mbar(g)"),
    ("Rotary Lobe Blower BL-06", "Mounting arrangement", "Mounted on ceiling", "Ceiling-mounted"),
    ("Air Receiver AR-08", "Volume", "500 l", "500 L"),
    ("Air Receiver AR-08", "Volume", "1500 litres", "1500 L"),
    ("Air Receiver AR-08", "Maximum pressure", "16 barg", "16.0 bar(g)"),
    ("Refrigerated Dryer RD-03", "Pressure dew point", "7C", "+7 °C"),
    ("Refrigerated Dryer RD-03", "Pressure dew point", "-20 C", "-20 °C"),
    ("Refrigerated Dryer RD-03", "Pressure dew point", "+4 °C", "+4 °C"),
    ("Refrigerated Dryer RD-03", "Pressure dew point", "10 °C", "+10 °C"),
]
REG = {"Oil-free rotary screw", "Reciprocating, single stage"}

import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "a"
if VARIANT == "b":
    sys.path.insert(0, str(Path(__file__).parent))
    from unseen_b import HVAC, CWX, PKX

SETS = {
    "hvac": (HVAC, "HVX-STD-134_hvac_ventilation_equipment_normalization_rules.pdf", set()),
    "cwx": (CWX, "CWX-STD-221_chilled_water_pumping_normalization_rules.pdf", CWX_REVIEW),
    "pkx": (PKX, "PKX-STD-330_package_compressor_normalization_rules.pdf", {"Compressor type"}),
}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


for key, (rows, pdf, review_attrs) in SETS.items():
    text = norm("".join(p.get_text() for p in fitz.open(ROOT / "graphs/structured_documents/data/input" / pdf)))
    leaked = [r[2] for r in rows if re.search(r"(?<![\w-])" + re.escape(norm(r[2])) + r"(?![\w-])", text) and r[2] not in REG]
    assert not leaked, f"{key}: значения есть в документе: {leaked}"
    pd.DataFrame([r[:3] for r in rows], columns=["Item", "Attribute", "Value"]).to_excel(
        ROOT / f"graphs/rag_agent/data/input/{key}_unseen_{VARIANT}.xlsx", index=False)
    expected = []
    for r in rows:
        exp = r[3]
        review = r[1] in review_attrs and exp == PENDING or (key == "cwx" and r[1] == "Sound pressure level")
        if exp == PENDING:
            # вне реестра верны оба исхода: «pending registration: …» или запись как есть — с пометкой
            exp = [PENDING, r[2]]
        expected.append({"expected": exp, "review": True if review else None})
    (SP / f"{key}_unseen_{VARIANT}.json").write_text(json.dumps(expected, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(key, len(rows), "строк, пересечений с документом нет (кроме записей реестра:",
          sum(r[2] in REG for r in rows), ")")
