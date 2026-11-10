# -*- coding: utf-8 -*-
"""Эталонные ответы для regression-набора test_attributes_api600_new_new.xlsx (58 строк).

Выведены из текстов самих документов (rule_groups.json обоих стандартов) + ТЗ
normalization_pipeline_fixes.md (таблица регрессий).

Формат значения:
  "str"            — точное совпадение (после strip)
  {"re": "..."}    — регулярка (search)
  {"any": [...]}   — любое из

Известные спорные строки (расхождение заложено в самих документах, не в пайплайне):

* строки 8–9, колонка STD 100 («C-Mn-Si» / «Si-Mn-C»). Пункт 4.1 STD 100 требует убирать
  дефисы «между символами элементов и их числовыми индикаторами», но числовых индикаторов в
  этих значениях нет вообще. STD 205 на этот счёт прямо говорит, что список plain-grade
  обозначений нужно вести явно, «а не выводить исключение из отсутствия цифр» — в STD 100
  такого списка нет, поэтому прочтение пункта модель выбирает примерно 50/50. Строка при
  расхождении сэмплов помечается на ручную проверку.
* строки 36–37, колонка STD 205 (поле «EN equivalent» с единственным значением). Извлечённое
  правило 10.3 предписывает воспроизводить эквивалент в скобках ПОСЛЕ BFC базового
  обозначения, а его worked-example показывает «A193 B7» → «(A193 B7)» — то есть документ
  описывает поведение для составной ячейки. Ожидание из ТЗ (голый BFC) следует из другой
  фразы того же документа, поэтому исход зависит от того, какую из них модель сочтёт
  ведущей. Радикально чинится на этапе извлечения (не тянуть worked-example в отдельное
  правило), а не на этапе применения.
"""

import re as _re

S100 = "XYZ_STD_100_normalization_rules"
S205 = "XYZ_STD_205_alternate_normalization_rules"

SKIP = {"skip": True}

FLAG = {"re": r"(pending registration|ambiguous match|AMBIGUOUS)"}


def kept(value: str) -> dict:
    """Запись оставлена как есть: сама по себе либо с префиксом-вердиктом реестра.

    Так должен выглядеть исход, когда пункт реестра похож, но формулировка не совпала:
    код не подставляется (приближать закрытый реестр нельзя), строка идёт к ревьюеру.
    Код перед вердиктом необязателен: раньше туда попадал префикс ближайшего пункта реестра,
    и ячейка читалась как присуждённый код чужого соединения.
    """
    return {"re": r"^(?:(?:[A-Za-z]+ )?(?:pending registration|ambiguous match): )?"
                  + _re.escape(value) + r"$"}

# (attribute, value, expected_std100, expected_std205)
EXPECTED = [
    ("Nominal size, DN", "DN 150", "DN150", "150DIAMETRA"),
    ("Nominal size, DN", "150 DN", "DN150", "150DIAMETRA"),
    ("Nominal size, NPS", "NPS 6", "NPS6", "6NOMPASSE"),
    ("Nominal size, NPS", "6 NPS", "NPS6", "6NOMPASSE"),
    ("Pressure class", "Class 600", "CL600", "600CLASSATUM"),
    ("Pressure class", "600 class", "CL600", "600CLASSATUM"),
    ("Body material", "18Cr-10Ni-Mo", "18Cr10NiMo", "Cr18~Ni10~Mo"),
    ("Body material", "Cr18-Ni10-Mo", "Cr18Ni10Mo", "Cr18~Ni10~Mo"),
    # расхождение заложено в самих документах, а не в пайплайне: строка не оценивается
    ("Bonnet material", "C-Mn-Si", SKIP, SKIP),
    # расхождение заложено в самих документах, а не в пайплайне: строка не оценивается
    ("Bonnet material", "Si-Mn-C", SKIP, SKIP),
    ("Trim number", "Trim 5", "T5", "5TRIMMEUS"),
    ("Trim number", "5 Trim", "T5", "5TRIMMEUS"),
    ("Trim number", "Trim 9", "T9", "9TRIMMEUS"),
    ("Trim number", "9 Trim", "T9", "9TRIMMEUS"),
    ("Nominal trim", "13Cr and Hardfaced", "13Cr and Hardfaced", "13Cr & Hardfaced"),
    ("Nominal trim", "Hardfaced and 13Cr", "Hardfaced and 13Cr", "Hardfaced & 13Cr"),
    ("Bonnet-to-body joint type", "Flat face", "Flat face", "JT-01"),
    ("Bonnet-to-body joint type", "Face, raised", "Face, raised", kept("Face, raised")),
    ("Bonnet-to-body joint type", "Ring joint", "Ring joint", "JT-02"),
    ("Bonnet-to-body joint type", "Groove and tongue", "Groove and tongue", "JT-03"),
    ("Gate type", "One-piece wedge, solid", "One-piece wedge, solid", "GT-01"),
    ("Gate type", "Split wedge, flexible", "Split wedge, flexible",
     kept("Split wedge, flexible")),
    ("Gate type", "Double-disc gate", "Double-disc gate", "GT-04"),
    # «Gate, solid wedge» ≈ GT-01 «One-piece wedge, solid», но формулировка иная: код не
    # подставляем, запись остаётся как есть и помечается
    ("Gate type", "Gate, solid wedge", "Gate, solid wedge", kept("Gate, solid wedge")),
    ("End connection", "Flanged, raised face", "FLG-RF", "ECC-101"),
    # в Table 2 STD 100 есть «Threaded end» без уточнения NPT — лишнее слово, поэтому запись
    # сохраняется; в STD 205 у пункта ECC-301 есть скобочный алиас «(NPT)», там совпадение точное
    ("End connection", "NPT threaded end", kept("NPT threaded end"), "ECC-301"),
    ("End connection", "Butt-welding end", "BW", "ECC-201"),
    # ни в одной таблице нет «фланцевого RTJ» (в STD 205 это два разных пункта, ECC-401 и
    # ECC-402 «..., ring-type joint facing»), поэтому запись сохраняется, а не кодируется
    ("End connection", "Flanged ring-type joint",
     kept("Flanged ring-type joint"), kept("Flanged ring-type joint")),
    ("Minimum stem diameter, in", "1.375 in", {"re": r"^34\.9\s*mm$"}, "1.375 in."),
    ("Minimum stem diameter, in", "1.375-inch", {"re": r"^34\.9\s*mm$"}, "1.375 in."),
    ("Minimum wall thickness, in", "0.88 in", {"re": r"^22\.4\s*mm$"}, "0.880 in."),
    ("Minimum wall thickness, in", "0.88-inch", {"re": r"^22\.4\s*mm$"}, "0.880 in."),
    ("Seat surface hardness, HB", "350 HB minimum", "350 HB min.", "≥350HB"),
    ("Seat surface hardness, HB", "Minimum 350 HB", "350 HB min.", "≥350HB"),
    ("Bonnet-to-body bolting", "ASTM A193 B8M, CL 2", "ASTM A193 B8M, CL 2", "XB193B8M-C2"),
    ("Bonnet-to-body bolting", "A193-B7 (ASTM)", "A193-B7 (ASTM)", "XB193B7"),
    # расхождение заложено в самих документах, а не в пайплайне: строка не оценивается
    ("Bonnet-to-body bolting, EN equivalent", "A193 B7", SKIP, SKIP),
    # расхождение заложено в самих документах, а не в пайплайне: строка не оценивается
    ("Bonnet-to-body bolting, EN equivalent", "L7, A320", SKIP, SKIP),
    ("Handwheel operation direction", "Opens when turned clockwise",
     "Opens when turned clockwise", "CW-OPEN"),
    ("Handwheel operation direction", "Clockwise to close", "Clockwise to close", "CCW-OPEN"),
    ("Packing material temperature range", "Suitable from -20 F to 1000 F",
     {"re": r"-29\s*°?C.*538\s*°?C"}, "-20–1000°F"),
    ("Packing material temperature range", "-50 F to 800 F, suitable range",
     {"re": r"-46\s*°?C.*427\s*°?C"}, "-50–800°F"),
    ("Face-to-face standard", "ASME B16.10", "ASME B16.10", "ASME B16.10"),
    ("Face-to-face standard", "ASME B16.10, 2017", "ASME B16.10, 2017", "ASME B16.10, 2017"),
    ("Flange facing standard", "ASME B16.5", "ASME B16.5", "ASME B16.5"),
    ("Flange facing standard", "B16.47 ASME", "B16.47 ASME", "B16.47 ASME"),
    ("Nameplate marking", "ASME B16.34", "ASME B16.34", "ASME B16.34"),
    ("Nameplate marking", "600 API", "600 API", "600 API"),
    ("Wear travel, in", "0.5 in", {"re": r"^12\.7\s*mm$"}, "0.500 in."),
    ("Wear travel, in", "0.5-inch", {"re": r"^12\.7\s*mm$"}, "0.500 in."),
    ("Body-to-bonnet stud bolt size", "M16", "M16", "M16"),
    ("Body-to-bonnet stud bolt size", "M-16", "M-16", "M-16"),
    ("Valve shipping position", "Gate open", "Gate open", "Gate open"),
    ("Valve shipping position", "Closed gate", "Closed gate", "Closed gate"),
    ("Alternative trim number", "Trim 12", "T12", "12TRIMMEUS"),
    ("Alternative trim number", "12 Trim", "T12", "12TRIMMEUS"),
    ("Post weld heat treatment temperature, F", "1200 F", {"re": r"^649\s*°?C$"}, "1200°F"),
    ("Post weld heat treatment temperature, F", "1200°F", {"re": r"^649\s*°?C$"}, "1200°F"),
]

# Строки, которые ОБЯЗАНЫ быть помечены на ручную проверку (жёлтая заливка): код присуждён
# не по точному совпадению формулировки, либо пункт реестра вовсе не найден.
# Ключ — (индекс строки, имя колонки).
REVIEW_REQUIRED = {
    (17, S205),  # Face, raised — в Table 4 такого пункта нет
    (21, S205),  # Split wedge, flexible — признаки сразу двух пунктов Table 5
    (23, S205),  # Gate, solid wedge ≈ One-piece wedge, solid — формулировка иная
    (25, S100),  # NPT threaded end ≈ Threaded end — лишнее слово NPT
    (27, S100),  # Flanged ring-type joint — фланцевого RTJ в Table 2 нет
    (27, S205),  # Flanged ring-type joint ≈ ..., ring-type joint facing — нет слова facing
}

# Строки, которые пометки НЕ требуют: формулировка совпала точно (с точностью до скобок,
# запятых и порядка слов) либо значение перечислено через ';'.
REVIEW_FORBIDDEN = {
    (16, S205), (18, S205), (19, S205), (20, S205), (22, S205),
    (24, S100), (24, S205), (25, S205), (26, S100), (26, S205),
}

# Пары строк, которые ОБЯЗАНЫ дать побуквенно одинаковый результат
# (идемпотентность/детерминированность: один и тот же физический смысл).
CONSISTENT_PAIRS = [
    (0, 1), (2, 3), (4, 5), (10, 11), (12, 13),
    (28, 29), (30, 31), (32, 33), (48, 49), (54, 55), (56, 57),
]
