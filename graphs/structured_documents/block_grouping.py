# -*- coding: utf-8 -*-
"""Группировка правил по номерам блоков: модель назначает, код собирает текст дословно.

Прежняя группировка просила модель ПЕРЕПИСАТЬ правила каждой страницы в группы, и при этом:
- терялись номера пунктов («6.4 Fire-resistance…» → «Fire-resistance…»; в иных документах
  не оставалось ни одного), и ссылки «see 6.2» переставали разрешаться;
- выпадали EXAMPLE, оказавшиеся в начале следующей страницы, и worked examples целиком;
- строки матрицы идентичности («какие атрибуты делают записи одним изделием») раскладывались
  по группам форматирования и отъедали места в выдаче ретрива.

Здесь документ разбирается на пронумерованные блоки (пункт вместе со своими EXAMPLE/NOTE,
строка таблицы, абзац без номера), модель только раскладывает номера блоков по группам,
а текст группы собирается из блоков как есть. Блок, не попавший ни в одну группу и не
отброшенный моделью явно, печатается — потерю видно сразу.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S")
ATTACHED_RE = re.compile(r"^\s*(?:EXAMPLE|NOTE)\b", re.I)


# Экранирование, которого нет в JSON: VLM пишет футы как «\'» («ft, feet, \'»), и страница целиком
# не разбиралась — «error pass» на проверке, а группировка молча выбрасывала её блоки.
_BAD_ESCAPE_RE = re.compile(r'\\(?=[^"\\/bfnrtu])')


def parse_page_content(content) -> dict | None:
    """Содержимое страницы как dict {"blocks": …}, либо None, если его не разобрать."""
    if isinstance(content, dict):
        return _attach_captions(content)
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, _BAD_ESCAPE_RE.sub("", text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _attach_captions(parsed)
    return None


def _attach_captions(content: dict) -> dict:
    """Заголовок, за которым сразу идёт таблица без названия, — подпись этой таблицы.

    VLM пишет подпись то в "title" таблицы, то отдельным блоком heading над ней. Во втором
    случае все таблицы страницы оставались без названий, и указатель таблиц склеивал их в
    одну — продолжение таблицы с прошлой страницы. Продолжение вверху страницы не задевается:
    над ним колонтитул (абзац) или ничего, а не заголовок."""
    blocks = content.get("blocks")
    if not isinstance(blocks, list):
        return content
    merged: list = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (isinstance(block, dict) and block.get("type") == "table" and not str(block.get("title") or "").strip()
                and isinstance(previous, dict) and previous.get("type") == "heading"
                and str(previous.get("text") or "").strip()):
            merged[-1] = {**block, "title": str(previous["text"]).strip()}
            continue
        merged.append(block)
    return {**content, "blocks": merged}


def _blocks(page: dict) -> list[dict]:
    content = parse_page_content(page.get("content"))
    if content is None:
        print(f"WARNING: страница {page.get('page_number')} не разобрана как JSON — её блоки потеряны")
        return []
    return content.get("blocks", [])


def _row_text(row: dict) -> str:
    """Строка таблицы в форме «колонка: 'значение'; …» — её же разбирает применяющий граф."""
    return "; ".join(f"{k}: '{v}'" for k, v in row.items() if str(v).strip())


def build_units(pages: list[dict]) -> list[dict]:
    """Плоский список блоков документа в порядке чтения.

    EXAMPLE/NOTE приклеиваются к предыдущему пункту и через границу страницы; абзац,
    начатый со строчной буквы, продолжает предыдущий; колонтитулы (одна и та же строка на
    большинстве страниц) отбрасываются. Абзацу без номера приписывается номер ближайшего
    заголовка-подпункта («11.1 Raw record 1: …»), чтобы на него было чем сослаться.
    """
    raw: list[tuple[str, str]] = []
    for page in pages:
        for block in _blocks(page):
            kind = block.get("type")
            if kind == "table":
                title = str(block.get("title") or "").strip()
                columns = [str(c) for c in block.get("columns") or []]
                raw.append(("table_head", f"{title} (columns: {'; '.join(columns)})".strip()))
                for row in block.get("rows") or []:
                    if isinstance(row, dict) and any(str(v).strip() for v in row.values()):
                        raw.append(("row", (f"[{title}] " if title else "") + _row_text(row)))
            elif kind in ("paragraph", "heading"):
                text = str(block.get("text", "")).strip()
                if text:
                    raw.append((kind, text))

    counts = Counter(text for kind, text in raw if kind == "paragraph")
    running = {text for text, n in counts.items() if n >= max(3, len(pages) // 2)}

    units: list[dict] = []
    section, heading_clause = "", ""
    for kind, text in raw:
        if text in running or re.fullmatch(r"Page \d+ of \d+", text):
            continue
        if kind == "heading":
            match = CLAUSE_RE.match(text)
            heading_clause = match.group(1) if match else ""
            # VLM размечает заголовком и целый пункт правила («11.4 Valve operators are written
            # Handwheel, with 'HW'…»). Как название раздела его текст в правила не попадал:
            # в одном документе так пропал 31 пункт. Пункт — это номер x.y и полноценный текст.
            if match and "." in match.group(1) and (len(text) > 80 or text.rstrip().endswith(".")):
                units.append({"kind": "clause", "section": section, "text": text})
                continue
            section = text
            continue
        last = units[-1] if units else None
        if kind == "paragraph" and last and last["kind"] == "clause" and ATTACHED_RE.match(text):
            last["text"] += "\n" + text
            continue
        if kind == "paragraph" and last and last["kind"] in ("clause", "text") and text[:1].islower():
            last["text"] += " " + text
            continue
        match = CLAUSE_RE.match(text) if kind == "paragraph" else None
        if match and "." in match.group(1):
            units.append({"kind": "clause", "section": section, "text": text})
        elif kind == "paragraph":
            prefix = f"{heading_clause} " if "." in heading_clause else ""
            units.append({"kind": "text", "section": section, "text": prefix + text})
        else:
            units.append({"kind": kind, "section": section, "text": text})
    for i, unit in enumerate(units):
        unit["id"] = i
    return units


# Сколько символов текста блоков модель раскладывает за один вызов. Документ длиннее режется
# на окна: 82-страничная спецификация (≈310 тыс. символов, 2300 блоков) одним вызовом
# уходила целиком в одну группу Global — модель просто перечисляла id подряд.
WINDOW_CHARS = 25000
GROUPING_ATTEMPTS = 3

_WINDOW_HINT = """

=== ФРАГМЕНТ ДОКУМЕНТА ===
Документ длинный и раскладывается по частям: это часть {part} из {total}. Id блоков сквозные по всему документу.
Группы, уже созданные по предыдущим частям (название — алиасы):
{known}
Если блок этой части о той же величине/атрибуте, что и одна из этих групп, используй В ТОЧНОСТИ её название (и тот же type/kind) — группы сольются. Новую группу создавай только для величины, которой в списке нет."""


def _windows(units: list[dict], limit: int) -> list[list[dict]]:
    """Окна не больше limit символов, разрезанные по смене раздела, а не посреди него.

    Раздел длиннее limit режется по блокам — пункт с его EXAMPLE/NOTE всё равно один блок.
    """
    sections: list[list[dict]] = []
    for unit in units:
        if sections and sections[-1][-1]["section"] == unit["section"]:
            sections[-1].append(unit)
        else:
            sections.append([unit])
    windows: list[list[dict]] = [[]]
    size = 0
    for section in sections:
        section_size = sum(len(u["text"]) for u in section)
        if windows[-1] and size + section_size > limit:
            windows.append([])
            size = 0
        for unit in section:
            if windows[-1] and size + len(unit["text"]) > limit:
                windows.append([])
                size = 0
            windows[-1].append(unit)
            size += len(unit["text"])
    return [w for w in windows if w]


def group_units(units: list[dict], llm, prompt: str,
                window_chars: int | None = WINDOW_CHARS) -> tuple[list[dict], list[dict]]:
    """Возвращает (группы в формате rule_groups.json, блоки, которые никуда не попали).

    Документ, чей текст не больше window_chars, раскладывается одним вызовом; длиннее — по
    окнам, и каждое окно видит названия групп, созданных до него. window_chars=None — всегда
    одним вызовом.
    """
    total = sum(len(u["text"]) for u in units)
    windows = [units] if window_chars is None or total <= window_chars else _windows(units, window_chars)
    groups: list[dict] = []
    lost: list[dict] = []
    for part, window in enumerate(windows, 1):
        hint = ""
        if len(windows) > 1:
            hint = _WINDOW_HINT.format(part=part, total=len(windows), known=_known_groups(groups))
            print(f"группировка: часть {part}/{len(windows)}, блоков {len(window)}")
        lost += _group_window(window, llm, prompt + hint, groups)
    if lost and len(windows) > 1:
        # В окне модель чаще оставляет без группы пункт, чья величина описана в другом окне
        # («Dial size is the nominal case diameter…»): добор видит уже все группы документа.
        print(f"группировка: добор {len(lost)} блоков без группы")
        lost = _group_window(lost, llm, prompt + _LEFTOVER_HINT.format(known=_known_groups(groups)), groups)
    return groups, lost


_LEFTOVER_HINT = """

=== ДОБОР ===
Эти блоки остались без группы после раскладки документа по частям. Все группы документа (название — алиасы):
{known}
Положи каждый блок в группу его величины/атрибута, используя В ТОЧНОСТИ её название (и тот же type/kind); новую группу — только если величины нет в списке; в skipped — только то, в чём нет ни требования, ни примера."""


def _known_groups(groups: list[dict]) -> str:
    return "\n".join(f"- {g['attribute']} [{g['type']}/{g['kind']}]"
                     + (f" — {', '.join(g['aliases'][:6])}" if g["aliases"] else "")
                     for g in groups) or "(пока нет)"


def _group_window(units: list[dict], llm, prompt: str, groups: list[dict]) -> list[dict]:
    """Раскладывает блоки одного окна, дописывая группы в groups; возвращает потерянные блоки."""
    payload = json.dumps([{"id": u["id"], "section": u["section"], "text": u["text"]} for u in units],
                         ensure_ascii=False)
    # Модель изредка возвращает обрезанный или битый JSON (на 82-страничном документе — одно окно
    # из десятка); ответ на тот же вопрос просто спрашивается заново, а не роняет всю экстракцию
    parsed = None
    for attempt in range(1, GROUPING_ATTEMPTS + 1):
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=payload)])
        try:
            parsed = json.loads(str(response.content))
            break
        except json.JSONDecodeError as e:
            print(f"группировка: ответ модели не JSON ({e}); попытка {attempt}/{GROUPING_ATTEMPTS}")
            if attempt == GROUPING_ATTEMPTS:
                raise

    by_id = {u["id"]: u for u in units}
    used: set[int] = set()
    for group in parsed.get("groups", []):
        ids = sorted(i for i in group.get("units", []) if isinstance(i, int) and i in by_id)
        if not ids:
            continue
        used.update(ids)
        rules: list[str] = []
        for i in ids:
            # пункт и его примеры — отдельными строками, как в прежнем формате rule_groups.json
            rules.extend(part for part in by_id[i]["text"].split("\n") if part.strip())
        entry = {
            "attribute": str(group.get("attribute", "")).strip() or "unnamed",
            "type": "global" if group.get("type") == "global" else "local",
            "kind": group.get("kind", "presentation"),
            "aliases": [a for a in group.get("aliases", []) if a],
            "rules": list(dict.fromkeys(rules)),
        }
        # между окнами модель может сменить регистр названия — это всё равно та же группа
        same = next((g for g in groups
                     if g["attribute"].casefold() == entry["attribute"].casefold()
                     and g["type"] == entry["type"]), None)
        if same:
            same["rules"] = list(dict.fromkeys(same["rules"] + entry["rules"]))
            same["aliases"] = list(dict.fromkeys(same["aliases"] + entry["aliases"]))
        else:
            groups.append(entry)

    skipped = {i for i in parsed.get("skipped", []) if isinstance(i, int)}
    return [u for u in units if u["id"] not in used and u["id"] not in skipped]
