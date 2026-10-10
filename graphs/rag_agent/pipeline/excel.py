"""Запись результата в .xlsx с жёлтой заливкой спорных ячеек."""
import io
import re

import pandas as pd


# Excel не принимает управляющие символы и обрывается на ячейке длиннее 32767 знаков.
_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


EXCEL_CELL_LIMIT = 32767


def _excel_safe(value):
    """Значение → то, что Excel примет: без управляющих символов и не длиннее предела."""
    if not isinstance(value, str):
        return value
    cleaned = _ILLEGAL_XLSX_RE.sub("", value)
    return cleaned[:EXCEL_CELL_LIMIT] if len(cleaned) > EXCEL_CELL_LIMIT else cleaned


def _write_excel_with_highlight(df: pd.DataFrame, ambiguous_cols: dict[str, list[bool]],
                                layout: dict | None = None) -> io.BytesIO:
    """Пишет DataFrame в Excel, заливая спорные ячейки жёлтым; у результата — лист «Сводка»."""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    wb = Workbook()
    ws = wb.active
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    ws.append([_excel_safe(c) for c in df.columns])
    for _, row in df.iterrows():
        ws.append([_excel_safe(v) for v in row.tolist()])
    # записанное значение — это данные, а не формула: «=SUM(...)» из датащита должно
    # остаться текстом, а не вычисляться при открытии файла
    for row in ws.iter_rows():
        for cell in row:
            if cell.data_type == "f":
                cell.data_type = "s"

    col_index = {name: i for i, name in enumerate(df.columns, start=1)}
    for col_name, flags in ambiguous_cols.items():
        c = col_index[col_name]
        for r, is_amb in enumerate(flags, start=2):
            if is_amb:
                ws.cell(row=r, column=c).fill = yellow

    if layout and layout.get("standards"):
        _write_summary(wb.create_sheet("Сводка"), df, layout, ambiguous_cols)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _write_summary(ws, df: pd.DataFrame, layout: dict, ambiguous_cols: dict[str, list[bool]]) -> None:
    """Лист «Сводка»: метрики по стандартам и где больше всего жёлтых строк — то же, что
    отвечает агент на «какой item больше всего с ошибками» (modules.results)."""
    from openpyxl.styles import Font
    from ..modules import results

    bold = Font(bold=True)

    def title(text):
        ws.append([])
        ws.append([text])
        ws.cell(row=ws.max_row, column=1).font = bold

    overview = results.overview(df, layout, ambiguous_cols)
    ws.append(["Строк", overview["rows"]])
    title("По стандартам")
    ws.append(["Стандарт", "Жёлтых (flagged)", "Изменено (changed)", "Не подтверждено кодом (low_confidence)",
               "Не распространяется"])
    for entry in overview["standards"]:
        ws.append([entry["standard"], entry["flagged"], entry["changed"], entry["low_confidence"],
                   entry["not_covered"]])
    groupings = [["Attribute"]] + [[c] for c in layout["context_columns"][:1]]
    if layout["context_columns"]:
        groupings.append([layout["context_columns"][0], "Attribute"])
    for keys in groupings:
        top = results.aggregate(df, layout, ambiguous_cols, keys, "flagged", top=15)
        if not top.get("top"):
            continue
        title("Больше всего жёлтых строк: " + " × ".join(top["group_by"]))
        ws.append(top["group_by"] + ["Жёлтых", "Из строк", "Доля"])
        for group in top["top"]:
            ws.append([group[k] for k in top["group_by"]] + [group["count"], group["of_rows"], group["share"]])
    for column in ws.columns:
        width = max(len(str(c.value or "")) for c in column)
        ws.column_dimensions[column[0].column_letter].width = min(60, max(10, width + 2))


XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def read_flagged_cells(content: bytes, columns: list[str]) -> dict[str, list[bool]]:
    """Жёлтые ячейки присланного обратно результата: колонка → флаг на каждую строку данных.

    pandas заливку не читает, а без неё в прошлом результате не найти, что было помечено на
    проверку. Учитываются только колонки из columns (колонки стандартов).
    """
    from openpyxl import load_workbook

    try:
        ws = load_workbook(io.BytesIO(content)).active
    except Exception:
        return {}
    header = [str(c.value) if c.value is not None else "" for c in ws[1]]
    flags: dict[str, list[bool]] = {}
    for name in columns:
        if name not in header:
            continue
        index = header.index(name) + 1
        flags[name] = [
            str(ws.cell(row=r, column=index).fill.start_color.rgb or "").upper().endswith("FFFF00")
            and ws.cell(row=r, column=index).fill.fill_type == "solid"
            for r in range(2, ws.max_row + 1)]
    return flags
