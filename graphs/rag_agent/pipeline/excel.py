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


def _write_excel_with_highlight(df: pd.DataFrame, ambiguous_cols: dict[str, list[bool]]) -> io.BytesIO:
    """Пишет DataFrame в Excel, заливая спорные ячейки жёлтым."""
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

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
