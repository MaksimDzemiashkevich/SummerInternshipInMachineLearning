# -*- coding: utf-8 -*-
"""Фоновое извлечение одного документа: граф извлечения → паспорта атрибутов → их проверка.

Запускает API отдельным процессом (graphs/rag_agent/modules/jobs.py), отвязанным от сервера: перезапуск
бэкенда (--reload на каждой правке кода) извлечение, идущее полчаса, не обрывает. Каждый шаг — свой
процесс тех же скриптов, что запускаются руками; состояние пишется в --status (JSON), вывод — в --log.

    venv2/Scripts/python -m graphs.structured_documents.extract_job --pdf <путь к PDF> --status <json> --log <txt>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    pdf, status, log = Path(args.pdf), Path(args.status), Path(args.log)
    doc = pdf.stem
    python = sys.executable
    # вывод скриптов — по-русски и с символами вроде «×»: в кодировке консоли Windows он падает
    env = {**os.environ, "STRUCTURED_DOC_PATH": str(pdf), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    steps = [
        ("извлечение текста и правил (VLM, группировка, роли перечней)",
         [python, "-m", "graphs.structured_documents.structured_documents_graph"]),
        ("паспорта атрибутов", [python, "-m", "graphs.structured_documents.attribute_spec", "--doc", doc]),
        ("проверка паспортов", [python, "-m", "graphs.structured_documents.spec_verify", "--doc", doc]),
    ]
    started = time.time()
    for number, (name, command) in enumerate(steps, start=1):
        _write(status, {"state": "running", "step": number, "steps": len(steps), "step_name": name,
                        "pid": os.getpid(), "started": started})
        with log.open("a", encoding="utf-8") as out:
            out.write(f"\n=== шаг {number}/{len(steps)}: {name} ===\n")
            out.flush()
            code = subprocess.call(command, stdout=out, stderr=subprocess.STDOUT, env=env)
        if code != 0:
            _write(status, {"state": "failed", "step": number, "steps": len(steps), "step_name": name,
                            "error": f"шаг завершился с кодом {code}", "log_tail": _tail(log),
                            "started": started, "finished": time.time()})
            return
    _write(status, {"state": "done", "step": len(steps), "steps": len(steps), "document": doc,
                    "started": started, "finished": time.time()})


if __name__ == "__main__":
    main()
