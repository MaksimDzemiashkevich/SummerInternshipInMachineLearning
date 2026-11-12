"""Фоновые задачи диалога: извлечение документа и долгая нормализация.

Пока задача идёт, чат свободен: агент отвечает сразу, а готовый результат фронтенд забирает сам
(GET /jobs) и показывает отдельным сообщением с файлом.

Извлечение — отдельный процесс (graphs/structured_documents/extract_job.py), отвязанный от сервера:
оно идёт десятки минут, и перезапуск бэкенда (--reload на каждой правке кода) его не обрывает. Его
состояние процесс пишет в файл, сервер этот файл читает. Одновременно идёт одно извлечение — VLM одна,
остальные ждут в очереди. Закончилось — документ подгружается в работающий сервис (add_rule_set) без
перезапуска.

Нормализация — поток в процессе сервера: ей нужны модель и наборы правил, которые уже загружены.
Перезапуск сервера её обрывает; такая задача помечается «прервана».
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from graphs.paths import JOBS_DIR

logger = logging.getLogger("normalization_pipeline")

ROOT = Path(__file__).resolve().parents[3]
JOBS_FILE = JOBS_DIR / "jobs.json"
# Как часто наблюдатель проверяет извлечения и запускает следующее из очереди.
TICK_SECONDS = 5


@dataclass
class Job:
    """Фоновая задача диалога: извлечение документа или нормализация файла; сохраняется в jobs.json."""
    job_id: str
    session_id: str
    kind: str                       # "extract" | "normalize"
    title: str
    state: str = "queued"           # queued | running | done | failed | lost
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    result: dict = field(default_factory=dict)
    error: str = ""
    delivered: bool = False         # результат уже ушёл пользователю (в ответе чата или через /jobs)
    pdf: str = ""                   # извлечение: путь к PDF
    document: str = ""              # извлечение: имя документа (папка в data/output)
    total_pages: int = 0
    step: str = ""


class Progress:
    """Сколько строк нормализации готово — для «сколько осталось?». Живёт в памяти: после
    перезапуска сервера нормализация всё равно потеряна."""

    def __init__(self) -> None:
        self.total = 0
        self.done = 0
        self._lock = threading.Lock()

    def add_total(self, rows: int) -> None:
        """Увеличивает общее число ячеек, которые предстоит посчитать."""
        with self._lock:
            self.total += rows

    def step(self, rows: int = 1) -> None:
        """Отмечает готовые ячейки (не больше общего числа)."""
        with self._lock:
            self.done = min(self.total, self.done + rows) if self.total else self.done + rows


_PROGRESS: dict[str, Progress] = {}
_JOBS: dict[str, Job] = {}
_EVENTS: dict[str, threading.Event] = {}
_LOCK = threading.RLock()
_ON_EXTRACTED: list[Callable[[Job], dict]] = []
_WATCHER: threading.Thread | None = None


# ---------------------------------------------------------------------- реестр
def _persist() -> None:
    """Записывает список задач на диск целиком (через временный файл)."""
    with _LOCK:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(j) for j in _JOBS.values()], ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(JOBS_FILE)


def _load() -> None:
    """Задачи прошлого запуска сервера: извлечение продолжает идти своим процессом, нормализация — нет."""
    if not JOBS_FILE.exists():
        return
    try:
        rows = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except ValueError:
        return
    week_ago = time.time() - 7 * 86400
    for row in rows:
        job = Job(**row)
        if job.created < week_ago:
            continue
        if job.kind == "normalize" and job.state in ("queued", "running"):
            job.state, job.error, job.finished = "lost", "сервер перезапускался во время нормализации", time.time()
        _JOBS[job.job_id] = job


_load()


def get(job_id: str) -> Job | None:
    """Задача по её id (None — такой нет)."""
    return _JOBS.get(str(job_id).strip())


def for_session(session_id: str) -> list[Job]:
    """Задачи одного диалога."""
    return [j for j in _JOBS.values() if j.session_id == session_id]


# ---------------------------------------------------------------------- извлечение
def _status_path(job: Job) -> Path:
    """Файл состояния процесса извлечения."""
    return JOBS_DIR / f"{job.job_id}.status.json"


def _log_path(job: Job) -> Path:
    """Журнал процесса извлечения."""
    return JOBS_DIR / f"{job.job_id}.log"


def queue_extraction(session_id: str, pdf: Path) -> Job:
    """Ставит документ в очередь извлечения (запустится, когда освободится VLM)."""
    try:
        import fitz
        pages = len(fitz.open(pdf))
    except Exception:
        pages = 0
    job = Job(job_id=uuid.uuid4().hex[:8], session_id=session_id, kind="extract",
              title=f"извлечение {pdf.stem}", pdf=str(pdf), document=pdf.stem, total_pages=pages)
    with _LOCK:
        _JOBS[job.job_id] = job
        _persist()
    tick()
    return job


def _spawn(job: Job) -> None:
    """Запускает извлечение отдельным процессом, который переживает перезапуск сервера."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "graphs.structured_documents.extract_job", "--pdf", job.pdf,
               "--status", str(_status_path(job)), "--log", str(_log_path(job))]
    options: dict = {"cwd": str(ROOT), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                     "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        # отдельная группа процессов без консоли: перезапуск сервера его не останавливает
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        options["start_new_session"] = True
    subprocess.Popen(command, **options)
    job.state, job.step = "running", "запуск"
    logger.info("[JOBS] извлечение %s запущено (%s)", job.document, job.job_id)


def _alive(pid: int) -> bool:
    """Жив ли процесс; если проверить нечем, считается живым."""
    # os.kill(pid, 0) на Windows не проверяет процесс, а завершает его
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True


def _refresh(job: Job) -> None:
    """Состояние извлечения из файла, который пишет его процесс."""
    path = _status_path(job)
    if not path.exists():
        if time.time() - job.created > 120 and job.step == "запуск":
            job.state, job.error, job.finished = "failed", "процесс извлечения не запустился", time.time()
        return
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return  # файл как раз переписывается
    job.step = f"шаг {status.get('step')}/{status.get('steps')}: {status.get('step_name', '')}"
    if status.get("state") == "running" and status.get("pid") and not _alive(status["pid"]):
        job.state, job.error, job.finished = "failed", "процесс извлечения завершился, не дописав состояние", time.time()
    elif status.get("state") == "failed":
        job.state, job.finished = "failed", status.get("finished") or time.time()
        job.error = f"{status.get('error', '')} на шаге «{status.get('step_name', '')}»"
        job.result = {"log_tail": status.get("log_tail", "")}
    elif status.get("state") == "done":
        job.state, job.finished = "done", status.get("finished") or time.time()
        for callback in _ON_EXTRACTED:
            try:
                job.result.update(callback(job) or {})
            except Exception as e:
                logger.warning("[JOBS] документ %s извлечён, но не подгружен: %s", job.document, e)
                job.result["load_error"] = str(e)


def pages_done(job: Job) -> int:
    """Сколько страниц уже прочитала VLM (по промежуточному файлу графа извлечения)."""
    folder = ROOT / "graphs" / "structured_documents" / "data" / "output" / job.document
    for name in ("pages_data.partial.json", "pages_data.json"):
        try:
            return len(json.loads((folder / name).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return 0


def on_extracted(callback: Callable[[Job], dict]) -> None:
    """Что сделать с извлечённым документом (подгрузить в сервис); вернуть сведения для отчёта."""
    _ON_EXTRACTED.append(callback)


def tick() -> None:
    """Обновляет идущие извлечения и запускает следующее из очереди, если VLM свободна."""
    with _LOCK:
        changed = False
        for job in [j for j in _JOBS.values() if j.kind == "extract" and j.state == "running"]:
            before = job.state
            _refresh(job)
            changed |= job.state != before
        if not any(j.kind == "extract" and j.state == "running" for j in _JOBS.values()):
            queued = sorted((j for j in _JOBS.values() if j.kind == "extract" and j.state == "queued"),
                            key=lambda j: j.created)
            if queued:
                _spawn(queued[0])
                changed = True
        if changed:
            _persist()


def start_watcher() -> None:
    """Фоновый наблюдатель за извлечениями (запускается при старте приложения)."""
    global _WATCHER
    if _WATCHER is not None and _WATCHER.is_alive():
        return

    def loop() -> None:
        while True:
            try:
                tick()
            except Exception as e:
                logger.warning("[JOBS] наблюдатель: %s", e)
            time.sleep(TICK_SECONDS)

    _WATCHER = threading.Thread(target=loop, name="jobs-watcher", daemon=True)
    _WATCHER.start()


# ---------------------------------------------------------------------- нормализация
def run_in_background(session_id: str, title: str, work: Callable[[], dict],
                      progress: Progress | None = None) -> Job:
    """Запускает работу в потоке; work() возвращает сведения о результате (file_id, сводку).
    progress — счётчик готовых строк, который work() заполняет по ходу."""
    job = Job(job_id=uuid.uuid4().hex[:8], session_id=session_id, kind="normalize", title=title,
              state="running", step="идёт")
    event = threading.Event()
    with _LOCK:
        _JOBS[job.job_id] = job
        _EVENTS[job.job_id] = event
        if progress is not None:
            _PROGRESS[job.job_id] = progress
        _persist()

    def run() -> None:
        try:
            job.result = work() or {}
            job.state = "done"
        except Exception as e:
            logger.warning("[JOBS] %s (%s) не удалась: %s", title, job.job_id, e)
            job.state, job.error = "failed", str(e)
        job.finished = time.time()
        with _LOCK:
            _persist()
        event.set()

    threading.Thread(target=run, name=f"job-{job.job_id}", daemon=True).start()
    return job


def wait(job: Job, seconds: float) -> bool:
    """Ждёт задачу не дольше seconds; True — закончилась."""
    event = _EVENTS.get(job.job_id)
    return bool(event and event.wait(seconds))


def mark_delivered(job: Job) -> None:
    """Отмечает, что сообщение о завершении задачи уже отдано в чат."""
    with _LOCK:
        job.delivered = True
        _persist()


# ---------------------------------------------------------------------- для агента и фронтенда
def describe(job: Job) -> dict:
    """Факты задачи для агента и интерфейса: состояние, шаг, прогресс страниц или ячеек."""
    info = {"job_id": job.job_id, "kind": job.kind, "title": job.title, "state": job.state,
            "step": job.step, "error": job.error}
    if job.kind == "extract":
        info.update(document=job.document, pages_total=job.total_pages, pages_read=pages_done(job))
    progress = _PROGRESS.get(job.job_id)
    if job.kind == "normalize" and progress is not None and progress.total:
        # строки × стандарты (и колонка разовых правил): столько ячеек считается всего
        info.update(cells_done=progress.done, cells_total=progress.total,
                    percent=round(100 * progress.done / progress.total))
    if job.state in ("done", "failed"):
        info["result"] = {k: v for k, v in job.result.items() if k != "log_tail"}
        info["minutes"] = round((job.finished - job.created) / 60, 1) if job.finished else None
    return info


def active(session_id: str) -> list[dict]:
    """Незавершённые задачи диалога (в очереди или идут)."""
    return [describe(j) for j in for_session(session_id) if j.state in ("queued", "running")]


def take_finished(session_id: str) -> list[Job]:
    """Закончившиеся задачи, результат которых пользователь ещё не видел; помечает их доставленными."""
    with _LOCK:
        finished = [j for j in for_session(session_id)
                    if j.state in ("done", "failed", "lost") and not j.delivered]
        for job in finished:
            job.delivered = True
        if finished:
            _persist()
    return finished
