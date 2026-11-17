"""Где лежат данные проекта — одно место для всех путей.

Извлечение пишет документы в DOCUMENTS_OUTPUT (папка на документ), сервис их оттуда читает;
сервис держит свои рабочие файлы в SERVICE_DATA.

Две корневые папки можно перенести переменными окружения (или строками в .env) — например,
держать данные вне репозитория или на томе контейнера:
    DOCUMENTS_DATA_DIR — PDF документов и всё, что из них извлечено (input/ и output/);
    SERVICE_DATA_DIR   — рабочие файлы сервиса: кэши индексов, фоновые задачи, сессии чата.
Относительный путь считается от корня репозитория. Без переменных — папки внутри репозитория.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

GRAPHS_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRAPHS_DIR.parent

# пути читаются при импорте — .env должен быть прочитан раньше, кто бы ни импортировал модуль первым
load_dotenv(REPO_ROOT / ".env")


def _dir_from_env(name: str, default: Path) -> Path:
    """Папка из переменной окружения name (относительная — от корня репозитория), иначе default."""
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


# извлечение правил: PDF документов на входе и папка на каждый извлечённый документ на выходе
STRUCTURED_DATA = _dir_from_env("DOCUMENTS_DATA_DIR", GRAPHS_DIR / "structured_documents" / "data")
DOCUMENTS_INPUT = STRUCTURED_DATA / "input"
DOCUMENTS_OUTPUT = STRUCTURED_DATA / "output"

# сервис нормализации: кэши поисковых индексов и тестовые таблицы, фоновые задачи, сессии чата
SERVICE_DATA = _dir_from_env("SERVICE_DATA_DIR", GRAPHS_DIR / "rag_agent" / "data")
SERVICE_INPUT = SERVICE_DATA / "input"
JOBS_DIR = SERVICE_DATA / "jobs"
