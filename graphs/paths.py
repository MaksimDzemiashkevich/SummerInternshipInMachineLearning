"""Где лежат данные проекта — одно место для всех путей.

Извлечение пишет документы в DOCUMENTS_OUTPUT (папка на документ), сервис их оттуда читает;
сервис держит свои рабочие файлы в SERVICE_DATA.
"""
from pathlib import Path

GRAPHS_DIR = Path(__file__).resolve().parent

# извлечение правил: PDF документов на входе и папка на каждый извлечённый документ на выходе
STRUCTURED_DATA = GRAPHS_DIR / "structured_documents" / "data"
DOCUMENTS_INPUT = STRUCTURED_DATA / "input"
DOCUMENTS_OUTPUT = STRUCTURED_DATA / "output"

# сервис нормализации: кэши поисковых индексов и тестовые таблицы, фоновые задачи, сессии чата
SERVICE_DATA = GRAPHS_DIR / "rag_agent" / "data"
SERVICE_INPUT = SERVICE_DATA / "input"
JOBS_DIR = SERVICE_DATA / "jobs"
