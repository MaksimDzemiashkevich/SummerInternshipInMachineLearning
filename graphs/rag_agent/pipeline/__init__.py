"""Конвейер нормализации таблицы по документам-стандартам.

Слои импортируют только предыдущие: common → answers / subjects → deterministic → retrieval →
guards → normalizer → postprocess → frames / graph → user_rules → excel / runner.

Два пути нормализации столбца документа (runner._run_rule_set):
- основной — исполнитель паспортов атрибутов (graphs/rag_agent/spec_runner.py), для документов с
  attribute_spec.json; из этого пакета он берёт frames, subjects, parsing и common;
- запасной — граф по строкам (graph.py) с поиском правил (retrieval), детерминированными таблицами
  (deterministic), проверками ответа (guards) и пост-проходами (postprocess): для документов без
  паспортов и при NORMALIZER=legacy. Его же механикой пользуется колонка разовых правил (user_rules).
"""
