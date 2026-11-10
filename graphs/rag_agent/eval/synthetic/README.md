# Синтетические стандарты и наборы «новых значений»

Проверка пайплайна на значениях, которых **нет** в тексте стандарта (он не должен узнавать
примеры, а должен применять правила), и на стандартах, собранных под трудные случаи.
Всё запускается из корня репозитория через `venv2`. Каждый прогон вызывает LLM (≈ 6 вызовов на строку).

Не запускайте прогоны, пока идёт извлечение: скрипт при старте читает все папки
`structured_documents/data/output/*` и может поймать недописанный `rule_groups.json`.

## Что где

| Файл | Что делает |
|---|---|
| `eval_doc.py` | Прогон одного стандарта по одной таблице со сверкой с эталоном: точность, расхождения, лишние и пропущенные жёлтые пометки |
| `count_calls.py` | То же без сверки, но считает вызовы LLM и токены (вход, из кэша, выход) |
| `regroup.py` | Перегруппировывает готовый `data_after_check.json` документа в папку `<документ>__v2` (без VLM) |
| `make_cwx.py`, `make_pkx.py`, `make_bhx.py` | Генерируют PDF синтетического стандарта в `structured_documents/data/input/`, таблицу в `rag_agent/data/input/` и эталон в `expected/` |
| `make_unseen.py` + `unseen_b.py` | Генерируют наборы новых значений A и B для HVX/CWX/PKX (`<ключ>_unseen_a.xlsx` / `_b.xlsx`); падают, если значение встречается в тексте стандарта |
| `expected/*.json` | Эталоны: `{"expected": строка, список вариантов или {"re": регулярка}, "review": true/false/null}` по строкам таблицы |

Стандарты:

- **CWX-STD 221** — ссылки на пункт другого стандарта (OGX-STD 410) и на отсутствующий (ZZX-STD 000), знаки после запятой по классам оборудования, форма `<n.n> kW` только в приложении.
- **PKX-STD 330** — правила только в примерах в начале следующей страницы, правило только в образце датащита приложения, таблица кодов разорвана между страницами.
- **BHX-STD 077** — ни одного примера «было → стало», только описание правильного вида.

## Команды

```bash
# один прогон со сверкой
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.eval_doc \
    PKX-STD-330_package_compressor_normalization_rules__v2 \
    graphs/rag_agent/data/input/pkx_unseen_a.xlsx \
    graphs/rag_agent/eval/synthetic/expected/pkx_unseen_a.json

# вызовы и токены
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.count_calls \
    PKX-STD-330_package_compressor_normalization_rules__v2 graphs/rag_agent/data/input/pkx_unseen_a.xlsx

# перегруппировать документ новой группировкой в <документ>__v2
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.regroup HVX-STD-134_hvac_ventilation_equipment_normalization_rules

# пересоздать синтетический стандарт (потом извлечение через STRUCTURED_DOC_PATH=...)
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.make_bhx
# пересоздать наборы новых значений
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.make_unseen a
venv2/Scripts/python -m graphs.rag_agent.eval.synthetic.make_unseen b
```

Пары «таблица → эталон»:

| Таблица (`rag_agent/data/input/`) | Эталон (`expected/`) | Стандарт |
|---|---|---|
| `hvac_ventilation_equipment_test_corpus.xlsx` | `hvac_expected.json` | HVX-STD-134…`__v2` |
| `hvac_unseen_a.xlsx` / `_b` | `hvac_unseen_a.json` / `_b` | HVX-STD-134…`__v2` |
| `cwx_chilled_water_test_corpus.xlsx` | `cwx_expected.json` | CWX-STD-221…`__v2` |
| `cwx_unseen_a.xlsx` / `_b` | `cwx_unseen_a.json` / `_b` | CWX-STD-221…`__v2` |
| `pkx_compressor_test_corpus.xlsx` | `pkx_expected.json` | PKX-STD-330…`__v2` |
| `pkx_unseen_a.xlsx` / `_b` | `pkx_unseen_a.json` / `_b` | PKX-STD-330…`__v2` |
| `bhx_conveyor_test_corpus.xlsx` | `bhx_expected.json` | BHX-STD-077… |

## Базовые цифры (24.09.2026, новая группировка + текущий рантайм)

| Набор | Точность |
|---|---|
| HVX / CWX / PKX, значения из стандарта | 100% во всех прогонах |
| HVX, новые значения A + B | 98.4–100% |
| CWX, новые значения A + B | 98.6–100% |
| PKX, новые значения A + B | 97.6–100% |
| BHX (без примеров) | 37–38 из 39 |
| `run_eval` (XYZ STD 100/205) | 98.1–99.1% |

Известные постоянные расхождения: BHX `Deck count: 2` (guard убирает «decks», обязательное по
правилу, когда в стандарте нет примера), BHX `Discharge temperature: 60 °C` (пункт 4.3 велит пометить,
пометки нет), разброс модели в регистре (`cast Aluminium`) и арифметике (`53.6 F → 13°C`).
