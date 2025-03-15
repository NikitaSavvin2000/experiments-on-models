<p align="center">

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![Code Coverage](coverage.svg)

</p>

# Установка проекта

#### Установка проекта
```bash
pip install pdm
pdm install
```


Активация окружения
```bash
source .venv/bin/activate
```


Добавление зависимостей 
```bash
pdm add pandas
```

Удалеение зависимостей
```bash
pdm remove
```


# Описание проекта

### Проект предназначен для проведения эксперементов над моедлями

### Структура проекта 
```bash
    .
├── LICENSE
├── README.md
├── coverage.svg
├── pdm.lock
├── pyproject.toml
├── sonar-project.properties
├── src
│   ├── config.py
│   └── gru_models
│       ├── main.py
│       └── params.yaml
└── tox.ini

```
### Рабочая область это директория src
```bash
├── src
│   ├── config.py
│   └── gru_models
│       ├── main.py
│       └── params.yaml

```
# Правила работы с репозиторием

### 1.  Работы произволятся строго в своей ветке. Ветка наследуется от самой свежей версии (от dev).
### 2.  Названия веток должны быть лаконичными и понятными.
### 3.  Слияние происходит только с веткой dev! И только по утверждению, после ревью.

### 4. Все результаты экспериментов должны оставаться локально и директори с экспериментами должны быть в .gitignor. Если вы забыли добавить в gitignor директорию с экспирементами нужно сделать следущее
```bash
rm -rf <path_to_dir>
git rm -rf <path_to_dir>
```
### 5.  Структура директроий с экспериментами должна быть всегда одинакова.
```bash
│   └── gru_models
│       ├── main.py
│       ├── params.yaml
│       └── experiments
│           └── exp_2025-03-15_13-08-45

```
### 5.1 Пример создания путей в исполняемом файле пример см в src/template/main.py

```bash
home_path = os.getcwd()
home_path = f"{home_path}/src/template"
experiments_path = f"{home_path}/experiments"
dir_name = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
BASE_PATH = f"{experiments_path}/{dir_name}"
os.makedirs(BASE_PATH, exist_ok=True)
params_file = f'{home_path}/params.yaml'
cur_running_path = f"{home_path}/main.py"
```
### 6. При проведении новых экспериментов на уровне директории src создать директорию с соответвующим именем, к примеру
```bash
cd src
mkdir xgboost_model
```
### 7. Все зависимости должны добавлятся через менеджер PDM!
### Делай так!
```bash
pdm add pandas
```
### Так не делай!
```bash
pip install pandas
```
### 8. Шаблон для создания новых экспериментов брать из experiments-on-models/src/template!

