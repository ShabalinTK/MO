# Voice Classifier

Проект для распознавания спикеров по аудиофайлам. Система использует MFCC признаки и модель RandomForest для определения спикера.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ShabalinTK/MO.git
   cd voice-classifier
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Подготовка датасета

1. Создайте папку `dataset/` с подпапками для спикеров:
   ```
   dataset/
   ├── Ed/
   │   ├── audio1.wav
   ├── Hom/
   │   ├── audio1.wav
   ...
   ```
2. Файлы должны быть WAV, 16 кГц, моно.

## Использование

1. Обучите модель:
   ```bash
   python voice_recognition.py
   ```

2. Запустите веб-интерфейс:
   ```bash
   streamlit run app.py
   ```

## Требования

- Python 3.8 или 3.9
- Microsoft Visual C++ Build Tools
