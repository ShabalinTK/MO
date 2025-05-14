import streamlit as st
from voice_recognition import preprocess_audio, extract_features, predict_speaker
from joblib import load
import os

st.title("Распознавание голоса")
st.write("Загрузите аудиофайл (WAV, 16 кГц, моно), чтобы определить спикера.")

# Загрузка модели и скейлера
try:
    model = load("model.joblib")
    scaler = load("scaler.joblib")
except FileNotFoundError:
    st.error("Модель не найдена. Сначала обучите модель, запустив voice_recognition.py.")
    st.stop()

# Загрузка файла
uploaded_file = st.file_uploader("Выберите аудиофайл", type=["wav"])

if uploaded_file:
    # Сохранение временного файла
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Предсказание
    with st.spinner("Обработка аудио..."):
        speaker = predict_speaker(temp_path, model, scaler)
        if speaker:
            st.success(f"Спикер: {speaker}")
        else:
            st.error("Ошибка обработки аудио. Проверьте формат файла.")
    
    # Удаление временного файла
    if os.path.exists(temp_path):
        os.remove(temp_path)