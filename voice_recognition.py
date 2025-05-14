import librosa
import numpy as np
import webrtcvad
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

# Предобработка аудио
def preprocess_audio(audio_path, sample_rate=16000):
    """
    Загружает аудио, снижает шум и выделяет голосовые фрагменты.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Инициализация VAD
        vad = webrtcvad.Vad(3)  # Уровень агрессивности 3
        audio_int16 = (audio * 32768).astype(np.int16)
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        
        # Сегментация голоса
        segments = []
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame = audio_int16[i:i + frame_size]
            if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate):
                segments.append(audio[i:i + frame_size])
        
        return np.concatenate(segments) if segments else audio
    except Exception as e:
        print(f"Ошибка обработки {audio_path}: {e}")
        return None

# Извлечение признаков
def extract_features(audio, sample_rate=16000):
    """
    Извлекает MFCC признаки.
    """
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Ошибка извлечения признаков: {e}")
        return None

# Обучение модели
def train_model(X_train, y_train, model_path="model.joblib", scaler_path="scaler.joblib"):
    """
    Обучает и сохраняет модель.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Сохранение
    dump(model, model_path)
    dump(scaler, scaler_path)
    return model, scaler

# Предсказание для одного аудио
def predict_speaker(audio_path, model, scaler, sample_rate=16000):
    """
    Предсказывает спикера для аудиофайла.
    """
    audio = preprocess_audio(audio_path, sample_rate)
    if audio is None:
        return None
    features = extract_features(audio, sample_rate)
    if features is None:
        return None
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)[0]

# Основная функция
def main():
    # Путь к датасету
    data_dir = "dataset/"
    X, y = [], []
    
    # Загрузка данных
    for speaker in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(speaker_path, audio_file)
                    audio = preprocess_audio(audio_path)
                    if audio is None:
                        continue
                    features = extract_features(audio)
                    if features is None:
                        continue
                    X.append(features)
                    y.append(speaker)
    
    if not X:
        print("Ошибка: нет данных для обучения.")
        return
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение
    model, scaler = train_model(X_train, y_train)
    
    # Оценка на тестовых данных
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Точность на тестовых данных: {accuracy:.2f}")
    
    # Тест на новом аудио
    test_audio = "test_audio.wav"
    if os.path.exists(test_audio):
        speaker = predict_speaker(test_audio, model, scaler)
        if speaker:
            print(f"Предсказанный спикер для {test_audio}: {speaker}")
        else:
            print("Ошибка предсказания.")

if __name__ == "__main__":
    main()