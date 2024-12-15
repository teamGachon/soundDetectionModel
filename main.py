from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import librosa
import tensorflow as tf
import logging
from io import BytesIO

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 로드
MODEL_PATH = "car_detection_cnn_model2_48kHz.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 특징 추출 함수
def extract_features_from_audio(file, target_sample_rate=16000, max_pad_len=44):
    audio, sample_rate = librosa.load(file, sr=target_sample_rate, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc[np.newaxis, ..., np.newaxis]

# API 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 파일을 메모리에서 읽기
        audio_data = BytesIO(await file.read())

        # 특징 추출
        features = extract_features_from_audio(audio_data)

        # 모델 예측
        prediction = model.predict(features)

        # 예측 값 및 차량 감지 여부 계산
        prediction_value = float(prediction[0][0])  # 모델 예측 값 (float)
        vehicle_detected = prediction_value < 0.5  # 차량 감지 여부 (true/false)

        # 실시간 결과 로그 출력
        logger.info(f"result: {prediction_value:.8f}, Vehicle Detected: {vehicle_detected}")

        # 결과 반환
        return {
            "vehicleDetected": vehicle_detected,
            "result": prediction_value
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

# 메인 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
