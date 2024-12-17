import tensorflow as tf

# 모델 로드
interpreter = tf.lite.Interpreter(model_path="car_detection_raw_audio_model.tflite")
interpreter.allocate_tensors()

# 입력 텐서 정보 출력
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])  # 예상: [1, 96000]
print("Output shape:", output_details[0]['shape'])  # 예상: [1, 1]