import onnxruntime as ort
import joblib
import numpy as np
import pandas as pd
import os

# Path ke file
MODEL_PATH = "models/model_cnn.onnx"
SCALER_PATH = "models/scaler_stunting.pkl"

class StuntingDetector: 
    def __init__(self):
        self.scaler = joblib.load(SCALER_PATH)
        
        self.feature_names = self.scaler.feature_names_in_
        
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: list):
        input_df = pd.DataFrame([features], columns=self.feature_names)

        scaled_data = self.scaler.transform(input_df)

        input_3d = scaled_data.reshape(1, 7, 1).astype(np.float32)

        prediction = self.session.run(None, {self.input_name: input_3d})
        
        prob_array = prediction[0]
        
        probability = float(prob_array.flatten()[0])
        
        status = "Stunting" if probability > 0.5 else "Normal"
        
        return {
            "status": status,
            "probability": round(probability, 4),
            "input_data": features
        }

detector = StuntingDetector()