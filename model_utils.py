from tensorflow.keras.models import load_model, Model
from project_codes.codes.config import model_path

model = load_model(model_path, compile=False)
#model = Model(inputs=model.input, outputs=model.output)

def get_model():
  return model