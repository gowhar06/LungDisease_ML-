import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
MODEL_PATH = 'lung_disease_model.h5'
model = load_model(MODEL_PATH) 
def predict_image(img): 
    img = img.resize((224, 224))  # Adjust the size to match your model's expected input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension 
    prediction = model.predict(img_array) 
    class_names = ['Lung Opacity', 'Normal', 'Viral Pneumonia'] 
    return class_names[np.argmax(prediction)] 
about_content = """
## About Lungs
The lungs are essential organs in the respiratory system, responsible for oxygenating blood and removing carbon dioxide from the body. They play a crucial role in maintaining overall health by enabling the exchange of gases during breathing. Healthy lungs ensure that every part of the body receives adequate oxygen, which is vital for energy production and proper functioning.

### Common Lung Disease Classifications: 
#### 1. Lung Opacity:
    - What is it?  
        Lung opacity refers to abnormal areas on a chest X-ray or CT scan that appear denser than the surrounding lung tissue. 
        It is not a disease itself but an indicator of underlying issues.
    - Causes:  
        Infections (like pneumonia), inflammation, tumors, or fluid build-up.
    - Symptoms: 
        Shortness of breath, coughing, and chest discomfort.

#### 2. Viral Pneumonia:
    - What is it?  
        Viral pneumonia is a lung infection caused by viruses, leading to inflammation of the air sacs (alveoli). 
        This can result in fluid or pus filling the lungs, making breathing difficult.
    - Causes:  
        Influenza, coronaviruses, or respiratory syncytial virus (RSV).
    - Symptoms: 
        Fever, persistent cough, difficulty breathing, fatigue, and chest pain.

### **Why Early Detection is Important**:
Lung diseases can severely impact quality of life and, in some cases, become life-threatening. Timely diagnosis and treatment help prevent complications and improve patient outcomes.
"""# Gradio interface
with gr.Blocks() as interface: 
    gr.Markdown("## Welcome to the Lung Disease Detection System! 🚑")
    gr.Markdown("Upload a chest X-ray image to classify it as Lung Opacity, Normal, or Viral Pneumonia.")
    # Prediction interface
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Chest X-Ray")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction")
    submit_btn = gr.Button("Click here to Classify")
    submit_btn.click(predict_image, inputs=image_input, outputs=prediction_output) 
    gr.Markdown(about_content)
# Launch the interface
if __name__ == "__main__":
    interface.launch(server_port=7860, server_name="0.0.0.0", share=True)
    