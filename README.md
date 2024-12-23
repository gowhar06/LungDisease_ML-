![image](https://github.com/user-attachments/assets/2066160a-9442-4a1e-a821-3c2c71e12702)

# LUNG DISEASE CLASSIFICATION AND DETECTION USING MACHINE LEARNING TECHNIQUES 
Assistance for doctors in disease detection can be very useful in environments with scarce resources and personnel. Historically, many patients could have been cured with early detection of the disease. 
To assist doctors, it is essential to have a versatile system that can timely detect multiple diseases in the lungs with high accuracy. 
The goal of this project is to develop a system for the automated classification of lung diseases, specifically focusing on Pneumonia, Tuberculosis, and COVID-19, using machine learning techniques. 
Early and accurate detection of these diseases is critical for effective treatment; however, manual analysis of chest X-ray or CT scan images is often labor-intensive and prone to human error. 
This project leverages Image Processing Toolbox and Deep Learning Toolbox to create a streamlined process for identifying lung diseases from medical imaging data. 
The system consists of four main stages: image preprocessing, feature extraction, model training, and classification. 
Image preprocessing involves resizing, normalization, and augmentation to enhance data quality. 
Model performance is evaluated using metrics such as accuracy, precision, and recall. 
This approach provides an efficient and reliable solution to assist healthcare professionals in early disease detection and informed clinical decision-making.


## Table of Contents
  * Project Overview
  * Features
  * Technologies Used
  * Dataset
  * Model Training and Evaluation 


## 🚀 Project Overview
Lung diseases are a significant public health concern, often requiring timely and accurate diagnosis to prevent severe health complications. 
This project leverages machine learning to create an automated system for:
  * 🔍 Detection: Identifying lung abnormalities from medical images.
  * 📊 Classification: Differentiating between diseases such as pneumonia, tuberculosis, and lung cancer.
The primary goal is to assist healthcare professionals by reducing diagnostic time and improving accuracy, especially in under-resourced areas.

## ✨ Features  

- **Automated Detection**:  
  Quickly detects abnormalities in lung imaging data.  

- **Disease Classification**:  
  Classifies lung diseases like pneumonia, tuberculosis, and lung cancer.  

- **User-Friendly Deployment**:  
  Can be deployed as a web application or API for real-time predictions.  

- **Scalable and Robust**:  
  Capable of processing large datasets efficiently.  

- **Explainability**:  
  Provides interpretable results to aid healthcare professionals.  

## 🛠️ Technologies Used  

### **Programming Languages & Frameworks** 💻  
- **Python** 🐍: Core programming language for model development and deployment.  
- **TensorFlow/Keras** 🤖: Frameworks used for building and training the deep learning model.  
- **Gradio** 🎨: For creating an interactive and user-friendly interface.

### **Deployment Platform** 🌐  
- **Hugging Face Spaces** 🧠: Hosting the app and making it publicly accessible.  

### **Tools** 🛠️  
- **Google Colab** 📚: Environment for training and evaluating the model.  
- **NumPy** ➕: For numerical computations and preprocessing.  
- **Pillow** 🖼️: For image processing in Python.  

### **Version Control** 📂  
- **Git** 🕵️: For managing and tracking changes to the project.  
- **GitHub** 🌐: Repository hosting service for version control.
 

## 📂 Dataset  

### **Description**
The dataset used in this project consists of chest X-ray images categorized into three classes:
1. **Lung Opacity**: X-rays showing opacity in lung regions.
2. **Normal**: Healthy lung X-rays with no abnormalities.
3. **Viral Pneumonia**: X-rays indicating the presence of viral pneumonia.

### **Source**
The dataset is typically organized in the following folder structure after extraction:
```
dataset/
│
├── Lung Opacity/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── Viral Pneumonia/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### **Preprocessing Steps**
1. **Normalize Pixel Values**: All pixel values are normalized to a range of `[0, 1]` by dividing by 255.
2. **Resize Images**: Each image is resized to `224x224` pixels to match the input size of the model.
3. **Augmentation (Optional)**: Techniques like rotation, flipping, and zooming are applied to increase dataset diversity (only during training).

### **Dataset Preparation Steps**
1. Download the dataset archive as a `.zip` file, named `archive.zip`.
2. Unzip the archive:
   ```bash
   unzip archive.zip -d dataset
   ```
3. Verify that the dataset is structured into separate folders for each class as shown above.
 

## 🤖 Model Training , Export and Deployment 

### **1. Model Training** 🧠

#### **Dataset Preparation** 📂  
- The dataset used is a collection of chest X-ray images categorized into three classes:  
  1. **Lung Opacity** 🫁  
  2. **Normal** ✅  
  3. **Viral Pneumonia** 🦠  
- The dataset was preprocessed by resizing the images to 224x224 pixels and normalizing pixel values to a range of [0, 1].

#### **Training Workflow** 🔄  
- A **Convolutional Neural Network (CNN)** was implemented using TensorFlow/Keras.  
- Data augmentation techniques such as rotation, flipping, and zooming were applied to enhance the model's robustness.  
- **Training Parameters**:  
  - Optimizer: Adam  
  - Loss Function: Categorical Crossentropy  
  - Metrics: Accuracy  
  - Epochs: 10 (adjustable based on performance)  
- The training process was carried out in **Google Colab**, leveraging GPU acceleration for faster computations.

---

### **2. Model Export** 📦  

#### **Steps to Save the Model**:  
1. After achieving satisfactory accuracy, the trained model was exported using the `.h5` format:  
   ```python
   model.save('model.h5')
   ```
2. This format ensures compatibility for reloading the model during deployment.  

#### **Verification**:  
- The saved model was reloaded to validate its integrity and compatibility:  
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('model.h5')
  ```

---

### **3. Deployment** 🌐  

#### **Platform**:  
- The application was deployed on **Hugging Face Spaces**, an easy-to-use platform for hosting machine learning models.

#### **Steps for Deployment**:  
1. **Prepare the App Code**:  
   - The app was built using **Gradio** for its interactive interface.  
   - Key functionalities included:  
     - Loading the model (`model.h5`).  
     - Processing uploaded images.  
     - Predicting the class of the image using the trained model.
2. **App file**:[app.py]( https://github.com/gowhar06/LungDisease_ML-/blob/main/app.py )
   - Example `app.py` code:  
     ```python
     import gradio as gr
     from tensorflow.keras.models import load_model
     from tensorflow.keras.preprocessing import image
     import numpy as np

     model = load_model('model.h5')

     def predict_image(img):
         img = img.resize((224, 224))
         img_array = np.array(img) / 255.0
         img_array = np.expand_dims(img_array, axis=0)
         prediction = model.predict(img_array)
         class_names = ['Lung Opacity', 'Normal', 'Viral Pneumonia']
         return class_names[np.argmax(prediction)]

     interface = gr.Interface(
         fn=predict_image,
         inputs=gr.Image(type="pil"),
         outputs="text",
         title="Lung Disease Detection",
         description="Upload a chest X-ray image to classify."
     )

     if __name__ == "__main__":
         interface.launch()
     ```

2. **Upload Files to Hugging Face**:  
   - Upload the following files:  
     - `app.py`  
     - `model.h5`  
     - `requirements.txt`  

3. **Create `requirements.txt`**:  
   - Include all dependencies required for the app:  
     ```plaintext
     gradio==3.36.0
     tensorflow
     numpy
     Pillow
     ```

4. **Deploy the Space**:  
   - Go to the Hugging Face Spaces dashboard and create a new Space.  
   - Choose **Gradio** as the framework.  
   - Upload the files and wait for the build process to complete.  

5. **Set the App to Public**:  
   - Enable public access to allow others to use your app.  

---

### **Usage** 🖥️  
- Access the app via the public URL provided by Hugging Face Spaces.  
- Upload a chest X-ray image to classify it as **Lung Opacity**, **Normal**, or **Viral Pneumonia**.

---

Let me know if you need further clarification or additional sections! 😊
By combining cutting-edge ML techniques with real-world datasets, this project offers a powerful tool for improving diagnostic workflows. 🌟


## 📜 License  

This project is licensed under the **MIT License**. For more details, refer to the [LICENSE](LICENSE) file.  


## 📞 Contact  

Feel free to reach out for any inquiries, suggestions, or collaboration opportunities:  
  - ✉️ **Email**: sshaikgowhar@example.com  
  - 🌐 **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/shaikgowhar672004/)   
