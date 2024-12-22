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

### **Sample Dataset Code**
If you want to load and preprocess the dataset programmatically in Python:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataset_path = "dataset"
# ImageDataGenerator for loading and augmenting data
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    validation_split=0.2  # Split 20% of data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```
## **Next Steps**
1. **Model Training**: Train a CNN model using TensorFlow/Keras.
2. **Model Export**: Save the trained model as `model.h5`.
3. **Deployment**: Deploy the trained model using Gradio on Hugging Face Spaces.


## 🤖 Model Training and Evaluation  

- **Model**:  
  A **Convolutional Neural Network (CNN)** is employed to extract features and classify diseases.  

- **Metrics Used**:  
  - ✅ Accuracy  
  - 🏆 Precision  
  - 🔄 Recall  
  - 💡 F1-Score  
  - 📈 ROC-AUC  

- **Steps**:  
  1. Train the model on augmented and preprocessed datasets.  
  2. Evaluate its performance using a separate test dataset.  
  3. Fine-tune the model to improve results based on evaluation metrics.  

By combining cutting-edge ML techniques with real-world datasets, this project offers a powerful tool for improving diagnostic workflows. 🌟


## 📜 License  

This project is licensed under the **MIT License**. For more details, refer to the [LICENSE](LICENSE) file.  


## 📞 Contact  

Feel free to reach out for any inquiries, suggestions, or collaboration opportunities:  
  - ✉️ **Email**: sshaikgowhar@example.com  
  - 🌐 **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/shaikgowhar672004/)   
