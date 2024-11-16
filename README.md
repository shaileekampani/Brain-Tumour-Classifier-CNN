# 🧠 Brain Tumour Detection with CNN & Flask

Welcome to the Brain Tumour Detection project! This project leverages a Convolutional Neural Network (CNN) to classify brain tumours based on MRI images, providing insights to support healthcare professionals. With an easy-to-use Flask interface, users can upload their own MRI images or select random samples from the dataset for classification and information.

---

## 📖 Index
1. [Introduction](#-introduction)
2. [Project Features](#-project-features)
3. [Technologies Used](#-technologies-used)
4. [Screenshots](#-screenshots)
5. [Run and Setup](#-run-and-setup)

---

## 🔍 Introduction
This project aims to assist healthcare professionals by automating the detection of brain tumuors using MRI images. Utilizing a Convolutional Neural Network (CNN) architecture, the model classifies images into three categories of tumors or identifies if no tumor is present. We also leveraged explainable AI techniques such as LIME in order for interpretability of the decisions made by the model and this was done using a pretrained model which is EfficientNet B1. The Flask app provides a user-friendly interface for uploading MRI scans and receiving classification results, offering essential insights and explanations.

---

## 🌟 Project Features
- **Upload MRI Images**: Users can upload their own MRI images for classification.
- **Random Image Selection**: Select random images from the dataset to test the model’s accuracy.
- **Classification and Explanation**: The model identifies the tumor type and gives additional information on the tumor class using an integrated AI assistant.
- **Future Integration of Explainable AI**: Future plans include integrating LIME into the app to provide visual explanations of model predictions.

---

## 💻 Technologies Used
- **Python**: The primary programming language for model development and Flask app.
- **Flask**: Used to create the web application interface.
- **Convolutional Neural Network (CNN)**: Model architecture for learning and classification.
- **EfficientNet B1**: Used for image feature extraction and to get insights using LIME.
- **OpenAI API**: Provides additional information on tumuor classification results.

---

## 🖼️ Screenshots
**Web Application Interface**

![Web App Image 1](https://github.com/user-attachments/assets/6ace08c8-cf21-4b53-ae7b-fa2a3ca3e206)
![Web App Image 2](https://github.com/user-attachments/assets/6c985668-54e8-474f-8576-fa02a8d21c1b)

---

## ⚙️ Run and Setup

### Setting Up the Flask Application
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ahmedrazagit/Brain-Tumour-Classifier-CNN.git
    ```

2. **Install Dependencies**:
    - Ensure all required Python packages are installed:
      ```bash
      pip install -r requirements.txt
      ```

### Running the Flask Application
1. **Start the Flask Server**:
    ```bash
    python app.py
    ```

2. **Access the Application**:
    - Open a web browser and go to `http://127.0.0.1:5000` (or the address displayed in your terminal) to view the application.

---

Enjoy exploring the project and feel free to contribute ideas for future improvements! 😊

