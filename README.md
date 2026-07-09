# Final_Year_Project
Alzheimer's Disease Stage Classification using VGG16 is a deep learning project that analyzes brain MRI images to classify Alzheimer's disease into different stages. It uses transfer learning with the VGG16 model to achieve accurate classification, supporting early detection and improving diagnostic efficiency.
# Alzheimer's Disease Stage Classification using Deep Learning

## Project Overview
This project focuses on the automatic classification of Alzheimer's disease stages using Brain MRI images and Deep Learning. The model is built using the VGG16 Convolutional Neural Network (CNN) with transfer learning to accurately classify different stages of Alzheimer's disease, helping support early diagnosis and clinical decision-making.

---

## Problem Statement
Alzheimer's disease is a progressive neurological disorder that affects memory and cognitive functions. Early diagnosis is difficult through manual analysis of MRI scans. This project aims to automate the classification process using deep learning techniques.

---

## Objectives
- Detect Alzheimer's disease from Brain MRI images.
- Classify MRI images into different Alzheimer's stages.
- Improve diagnostic accuracy using transfer learning.
- Assist healthcare professionals with faster predictions.

---

## Technologies Used
- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- VGG16 (Transfer Learning)

---

## Dataset
The project uses Brain MRI images containing multiple Alzheimer's disease stages.

Classes:
- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

---

## Deep Learning Model
- Pre-trained VGG16
- Transfer Learning
- CNN Classifier
- Softmax Output Layer

---

## Features
- MRI Image Classification
- Data Preprocessing
- Image Augmentation
- Transfer Learning
- Model Training
- Prediction on New MRI Images
- Accuracy and Loss Visualization

---

## Project Structure

```
Alzheimers-Disease-Stage-Classification/
│
├── dataset/
├── models/
├── notebooks/
├── images/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── Project_Report.pdf
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/vishwaaradhyasm2003/Final_Year_Project.git
```

Go to project folder

```bash
cd Final_Year_Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## Results

The model successfully classifies MRI images into different Alzheimer's disease stages using the VGG16 deep learning architecture. It achieves high classification performance and demonstrates the effectiveness of transfer learning for medical image analysis.

---

## Future Improvements

- Increase dataset size
- Improve model accuracy
- Deploy as a web application
- Add explainable AI (Grad-CAM)
- Support real-time predictions

---

## Author
**1.Sunitha Seervi, **
**2.Vishwa Aradhya S M, **
**3.Ramya M, **
**4.Sharan Kumar**

Information Science and Engineering

GitHub: https://github.com/vishwaaradhyasm2003

---

## License

This project is developed for educational and research purposes.
