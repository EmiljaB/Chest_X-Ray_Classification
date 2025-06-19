# Chest_X-Ray_Classification
Identifying Lung Diseases from Chest X-Rays

Kaggle Link: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis
<div align="center">
  <img src="https://github.com/user-attachments/assets/9209983a-3150-4308-a922-402ceb817b35" alt="Chest" width="400"/>
</div>

# Project Description
This project focuses on building an AI-powered diagnostic system for classifying chest X-ray images into four categories: COVID-19, Pneumonia, Tuberculosis, or Normal. We developed and trained multiple models, including a custom CNN, a ResNet50 with class-balanced training, and a hybrid CNN + Vision Transformer (ViT) model to enhance feature extraction and diagnostic accuracy.

To make the model accessible and user-friendly, we also created a simple Flask web application that allows users to upload a chest X-ray image and receive an instant diagnosis. The interface is designed for ease of use by healthcare professionals or researchers and provides a lightweight frontend connected to our trained models.

## Web App

This repository includes a Flask web app for running the Chest X-ray Classifier with a clean user interface.
This end-to-end pipeline demonstrates how deep learning and transformer-based architectures can be effectively combined with web technologies to support real-time medical image analysis.

### How to run locally: The app will be available at http://127.0.0.1:5000
```bash
cd chest_xray_webapp
pip install -r requirements.txt
python app.py

