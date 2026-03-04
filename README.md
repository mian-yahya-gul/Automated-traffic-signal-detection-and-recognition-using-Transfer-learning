Automated Traffic Sign Detection and Recognition (Final Version)

Project Overview

This project, part of my Final Year Project (FYP), focuses on the development of an Automated Traffic Sign Detection and Recognition System using Deep Learning techniques. The aim is to build a fully functional application that can recognize and classify traffic signs in real    time.

The final version of the system integrates a trained deep learning model (such as YOLO) for recognizing traffic signs with high accuracy. The system is designed for use in autonomous vehicles and smart traffic management systems.

Features in Final Version


Real Time Traffic Sign Recognition: Integration of a trained deep learning model (YOLO) to recognize traffic signs accurately.

Image Upload: Users can upload images of traffic signs for recognition.

Prediction Display: Displays the predicted traffic sign along with confidence levels.

Real Time Video Stream Processing: Detects and recognizes traffic signs in live video streams (for integration in autonomous vehicles).

Database Integration: Stores user information and recognized sign data.

Enhanced UI: Modern, userfriendly interface using Tkinter

Cross Platform Deployment: Plans to develop a mobile app using Kivy or Flutter.

Technologies Used

Python 3.10+

YOLOv8: Pre trained deep learning model for traffic sign recognition.

SQLite3: For user data storage and management.

Tkinter: For building the graphical user interface (UI).

OpenCV: For real-time image and video processing.

PIL (Pillow): For image preprocessing (resizing, normalization).


Project Structure

FYP-Traffic-Sign-Recognition
│
├── Detections.db         # Database file
├── best_roboflow.pt      # Trained Model
├── app.py                # Main application for UI handling and flow
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation


Final Version Workflow




Upload Image:

The user uploads an image of a traffic sign through the UI.

Traffic Sign Recognition:

The image is passed through the trained deep learning model (YOLO).

The model processes the image and predicts the traffic sign.

Prediction Display:

The system displays the predicted traffic sign label along with the confidence score.

Real Time Recognition :

If using a video stream, the system processes each frame in real time and recognizes traffic signs live.

Database Logging:

The system logs the prediction data (e.g., image path, predicted sign, confidence) into the database for future analysis or feedback.

Future Work

Model Improvement:

Fine tune the model using more complex datasets like LISA, GTSRB, or custom datasets.

Explore advanced models (e.g., ResNet, VGG) for improved accuracy.

Real Time Video Stream:

Improve video stream processing for efficient real time recognition, specifically for autonomous vehicle integration.

Cross Platform Deployment:

Develop a mobile app version using Kivy or Flutter for wider user accessibility.

Deploy the application as a web service using Flask or FastAPI for cloud based recognition.

Enhanced User Interface:

Design a more interactive and modern UI, possibly with PyQt for desktop apps or a mobile interface for Android.

Supervisor

Name: Zaid Ismail

Email: zaid.ismail@vu.edu.pk

Author

Name: Yahya

Reg #: BC230415847

Email: kakakhel49@yahoo.com