import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage, font
from PIL import Image, ImageTk
from ultralytics import YOLO
import sqlite3
from datetime import datetime

# Load YOLOv8 model
model_path = "/Users/yahya/Documents/fyp-roboflow/best_roboflow.pt"  # Path to your saved YOLOv8 model
model = YOLO(model_path)  # Loading custom YOLOv8 model

# Initialize OpenCV window
cap = None
is_image = False  # Flag to indicate if we are displaying an image
is_processing_image = False  # Flag to indicate if image processing is happening
frame_counter = 0  # To count the number of frames processed

# Database Functions
def create_database():
    # Connect to SQLite database (it will create the database if it doesn't exist)
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    
    # Create table to store the image predictions if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        class_name TEXT,
        confidence_score REAL,
        timestamp TEXT
    )
    ''')
    
    # Commit and close the connection
    conn.commit()
    conn.close()

def insert_into_database(image_name, class_name, confidence_score):
    # Debugging: print data to be inserted
    print(f"Inserting into database: {image_name}, {class_name}, {confidence_score}")

    # Ensure class_name is a string and confidence_score is a float
    class_name = str(class_name)  # Convert to string if it's not
    confidence_score = float(confidence_score)  # Ensure it's a float
    
    # Connect to SQLite database
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Insert data into the predictions table
    cursor.execute('''
    INSERT INTO predictions (image_name, class_name, confidence_score, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (image_name, class_name, confidence_score, timestamp))
    
    # Commit and close the connection
    conn.commit()
    conn.close()

# Function to open webcam
def open_webcam():
    global cap, is_image, is_processing_image
    is_image = False  # Set flag to False for video/webcam
    is_processing_image = False  # Reset image processing flag
    cap = cv2.VideoCapture(0)
    start_detection()

# Function to open video file
def open_video():
    global cap, is_image, is_processing_image
    is_image = False  # Set flag to False for video
    is_processing_image = False  # Reset image processing flag
    file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        start_detection()

# Function to upload and detect an image
def open_image():
    global is_image, is_processing_image
    is_image = True  # Set flag to True for image
    is_processing_image = True  # Flag image processing
    # Ask user to upload an image (accepting various formats)
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"))])
    if file_path:
        print(f"Image file selected: {file_path}")  # Debugging line
        img = cv2.imread(file_path)
        if img is None:
            print("Failed to load image")  # Debugging line
        else:
            start_detection_image(img, file_path)

# Function to start detection for video/webcam
def start_detection():
    global frame_counter
    while not is_image and not is_processing_image:  # Only run video detection if is_image is False
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)  # Get predictions

        # Check if results contain predictions
        if results:
            detections = results[0].boxes.xyxy  # Get bounding box coordinates (xyxy format)

            # Loop over detected objects
            for detection in detections:
                x1, y1, x2, y2 = detection.tolist()  # Extract bounding box coordinates

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Get class label and confidence score
                cls = int(results[0].boxes.cls[0])  # Extract class label
                conf = results[0].boxes.conf[0]  # Extract confidence score

                # Draw label (class name and confidence score)
                label = f'{model.names[cls]} {conf:.2f}'  # Model name and confidence score
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Insert into database every 10 frames
                frame_counter += 1
                if frame_counter % 10 == 0:
                    insert_into_database("webcam_or_video", str(model.names[cls]), conf)

        # Resize the frame to fit the canvas size (optional)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit the canvas size while maintaining the aspect ratio
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img)

        # Update image on tkinter window
        canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        window.update()

    cap.release()

# Function to start detection for image
def start_detection_image(img, file_path):
    print("Starting detection on the image...")  # Debugging line
    # Perform detection on the uploaded image
    results = model(img)  # Get predictions

    # Check if results contain predictions
    if results:
        print("Predictions found.")  # Debugging line
        detections = results[0].boxes.xyxy  # Get bounding box coordinates (xyxy format)

        # Loop over detected objects
        for detection in detections:
            x1, y1, x2, y2 = detection.tolist()  # Extract bounding box coordinates

            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Get class label and confidence score
            cls = int(results[0].boxes.cls[0])  # Extract class label
            conf = results[0].boxes.conf[0]  # Extract confidence score

            # Draw label (class name and confidence score)
            label = f'{model.names[cls]} {conf:.2f}'  # Model name and confidence score
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Insert data into the database
            insert_into_database(file_path, str(model.names[cls]), conf)  # Pass image name, class, and confidence

    # Convert processed image to PIL format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Resize image to fit canvas size
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_pil = img_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Update the canvas with the processed image
    canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
    
    # Keep the reference to the image to avoid it being garbage collected
    canvas.image = img_tk

# Function to stop webcam or video feed
def stop():
    if cap:
        cap.release()
    window.quit()

# Tkinter UI Setup
window = tk.Tk()
window.title("Traffic Sign Detection and Recognition")
window.geometry("800x600")
window.config(bg="#3D3D3D")  # offwhite background

# Set custom font
font_style = font.Font(family="Helvetica", size=12, weight="bold")

# Create canvas for video feed
canvas = tk.Canvas(window, width=800, height=600, bg="#E2DFD2")
canvas.pack()

# Define buttons with styling using pack() instead of grid()
btn_webcam = tk.Button(window, text="Open Webcam", command=open_webcam, font=font_style, padx=10, pady=5, relief="solid", bd=2, bg="#2980B9", fg="black")
btn_video = tk.Button(window, text="Open Video File", command=open_video, font=font_style, padx=10, pady=5, relief="solid", bd=2, bg="#2980B9", fg="black")
btn_image = tk.Button(window, text="Upload Image", command=open_image, font=font_style, padx=10, pady=5, relief="solid", bd=2, bg="#2980B9", fg="black")
btn_stop = tk.Button(window, text="Stop", command=stop, font=font_style, padx=10, pady=5, relief="solid", bd=2, bg="#E74C3C", fg="black")

# Arrange buttons using pack() method
btn_webcam.pack(pady=10)
btn_video.pack(pady=10)
btn_image.pack(pady=10)
btn_stop.pack(pady=10)

# Add hover effect to buttons
def on_enter(event):
    event.widget.config(bg="#1ABC9C")

def on_leave(event):
    event.widget.config(bg="#2980B9")

btn_webcam.bind("<Enter>", on_enter)
btn_webcam.bind("<Leave>", on_leave)

# Initialize the database
create_database()

# Start Tkinter main loop
window.mainloop()
