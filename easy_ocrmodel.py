import easyocr
import cv2
from PIL import Image
import numpy as np
import streamlit as st
import os
import time
import matplotlib.pyplot as plt

# Initialize EasyOCR  for CPU and GPU
reader_cpu = easyocr.Reader(['en'], gpu=False)  
reader_gpu = easyocr.Reader(['en'], gpu=True) 

st.title("Video OCR with CPU and GPU Comparison")

def extract_frames_from_video(video_path, frame_rate=1):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0
    success, frame = video_capture.read()

    while success:
        if frame_id % frame_rate == 0:  
            frames.append(frame)
        success, frame = video_capture.read()
        frame_id += 1

    video_capture.release()
    return frames

def ocr_on_frames(frames, reader):
    extracted_texts = []
    successful_extractions = 0
    total_extractions = 0
    start_time = time.time()

    for idx, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            # Perform OCR
            results = reader.readtext(np.array(pil_image))
            text = ' '.join([result[1] for result in results]).strip()  
        except Exception as e:
            st.error(f"Error during OCR processing: {e}")
            text = ""

        extracted_texts.append(text)

        if text:
            successful_extractions += 1
        total_extractions += 1

        st.image(pil_image, caption=f'Frame {idx} - Extracted Text: {text}', use_column_width=True)

    end_time = time.time()
    processing_time = end_time - start_time
    fps = total_extractions / processing_time

    return extracted_texts, successful_extractions, total_extractions, fps

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Video successfully uploaded!")

        st.text("Extracting frames from the video...")
        frames = extract_frames_from_video(file_path, frame_rate=30)

        st.text("Running OCR on CPU...")
        cpu_extracted_texts, cpu_successful_extractions, cpu_total_extractions, cpu_fps = ocr_on_frames(frames, reader_cpu)

        st.text("Running OCR on GPU...")
        gpu_extracted_texts, gpu_successful_extractions, gpu_total_extractions, gpu_fps = ocr_on_frames(frames, reader_gpu)

        cpu_accuracy = (cpu_successful_extractions / cpu_total_extractions) * 100 if cpu_total_extractions > 0 else 0
        gpu_accuracy = (gpu_successful_extractions / gpu_total_extractions) * 100 if gpu_total_extractions > 0 else 0

        st.write(f"**CPU Model Accuracy**: {cpu_accuracy:.2f}%, **FPS**: {cpu_fps:.2f}")
        st.write(f"**GPU Model Accuracy**: {gpu_accuracy:.2f}%, **FPS**: {gpu_fps:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.bar(['CPU', 'GPU'], [cpu_fps, gpu_fps], color=['blue', 'orange'])
        ax1.set_ylabel('Frames Per Second (FPS)')
        ax1.set_title('FPS Comparison: CPU vs GPU')

        ax2.bar(['CPU', 'GPU'], [cpu_accuracy, gpu_accuracy], color=['blue', 'orange'])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Comparison: CPU vs GPU')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
