import os
import shutil
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('/Users/apple/Desktop/Detection_folder/Yolov8/runs/detect/train2/weights/best.pt')

# Function to track objects in a video
def track_objects(video_path, output_dir):
    result = model.track(video_path, save=True, save_dir=output_dir)
    # Find the latest track folder
    latest_track_folder = max([os.path.join(output_dir, d) for d in os.listdir(output_dir)], key=os.path.getmtime)
    tracked_video_path = os.path.join(latest_track_folder, "temp_video.mp4")
    print(latest_track_folder)
    return tracked_video_path

# Function for area quantification
def quantify_area(original_image_path, mask_image_path):
    # Load the original image and mask
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is binary (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Assuming white pixels represent the lesion, count them
    white_pixels = np.sum(binary_mask == 255)

    # Hypothetical scale factor: Assuming each pixel represents 0.25 square units
    scale_factor_per_pixel = 0.25
    real_area = white_pixels * scale_factor_per_pixel

    # Draw area on the image
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :] = [255, 0, 0]  # BGR format, red
    alpha = 0.4  # Transparency factor
    colored_mask_applied = np.where(binary_mask[:, :, None] == 255, (colored_mask * alpha).astype(np.uint8), 0)
    highlighted_image = cv2.add(original_image, colored_mask_applied)

    # Draw area text
    area_text = f'Area: {real_area} sq units'
    font_scale = 0.7
    font_thickness = 2
    text_width, text_height = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x, text_y = 10, text_height + 20
    cv2.putText(highlighted_image, area_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    return highlighted_image

# Streamlit UI
st.title('Object Tracking')

# Video tracking section
st.header('Video Tracking')
video_file = st.file_uploader("Upload a video", type=['mp4'])

if video_file is not None:
    st.video(video_file)
    if st.button('Track Objects'):
        with st.spinner('Tracking objects...'):
            # Save uploaded video to a temporary location
            video_path = "temp_video.mp4"
            output_dir = "runs/detect"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            tracked_video_path = track_objects(video_path, output_dir)
            if tracked_video_path:
                st.video(tracked_video_path)
            else:
                st.error("Error occurred while tracking objects.")

# Area quantification section
st.header('Area Quantification')
original_image_file = st.file_uploader("Upload the original image", type=['jpg', 'png'])
mask_image_file = st.file_uploader("Upload the corresponding mask image", type=['jpg', 'png'])

if original_image_file is not None and mask_image_file is not None:
    st.image([original_image_file, mask_image_file], caption=['Original Image', 'Mask Image'], width=300)

    if st.button('Quantify Area'):
        with st.spinner('Quantifying area...'):
            original_image = Image.open(original_image_file)
            mask_image = Image.open(mask_image_file)
            original_image_path = "temp_original.jpg"
            mask_image_path = "temp_mask.jpg"
            original_image.save(original_image_path)
            mask_image.save(mask_image_path)
            result_image = quantify_area(original_image_path, mask_image_path)
            st.image(result_image, caption='Area Quantification Result', use_column_width=True)

# Cleanup temporary files
if os.path.exists("temp_video.mp4"):
    os.remove("temp_video.mp4")
if os.path.exists("temp_original.jpg"):
    os.remove("temp_original.jpg")
if os.path.exists("temp_mask.jpg"):
    os.remove("temp_mask.jpg")