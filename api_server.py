import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
import face_recognition
import shutil

# Initialize the Flask application
app = Flask(__name__)

# --- Global variables to hold face data ---
KNOWN_ENCODINGS = []
KNOWN_NAMES = []

# --- Path Configuration ---
CSHARP_OFFICIAL_PHOTOS_PATH = 'F:/AMS_FaceScan/studentsphotos/'
CSHARP_TRAINING_PHOTOS_PATH = 'F:/AMS_FaceScan/TrainingImages/'
AI_IMAGES_PATH = './AI_Images/'


# --- Sync images (combine sync_images.py logic) ---
def sync_all_images():
    """Sync photos from official and training folders to AI folder"""
    # Clean the old AI folder
    if os.path.exists(AI_IMAGES_PATH):
        shutil.rmtree(AI_IMAGES_PATH)
    os.makedirs(AI_IMAGES_PATH)

    copied_count = 0

    # Copy official photos
    print(f"Copying official photos from {CSHARP_OFFICIAL_PHOTOS_PATH}...")
    for filename in os.listdir(CSHARP_OFFICIAL_PHOTOS_PATH):
        source_file = os.path.join(CSHARP_OFFICIAL_PHOTOS_PATH, filename)
        destination_file = os.path.join(AI_IMAGES_PATH, filename)
        shutil.copy2(source_file, destination_file)
        copied_count += 1

    # Copy training photos
    print(f"Copying training photos from {CSHARP_TRAINING_PHOTOS_PATH}...")
    for filename in os.listdir(CSHARP_TRAINING_PHOTOS_PATH):
        source_file = os.path.join(CSHARP_TRAINING_PHOTOS_PATH, filename)
        destination_file = os.path.join(AI_IMAGES_PATH, filename)
        shutil.copy2(source_file, destination_file)
        copied_count += 1

    print(f"Sync complete. Total {copied_count} images copied to {AI_IMAGES_PATH}")


# --- Face Data Loading Function ---
def load_known_faces():
    """
    Loads all images from the AI_Images folder, calculates their encodings,
    and stores them in the global lists. This runs once at startup.
    """
    global KNOWN_ENCODINGS, KNOWN_NAMES

    if not os.path.exists(AI_IMAGES_PATH):
        print(f"FATAL ERROR: The data folder '{AI_IMAGES_PATH}' was not found!")
        return

    KNOWN_ENCODINGS.clear()
    KNOWN_NAMES.clear()

    image_files = [f for f in os.listdir(AI_IMAGES_PATH) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Loading {len(image_files)} total images (official + training)...")

    for filename in image_files:
        try:
            image_path = os.path.join(AI_IMAGES_PATH, filename)
            current_img = face_recognition.load_image_file(image_path)
            encodes = face_recognition.face_encodings(current_img)

            if encodes:
                KNOWN_ENCODINGS.append(encodes[0])
                base_name = os.path.splitext(filename)[0]
                roll_no = base_name.split('_')[0]
                KNOWN_NAMES.append(roll_no.lower())  # Store names in lowercase
        except Exception as e:
            print(f"Warning: Could not process image {filename}. Reason: {e}")

    print(f"--- Face loading complete. {len(KNOWN_ENCODINGS)} faces loaded successfully. ---")


# --- API Endpoints ---

@app.route('/validate_and_process_photo', methods=['POST'])
def validate_and_process_photo():
    """
    Endpoint for student profile page: Verifies if an uploaded photo matches
    the student's official photo (1:1 comparison).
    """
    json_data = request.get_json()
    if not json_data or 'image_base64' not in json_data or 'rollNo' not in json_data:
        return jsonify({"status": "failure", "message": "Missing required data (image or rollNo)."}), 400

    uploader_roll_no = json_data['rollNo'].lower()  # Convert incoming rollNo to lowercase

    if uploader_roll_no not in KNOWN_NAMES:
        return jsonify({"status": "failure", "message": "Your face does not match."})

    try:
        image_b64_string = json_data['image_base64']
        if ',' in image_b64_string:
            image_b64_clean = image_b64_string.split(',')[1]
        else:
            image_b64_clean = image_b64_string

        img_data = base64.b64decode(image_b64_clean)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Invalid image data or format. Details: {e}"})

    face_locations = face_recognition.face_locations(img)
    if len(face_locations) != 1:
        return jsonify({"status": "failure", "message": "Please upload a clear photo with ONE face."})

    try:
        new_face_encoding = face_recognition.face_encodings(img, face_locations)[0]

        # Find the first official encoding for the student for comparison
        official_index = KNOWN_NAMES.index(uploader_roll_no)
        official_encoding = KNOWN_ENCODINGS[official_index]

        # Compare faces with tolerance for minor discrepancies
        matches = face_recognition.compare_faces([official_encoding], new_face_encoding, tolerance=0.6)
        is_match = matches[0]

        if is_match:
            return jsonify({"status": "success", "message": "Identity verified. New training photo accepted."})
        else:
            return jsonify(
                {"status": "failure", "message": "Your face does not match. Please upload a clearer image."})
    except Exception as e:
        return jsonify({"status": "failure", "message": f"An error occurred during face comparison: {e}"}), 500


@app.route('/recognize_attendance', methods=['POST'])
def recognize_attendance():
    """
    Endpoint for real-time attendance page: Identifies all known faces
    in a video frame (1: N comparison).
    """
    json_data = request.get_json()
    if not json_data or 'image_base64' not in json_data:
        return jsonify({"error": "No image data sent."}), 400

    try:
        image_b64_string = json_data['image_base64']
        if ',' in image_b64_string:
            image_b64_clean = image_b64_string.split(',')[1]
        else:
            image_b64_clean = image_b64_string

        img_data = base64.b64decode(image_b64_clean)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("Could not decode image frame")
    except Exception as e:
        return jsonify({"error": f"Invalid image data. Details: {e}"})

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    recognized_roll_numbers = set()
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding, tolerance=0.6)
        if True in matches:
            face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = KNOWN_NAMES[best_match_index]
                recognized_roll_numbers.add(roll_no)


    return jsonify({"recognized_rolls": list(recognized_roll_numbers)})


# --- Main Execution Block ---
if __name__ == '__main__':
    # Sync images first
    sync_all_images()

    # Load faces from the AI_Images folder
    load_known_faces()

    print("Starting Flask server... Ready to accept requests from C# application.")
    app.run(host='0.0.0.0', port=5000, debug=False)
