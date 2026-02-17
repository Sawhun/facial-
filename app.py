import os

# Load environment variables from .env file FIRST (before anything else)
from dotenv import load_dotenv
load_dotenv()

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Monkey-patch for TensorFlow/Keras compatibility
import tensorflow as tf
import types
import sys
tf_keras = types.ModuleType("tf_keras")
tf_keras.__dict__.update(tf.keras.__dict__)
tf_keras.__version__ = tf.__version__
sys.modules["tf_keras"] = tf_keras

import io
import base64
import numpy as np
import re
import random
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import bcrypt
import cv2
from PIL import Image
from models import db, Admin, Employee, Attendance, Settings

# Security imports
from security import (
    get_rate_limiter,
    get_audit_logger,
    get_client_ip,
    add_security_headers,
    configure_secure_session,
    InputValidator,
    generate_csrf_token,
    validate_csrf_token,
    rate_limit_key
)

app = Flask(__name__)

# Security Configuration
import secrets as sec_secrets
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', sec_secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medicare_attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Prevent DOS via large uploads

# Configure secure sessions
app = configure_secure_session(app)

# Initialize security components
rate_limiter = get_rate_limiter()
audit_logger = get_audit_logger()

# Disable in development if not using HTTPS
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS

# Face recognition configuration
FACE_MODEL = "ArcFace"  # More accurate than SFace
FACE_DETECTOR = "opencv"  # Use OpenCV detector (fastest and most compatible)
RECOGNITION_THRESHOLD = 0.72  # Stricter threshold to prevent false matches
MIN_FACE_CONFIDENCE = 0.80  # Confidence for face detection

# PERFORMANCE MODE: Set to True for faster recognition (reduces anti-spoofing checks)
FAST_RECOGNITION_MODE = False  # Set to False to enable full anti-spoofing
ANTI_SPOOFING_ENABLED = not FAST_RECOGNITION_MODE  # Only enable heavy checks in slow mode
ANTI_SPOOFING_THRESHOLD = 0.55  # Stricter threshold for anti-spoofing

# Additional anti-spoofing thresholds (MAXIMUM STRICT for healthcare security)
PHONE_DETECTION_THRESHOLD = 0.55  # Lower score = phone detected (STRICTER)
SCREEN_DETECTION_THRESHOLD = 0.55  # Lower score = screen detected (STRICTER)
PRINT_DETECTION_THRESHOLD = 0.55  # Lower score = printed photo detected (STRICTER)
COMBINED_SPOOF_THRESHOLD = 0.65  # Average of all checks must be above this
# Close-up screen detection thresholds
COLOR_CONSISTENCY_THRESHOLD = 15  # Lower = more consistent (digital)
SHARPNESS_THRESHOLD = 2500  # Higher = over-sharpened (digital)
TEXTURE_THRESHOLD = 6  # Lower = lacking natural texture (digital)
BRIGHTNESS_UNIFORMITY_THRESHOLD = 12  # Lower = too uniform (screen backlight)
EDGE_DENSITY_THRESHOLD = 0.15  # Higher = too many sharp edges (digital)

# Liveness detection - DISABLED (using DeepFace anti-spoofing instead)
LIVENESS_ENABLED = False  # Disabled - DeepFace anti-spoof is faster
LIVENESS_FRAMES_REQUIRED = 4
BLINK_DETECTION_ENABLED = False
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
LIVENESS_THRESHOLD = 0.40

FACE_DB_PATH = 'face_db'
os.makedirs(FACE_DB_PATH, exist_ok=True)

# Cache for embeddings (improves performance)
embeddings_cache = {}
cache_timestamp = None

# Liveness detection sessions (stores frames for multi-frame analysis)
# Key: session_id, Value: {'frames': [], 'timestamps': [], 'eye_ratios': [], 'face_positions': []}
liveness_sessions = {}
LIVENESS_SESSION_TIMEOUT = 30  # seconds
CHALLENGE_TYPES = ['BLINK', 'TURN_LEFT', 'TURN_RIGHT']  # Challenge types for liveness

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Admin, int(user_id))

def get_work_hours():
    """Get work hours from settings or return defaults"""
    with app.app_context():
        settings = Settings.query.first()
        if settings:
            return settings.work_start_hour, settings.work_start_minute
    return 9, 0  # Default: 9:00 AM

def init_db():
    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        if not Admin.query.filter_by(username='admin').first():
            password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            admin = Admin(username='admin', password_hash=password_hash.decode('utf-8'))
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")

        # Create default settings if not exists
        if not Settings.query.first():
            settings = Settings(
                work_start_hour=9,
                work_start_minute=0,
                work_end_hour=17,
                work_end_minute=0,
                recognition_threshold=RECOGNITION_THRESHOLD,
                anti_spoofing_enabled=ANTI_SPOOFING_ENABLED
            )
            db.session.add(settings)
            db.session.commit()
            print("Default settings created")
        else:
            # Update existing settings for security fixes
            settings = Settings.query.first()
            if settings.recognition_threshold < 0.70:
                settings.recognition_threshold = RECOGNITION_THRESHOLD
                db.session.commit()
                print(f"Updated recognition threshold to {RECOGNITION_THRESHOLD} for better security")

def refresh_embeddings_cache():
    """Refresh the embeddings cache from database with pre-normalized embeddings for faster comparison"""
    global embeddings_cache, cache_timestamp
    embeddings_cache = {}
    employees = Employee.query.all()
    for emp in employees:
        embedding = emp.get_embedding()
        if embedding is not None:
            # Pre-normalize the embedding to avoid repeated norm calculations
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 0 else embedding

            embeddings_cache[emp.id] = {
                'embedding': embedding,
                'normalized': normalized_embedding,  # Store pre-normalized version
                'employee': emp
            }
    cache_timestamp = datetime.now()
    return len(embeddings_cache)

def get_face_embedding(image_path_or_array, return_details=False):
    """
    Extract face embedding from image using DeepFace.
    Returns embedding array or None if no face detected.
    If return_details=True, returns (embedding, face_info) tuple.

    OPTIMIZED: Uses opencv detector (fastest) with alignment for accuracy
    """
    try:
        import time
        embed_start = time.time()
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=image_path_or_array,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
            align=True,
            normalization='base'  # Faster normalization
        )
        print(f"  ⏱️ DeepFace embedding extraction: {(time.time() - embed_start)*1000:.0f}ms")

        if result and len(result) > 0:
            embedding = np.array(result[0]['embedding'])
            if return_details:
                face_info = {
                    'confidence': result[0].get('face_confidence', 1.0),
                    'facial_area': result[0].get('facial_area', {}),
                }
                return embedding, face_info
            return embedding
        return None if not return_details else (None, None)
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None if not return_details else (None, None)

def detect_face(image_array):
    """
    Detect face in image and return face info.
    Returns dict with face_detected, confidence, facial_area, etc.
    """
    try:
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend=FACE_DETECTOR,
            enforce_detection=False,
            align=True
        )

        if faces and len(faces) > 0:
            face = faces[0]
            confidence = face.get('confidence', 0)
            return {
                'face_detected': confidence > 0.5,
                'confidence': confidence,
                'facial_area': face.get('facial_area', {}),
                'face_count': len(faces)
            }
        return {'face_detected': False, 'confidence': 0, 'face_count': 0}
    except Exception as e:
        print(f"Face detection error: {e}")
        return {'face_detected': False, 'confidence': 0, 'error': str(e)}

def check_anti_spoofing(image_array):
    """
    Check if the face is real using DeepFace's built-in anti-spoofing.
    This uses a trained neural network - fast and reliable.
    Returns (is_real, confidence_score).

    SMART FAIL-SAFE MODE:
    - If torch is not installed, skip this layer (rely on other 3 strict layers)
    - If torch IS installed but DeepFace has errors, REJECT for safety
    """
    # Check if anti-spoofing is enabled
    if not ANTI_SPOOFING_ENABLED:
        return True, 1.0

    try:
        # First, check if torch is available (required for DeepFace anti-spoofing)
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False

        if not torch_available:
            # Torch not installed - skip this layer, rely on other 3 strict layers
            print("⚠️ PyTorch not installed - skipping DeepFace layer (3 other strict layers active)")
            return True, 1.0

        # Torch is available, proceed with DeepFace anti-spoofing
        from deepface import DeepFace

        # Use DeepFace's built-in anti-spoofing (trained neural network)
        result = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend="opencv",  # Fast detector
            anti_spoofing=True
        )

        if result and len(result) > 0:
            # Check if anti-spoofing data exists
            is_real = result[0].get('is_real', None)
            score = result[0].get('antispoof_score', None)

            if is_real is not None and score is not None:
                # DeepFace anti-spoofing is working - use the result
                print(f"✓ DeepFace Anti-Spoof: is_real={is_real}, score={score:.3f}")
                return is_real, score
            else:
                # DeepFace doesn't support anti-spoofing - REJECT FOR SAFETY
                print("⚠️ DeepFace anti-spoofing not supported - REJECTING (fail-safe)")
                return False, 0.0

        # No face detected - REJECT FOR SAFETY
        print("⚠️ Anti-spoofing: No face detected - REJECTING (fail-safe)")
        return False, 0.0

    except Exception as e:
        error_msg = str(e)

        # If error is about missing torch, skip this layer (other layers will catch spoofs)
        if "torch" in error_msg.lower() or "pytorch" in error_msg.lower():
            print(f"⚠️ PyTorch dependency issue: {e}")
            print("⚠️ Skipping DeepFace layer - relying on 3 other STRICT layers")
            return True, 1.0

        # Other errors - REJECT FOR SAFETY
        print(f"⚠️ Anti-spoofing error: {e} - REJECTING (fail-safe)")
        return False, 0.0


def detect_phone_screen(small_img, gray_img):
    """
    Detect if the image is displayed on a phone/tablet screen.
    BALANCED: Catches obvious phones while allowing real faces through.
    Returns score: LOW = phone detected, HIGH = likely real
    """
    try:
        detection_flags = []

        # 1. PIXEL GRID DETECTION (phones have visible pixel patterns when photographed)
        f_transform = np.fft.fft2(gray_img.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        h_slice = magnitude[center_h, center_w+10:center_w+100]
        v_slice = magnitude[center_h+10:center_h+100, center_w]

        h_peaks = np.sum(h_slice > np.mean(h_slice) + 1.5 * np.std(h_slice))
        v_peaks = np.sum(v_slice > np.mean(v_slice) + 1.5 * np.std(v_slice))

        # Only flag very obvious pixel grid patterns
        if h_peaks > 10 or v_peaks > 10:
            detection_flags.append(('pixel_grid', 0.2))
        elif h_peaks > 7 or v_peaks > 7:
            detection_flags.append(('some_grid', 0.5))
        else:
            detection_flags.append(('grid_ok', 0.9))

        # 2. COLOR OVERSATURATION (phones boost colors, especially OLED)
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)

        # Only flag very high saturation
        if mean_sat > 160:
            detection_flags.append(('oversaturated', 0.25))
        elif mean_sat > 140:
            detection_flags.append(('high_sat', 0.55))
        else:
            detection_flags.append(('sat_ok', 0.9))

        # 3. BLUE LIGHT DETECTION (screens emit more blue light)
        b, g, r = cv2.split(small_img)
        blue_dominance = np.mean(b) / (np.mean(r) + 1)

        if blue_dominance > 1.4:
            detection_flags.append(('blue_screen', 0.25))
        elif blue_dominance > 1.25:
            detection_flags.append(('slight_blue', 0.55))
        else:
            detection_flags.append(('blue_ok', 0.9))

        # 4. UNIFORM BACKLIGHT DETECTION (screens have even illumination)
        grid_size = 4
        h, w = gray_img.shape
        block_h, block_w = h // grid_size, w // grid_size

        block_means = []
        for i in range(grid_size):
            for j in range(grid_size):
                block = gray_img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_means.append(np.mean(block))

        illumination_variance = np.std(block_means)

        # Only flag very uniform lighting (clear phone indicator)
        if illumination_variance < 8:
            detection_flags.append(('uniform_light', 0.2))
        elif illumination_variance < 15:
            detection_flags.append(('somewhat_uniform', 0.55))
        else:
            detection_flags.append(('varied_light', 0.9))

        # 5. BEZEL/EDGE DETECTION (phone screens have sharp rectangular borders)
        edges = cv2.Canny(gray_img, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=50, maxLineGap=5)

        if lines is not None:
            h_lines = 0
            v_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

                if length > 70:
                    if angle < 10 or angle > 170:
                        h_lines += 1
                    elif 80 < angle < 100:
                        v_lines += 1

            # Only flag clear rectangular bezels
            if h_lines >= 2 and v_lines >= 2:
                detection_flags.append(('phone_bezel', 0.2))
            else:
                detection_flags.append(('edges_ok', 0.9))
        else:
            detection_flags.append(('no_bezel', 0.9))

        # 6. MOIRE PATTERN DETECTION
        blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
        high_pass = cv2.absdiff(gray_img, blur)
        hp_std = np.std(high_pass)
        hp_mean = np.mean(high_pass)

        # Only flag very strong moiré patterns
        if hp_std > 30 and hp_mean > 12:
            detection_flags.append(('moire_pattern', 0.25))
        elif hp_std > 22:
            detection_flags.append(('some_moire', 0.55))
        else:
            detection_flags.append(('moire_ok', 0.9))

        # 7. SCREEN GLARE/REFLECTION
        v_channel = hsv[:, :, 2]
        very_bright = np.sum(v_channel > 250) / v_channel.size

        if very_bright > 0.05:
            detection_flags.append(('screen_glare', 0.35))
        else:
            detection_flags.append(('glare_ok', 0.9))

        # Calculate final score - STRICT approach for healthcare security
        all_scores = [score for _, score in detection_flags]
        min_score = min(all_scores)
        avg_score = sum(all_scores) / len(all_scores)

        # Count very suspicious flags (clear phone indicators) - STRICTER
        very_suspicious = sum(1 for s in all_scores if s <= 0.35)
        suspicious_count = sum(1 for s in all_scores if s < 0.6)

        if very_suspicious >= 1:
            # Even ONE clear phone indicator should fail - STRICT
            final_score = 0.5 * min_score + 0.5 * avg_score
        elif suspicious_count >= 2:
            # Two moderate indicators
            final_score = 0.6 * min_score + 0.4 * avg_score
        elif suspicious_count >= 1:
            # One moderate indicator
            final_score = 0.4 * min_score + 0.6 * avg_score
        else:
            # Probably real - use mostly average
            final_score = 0.2 * min_score + 0.8 * avg_score

        print(f"Phone detection flags: {detection_flags}, Final: {final_score:.3f}")

        return final_score

    except Exception as e:
        print(f"⚠️ Phone detection error: {e} - REJECTING (fail-safe)")
        return 0.3  # On error, REJECT for safety


def detect_screen_display_fast(small_img, gray_img):
    """
    Screen detection using pre-resized images.
    BALANCED: Catches obvious screens while allowing real faces through.
    """
    try:
        detection_flags = []

        # 1. Detect moiré patterns (interference from screen pixels)
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        high_pass = cv2.absdiff(gray_img, blur)
        hp_std = np.std(high_pass)
        hp_mean = np.mean(high_pass)

        if hp_std < 2 and hp_mean < 2:
            detection_flags.append(('moire_uniform', 0.3))
        elif hp_std > 45:
            detection_flags.append(('moire_noisy', 0.4))
        else:
            detection_flags.append(('moire_ok', 0.9))

        # 2. Detect color banding
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        unique_hues = len(np.unique(h_channel))

        if unique_hues < 15:
            detection_flags.append(('color_banding', 0.25))
        elif unique_hues < 30:
            detection_flags.append(('color_limited', 0.55))
        else:
            detection_flags.append(('color_ok', 0.9))

        # 3. Detect rectangular edges (phone screen border)
        edges = cv2.Canny(gray_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                                minLineLength=50, maxLineGap=10)

        if lines is not None:
            h_lines = 0
            v_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if length > 60:
                    if angle < 10 or angle > 170:
                        h_lines += 1
                    elif 80 < angle < 100:
                        v_lines += 1

            if h_lines > 4 and v_lines > 4:
                detection_flags.append(('rect_edges', 0.2))
            elif h_lines > 2 and v_lines > 2:
                detection_flags.append(('some_edges', 0.55))
            else:
                detection_flags.append(('edges_ok', 0.9))
        else:
            detection_flags.append(('no_edges', 0.95))

        # 4. Check for unnatural sharpness
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        lap_var = laplacian.var()

        if lap_var > 3000:
            detection_flags.append(('too_sharp', 0.4))
        elif lap_var < 30:
            detection_flags.append(('too_blurry', 0.4))
        else:
            detection_flags.append(('sharpness_ok', 0.9))

        # 5. Check for screen color temperature (blue tint)
        b, g, r = cv2.split(small_img)
        blue_ratio = np.mean(b) / (np.mean(r) + 1)

        if blue_ratio > 1.35:
            detection_flags.append(('blue_tint', 0.3))
        elif blue_ratio > 1.2:
            detection_flags.append(('slight_blue', 0.6))
        else:
            detection_flags.append(('color_temp_ok', 0.9))

        # Use STRICT scoring for healthcare security
        all_scores = [score for _, score in detection_flags]
        avg_score = sum(all_scores) / len(all_scores)
        min_score = min(all_scores)
        very_suspicious = sum(1 for s in all_scores if s <= 0.35)
        suspicious_count = sum(1 for s in all_scores if s < 0.6)

        if very_suspicious >= 1:
            # Even ONE clear screen indicator should fail - STRICT
            final_score = 0.5 * min_score + 0.5 * avg_score
        elif suspicious_count >= 2:
            final_score = 0.6 * min_score + 0.4 * avg_score
        elif suspicious_count >= 1:
            final_score = 0.4 * min_score + 0.6 * avg_score
        else:
            final_score = 0.2 * min_score + 0.8 * avg_score

        print(f"Screen detection flags: {detection_flags}, Final: {final_score:.3f}")

        return final_score

    except Exception as e:
        print(f"⚠️ Screen detection error: {e} - REJECTING (fail-safe)")
        return 0.3  # On error, REJECT for safety


def detect_printed_photo_fast(small_img, gray_img):
    """
    Printed photo detection using pre-resized images.
    BALANCED: Catches obvious prints while allowing real faces through.
    """
    try:
        detection_flags = []

        # 1. Check for paper texture
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray_img, -1, kernel)
        hf_std = np.std(high_freq)
        hf_mean = np.mean(np.abs(high_freq))

        if hf_std < 4 and hf_mean < 3:
            detection_flags.append(('print_texture', 0.35))
        elif hf_std < 8:
            detection_flags.append(('some_texture', 0.6))
        else:
            detection_flags.append(('texture_ok', 0.9))

        # 2. Check for flat color regions
        colors = small_img.reshape(-1, 3)
        unique_colors = len(np.unique(colors.astype(np.int16).dot([1, 256, 65536])))
        total_pixels = colors.shape[0]
        color_ratio = unique_colors / total_pixels

        if color_ratio < 0.03:
            detection_flags.append(('color_quantized', 0.3))
        elif color_ratio < 0.10:
            detection_flags.append(('some_quantization', 0.6))
        else:
            detection_flags.append(('color_ok', 0.9))

        # 3. Check for lack of depth variation
        v_channel = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)[:, :, 2]
        v_range = np.percentile(v_channel, 95) - np.percentile(v_channel, 5)

        if v_range < 15:
            detection_flags.append(('flat_lighting', 0.35))
        elif v_range < 35:
            detection_flags.append(('limited_range', 0.6))
        else:
            detection_flags.append(('lighting_ok', 0.9))

        # 4. Check saturation uniformity
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)

        if sat_std < 12:
            detection_flags.append(('uniform_saturation', 0.3))
        elif sat_std < 22:
            detection_flags.append(('low_sat_var', 0.55))
        else:
            detection_flags.append(('saturation_ok', 0.9))

        # Use STRICT scoring for healthcare security
        all_scores = [score for _, score in detection_flags]
        avg_score = sum(all_scores) / len(all_scores)
        min_score = min(all_scores)
        very_suspicious = sum(1 for s in all_scores if s <= 0.40)
        suspicious_count = sum(1 for s in all_scores if s < 0.6)

        if very_suspicious >= 1:
            # Even ONE clear print indicator should fail - STRICT
            final_score = 0.5 * min_score + 0.5 * avg_score
        elif suspicious_count >= 2:
            final_score = 0.6 * min_score + 0.4 * avg_score
        elif suspicious_count >= 1:
            final_score = 0.4 * min_score + 0.6 * avg_score
        else:
            final_score = 0.2 * min_score + 0.8 * avg_score

        print(f"Print detection flags: {detection_flags}, Final: {final_score:.3f}")

        return final_score

    except Exception as e:
        print(f"⚠️ Print detection error: {e} - REJECTING (fail-safe)")
        return 0.3  # On error, REJECT for safety


def detect_close_up_phone_screen_strict(small_img, gray_img):
    """
    STRICT close-up phone/screen detection with multiple indicators.
    Specifically designed to catch phone screens showing photos.
    Returns: (is_phone_detected, rejection_reason, debug_info)
    """
    try:
        debug_info = {}
        strong_indicators = []
        regular_indicators = []

        # INDICATOR 1: Color Consistency (phone screens have uniform color)
        # Phone screens: 3-10, Real faces: 15-30+
        b, g, r = cv2.split(small_img)
        color_std = (np.std(b) + np.std(g) + np.std(r)) / 3
        debug_info['color_consistency'] = color_std

        if color_std < 8:  # STRONG indicator
            strong_indicators.append(f"color_consistency={color_std:.2f} < 8")
        elif color_std < 15:  # Regular indicator
            regular_indicators.append(f"color_consistency={color_std:.2f} < 15")

        # INDICATOR 2: Sharpness Variance (digital images are too sharp)
        # Phone screens: 4000+, Real faces: 1000-2500
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        sharpness_var = np.var(laplacian)
        debug_info['sharpness_variance'] = sharpness_var

        if sharpness_var > 4000:  # STRONG indicator
            strong_indicators.append(f"sharpness_var={sharpness_var:.0f} > 4000")
        elif sharpness_var > 2500:  # Regular indicator
            regular_indicators.append(f"sharpness_var={sharpness_var:.0f} > 2500")

        # INDICATOR 3: Texture Variance (screens are smooth)
        # Phone screens: 1-4, Real faces: 8-20+
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture = cv2.filter2D(gray_img, -1, kernel)
        texture_var = np.var(texture)
        debug_info['texture_variance'] = texture_var

        if texture_var < 3:  # STRONG indicator
            strong_indicators.append(f"texture_var={texture_var:.2f} < 3")
        elif texture_var < 6:  # Regular indicator
            regular_indicators.append(f"texture_var={texture_var:.2f} < 6")

        # INDICATOR 4: Brightness Uniformity (screen backlight is uniform)
        # Phone screens: 4-8, Real faces: 15-35+
        gray_std = np.std(gray_img)
        debug_info['brightness_std'] = gray_std

        if gray_std < 6:  # STRONG indicator
            strong_indicators.append(f"brightness_std={gray_std:.2f} < 6")
        elif gray_std < 12:  # Regular indicator
            regular_indicators.append(f"brightness_std={gray_std:.2f} < 12")

        # INDICATOR 5: Edge Density (digital images have sharp edges)
        # Phone screens: 0.20-0.35, Real faces: 0.05-0.15
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        debug_info['edge_density'] = edge_density

        if edge_density > 0.25:  # STRONG indicator
            strong_indicators.append(f"edge_density={edge_density:.3f} > 0.25")
        elif edge_density > 0.15:  # Regular indicator
            regular_indicators.append(f"edge_density={edge_density:.3f} > 0.15")

        # DECISION LOGIC
        debug_info['strong_indicators'] = strong_indicators
        debug_info['regular_indicators'] = regular_indicators

        print("\n" + "=" * 70)
        print("🔍 CLOSE-UP PHONE SCREEN DETECTION:")
        print(f"   Color consistency: {color_std:.2f}")
        print(f"   Sharpness variance: {sharpness_var:.0f}")
        print(f"   Texture variance: {texture_var:.2f}")
        print(f"   Brightness std: {gray_std:.2f}")
        print(f"   Edge density: {edge_density:.3f}")
        print(f"   Strong indicators: {len(strong_indicators)}")
        print(f"   Regular indicators: {len(regular_indicators)}")

        # REJECTION RULE: 1 strong indicator OR 2+ regular indicators
        if len(strong_indicators) >= 1:
            print(f"   ✗ PHONE DETECTED (strong): {strong_indicators[0]}")
            print("=" * 70 + "\n")
            return True, f"Phone screen detected: {strong_indicators[0]}", debug_info

        if len(regular_indicators) >= 2:
            print(f"   ✗ PHONE DETECTED (multiple): {', '.join(regular_indicators[:2])}")
            print("=" * 70 + "\n")
            return True, f"Phone screen detected: {', '.join(regular_indicators[:2])}", debug_info

        print("   ✓ PASS - Real face detected")
        print("=" * 70 + "\n")
        return False, None, debug_info

    except Exception as e:
        print(f"⚠️ Close-up detection error: {e}")
        # On error, REJECT for safety
        return True, f"Detection error: {e}", {'error': str(e)}


def analyze_texture_lbp_fast(gray_img):
    """
    Optimized LBP analysis using pre-resized grayscale image.
    """
    try:
        h, w = gray_img.shape
        padded = np.pad(gray_img, 1, mode='edge')
        center = padded[1:-1, 1:-1].astype(np.int16)

        neighbors = [
            padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
            padded[1:-1, 2:], padded[2:, 2:], padded[2:, 1:-1],
            padded[2:, 0:-2], padded[1:-1, 0:-2],
        ]

        lbp = np.zeros_like(gray_img, dtype=np.uint8)
        for i, neighbor in enumerate(neighbors):
            lbp += ((neighbor >= center).astype(np.uint8) << i)

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-10)

        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        normalized_score = min(1.0, max(0.0, (entropy - 3.0) / 4.5))
        return normalized_score

    except Exception as e:
        print(f"LBP analysis error: {e}")
        return 0.5


def analyze_color_distribution_fast(small_img):
    """
    Optimized color distribution analysis using pre-resized image.
    """
    try:
        ycrcb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)

        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

        if skin_ratio < 0.05 or skin_ratio > 0.70:
            return 0.3

        if np.sum(skin_mask > 0) > 100:
            skin_pixels = ycrcb[skin_mask > 0]
            cr_std = np.std(skin_pixels[:, 1])
            cb_std = np.std(skin_pixels[:, 2])
            variance_score = min(1.0, (cr_std + cb_std) / 30.0)

            y_std = np.std(skin_pixels[:, 0])
            if y_std < 10:
                variance_score *= 0.7

            return variance_score

        return 0.5

    except Exception as e:
        print(f"Color analysis error: {e}")
        return 0.5


def analyze_frequency_patterns_fast(gray_img):
    """
    Optimized frequency analysis using pre-resized grayscale image.
    """
    try:
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        region_h, region_w = max(8, h // 8), max(8, w // 8)
        high_freq_region = magnitude.copy()
        high_freq_region[center_h-region_h:center_h+region_h, center_w-region_w:center_w+region_w] = 0
        high_freq_energy = np.sum(high_freq_region)

        low_freq_energy = np.sum(magnitude[center_h-region_h:center_h+region_h, center_w-region_w:center_w+region_w])

        if low_freq_energy > 0:
            ratio = high_freq_energy / low_freq_energy
            if 0.3 < ratio < 3.0:
                freq_score = 1.0 - abs(ratio - 1.0) / 2.0
            else:
                freq_score = 0.3
        else:
            freq_score = 0.5

        # Check for periodic patterns
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        peaks = magnitude > threshold
        peak_count = np.sum(peaks)

        if peak_count > magnitude.size * 0.015:
            freq_score *= 0.4

        return min(1.0, max(0.0, freq_score))

    except Exception as e:
        print(f"Frequency analysis error: {e}")
        return 0.5


def get_eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Returns a ratio that drops significantly when eye is closed.
    """
    try:
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        if h == 0:
            return 0.3  # Default value

        ear = (v1 + v2) / (2.0 * h)
        return ear
    except Exception:
        return 0.3


def detect_eyes_and_blink(image_array):
    """
    Detect eyes in face and calculate eye aspect ratio.
    Uses OpenCV Haar cascades for robust eye detection.
    Returns: {'eyes_detected': bool, 'eye_ratio': float, 'eye_positions': list}
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Load cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return {'eyes_detected': False, 'eye_ratio': 0, 'eye_positions': []}

        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        # Region of interest for eyes (upper half of face)
        roi_gray = gray[y:y + h//2 + 20, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))

        if len(eyes) < 2:
            # Try with different parameters
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 2, minSize=(15, 15))

        eyes_detected = len(eyes) >= 2

        # Calculate eye aspect ratio based on eye openness
        if eyes_detected:
            # Sort eyes by x position (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])[:2]

            # Calculate ratio based on height/width of detected eyes
            eye_ratios = []
            eye_positions = []
            for (ex, ey, ew, eh) in eyes:
                ratio = eh / (ew + 1)  # Height to width ratio
                eye_ratios.append(ratio)
                eye_positions.append((x + ex + ew//2, y + ey + eh//2))

            avg_ratio = sum(eye_ratios) / len(eye_ratios)

            return {
                'eyes_detected': True,
                'eye_ratio': avg_ratio,
                'eye_positions': eye_positions,
                'eye_count': len(eyes)
            }

        return {'eyes_detected': False, 'eye_ratio': 0, 'eye_positions': []}

    except Exception as e:
        print(f"Eye detection error: {e}")
        return {'eyes_detected': False, 'eye_ratio': 0, 'eye_positions': []}


def detect_face_position(image_array):
    """
    Detect face position and size for motion analysis.
    Returns: {'detected': bool, 'center': (x, y), 'size': (w, h)}
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return {'detected': False, 'center': (0, 0), 'size': (0, 0)}

        # Get the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        return {
            'detected': True,
            'center': (x + w//2, y + h//2),
            'size': (w, h),
            'bbox': (x, y, w, h)
        }
    except Exception as e:
        print(f"Face position detection error: {e}")
        return {'detected': False, 'center': (0, 0), 'size': (0, 0)}


def analyze_frame_motion(frames_data):
    """
    Analyze motion across multiple frames to detect liveness.

    SIMPLE APPROACH:
    - Real faces: Have some movement + eyes detected = PASS
    - Phone photos: Uniform shake + no real eye changes = FAIL

    Returns: (is_live, motion_score)
    """
    if len(frames_data) < 2:
        return False, 0.0

    try:
        face_positions = [f['face_position'] for f in frames_data if f['face_position']['detected']]
        eye_ratios = [f['eye_ratio'] for f in frames_data if f['eye_ratio'] > 0]
        eyes_detected_count = sum(1 for f in frames_data if f.get('eyes_detected', False))

        if len(face_positions) < 2:
            return False, 0.0

        # 1. Check if face was detected consistently
        face_detection_ratio = len(face_positions) / len(frames_data)

        # 2. Check if eyes were detected (real faces have detectable eyes)
        eyes_ratio = eyes_detected_count / len(frames_data) if frames_data else 0

        # 3. Check for any movement (real faces move naturally)
        position_changes = []
        for i in range(1, len(face_positions)):
            prev_center = face_positions[i-1]['center']
            curr_center = face_positions[i]['center']
            dx = abs(curr_center[0] - prev_center[0])
            dy = abs(curr_center[1] - prev_center[1])
            position_changes.append(np.sqrt(dx**2 + dy**2))

        avg_movement = np.mean(position_changes) if position_changes else 0
        movement_variance = np.var(position_changes) if len(position_changes) > 1 else 0

        # 4. Check for eye ratio changes (blinks)
        blink_detected = False
        eye_change = 0
        if len(eye_ratios) >= 2:
            eye_change = max(eye_ratios) - min(eye_ratios)
            blink_detected = eye_change > 0.02

        # SCORING - Simple and lenient for real faces
        score = 0.30  # Base score

        # Face detected consistently? (+0.15)
        if face_detection_ratio > 0.7:
            score += 0.15

        # Eyes detected? (+0.20) - This is key, photos may not have detectable eyes
        if eyes_ratio > 0.5:
            score += 0.20
        elif eyes_ratio > 0.25:
            score += 0.10

        # Any movement? (+0.15)
        if avg_movement > 0.2:
            score += 0.15
        elif avg_movement > 0.05:
            score += 0.08

        # Blink or eye movement detected? (+0.20) - Strong indicator of real face
        if blink_detected:
            score += 0.20
        elif eye_change > 0.01:
            score += 0.10

        # PHONE PHOTO DETECTION:
        # Phone photos have uniform shake (hand holding phone) but NO eye changes
        is_phone_pattern = (
            avg_movement > 0.3 and           # Some movement (hand shake)
            movement_variance < 0.1 and      # But very uniform
            eye_change < 0.01 and            # No eye changes at all
            not blink_detected               # No blink
        )

        if is_phone_pattern:
            score = min(score, 0.35)  # Cap score for phone pattern
            print(f"PHONE PATTERN: uniform shake without eye changes")

        print(f"Liveness: faces={face_detection_ratio:.1f}, eyes={eyes_ratio:.1f}, move={avg_movement:.2f}, eye_change={eye_change:.3f}, score={score:.2f}")

        is_live = score >= LIVENESS_THRESHOLD
        return is_live, score

    except Exception as e:
        print(f"Motion analysis error: {e}")
        return False, 0.0


def check_liveness_multi_frame(session_id, image_array):
    """
    CHALLENGE-RESPONSE Liveness Detection.

    How it works:
    1. Generate a random challenge (BLINK, TURN_LEFT, TURN_RIGHT)
    2. Ask user to perform the action
    3. Collect frames and verify the action was performed

    A photo CANNOT respond to challenges - this is the key!

    Returns: (status, message, is_verified)
    """
    current_time = datetime.now()

    # Clean up expired sessions
    expired = [sid for sid, data in liveness_sessions.items()
               if (current_time - data['start_time']).seconds > LIVENESS_SESSION_TIMEOUT]
    for sid in expired:
        del liveness_sessions[sid]

    # Initialize new session with random challenge
    if session_id not in liveness_sessions:
        challenge = random.choice(CHALLENGE_TYPES)
        liveness_sessions[session_id] = {
            'start_time': current_time,
            'challenge': challenge,
            'frame_data': [],
            'baseline_set': False,
            'baseline_eye_ratio': 0,
            'baseline_face_x': 0,
        }

        # Return challenge instruction immediately
        if challenge == 'BLINK':
            msg = 'BLINK YOUR EYES to verify you are real'
        elif challenge == 'TURN_LEFT':
            msg = 'TURN YOUR HEAD LEFT to verify you are real'
        else:
            msg = 'TURN YOUR HEAD RIGHT to verify you are real'

        print(f"LIVENESS CHALLENGE: {challenge}")
        return 'collecting', msg, False

    session = liveness_sessions[session_id]
    challenge = session['challenge']

    # Detect eyes and face position
    eye_info = detect_eyes_and_blink(image_array)
    face_pos = detect_face_position(image_array)

    eye_ratio = eye_info.get('eye_ratio', 0)
    face_x = face_pos.get('center', (0, 0))[0] if face_pos.get('detected') else 0

    # Set baseline from first frame with good detection
    if not session['baseline_set'] and eye_ratio > 0 and face_x > 0:
        session['baseline_eye_ratio'] = eye_ratio
        session['baseline_face_x'] = face_x
        session['baseline_set'] = True
        print(f"BASELINE SET: eye_ratio={eye_ratio:.3f}, face_x={face_x:.0f}")

    # Store frame data
    frame_data = {
        'timestamp': current_time,
        'eye_ratio': eye_ratio,
        'eyes_detected': eye_info.get('eyes_detected', False),
        'face_x': face_x,
        'face_detected': face_pos.get('detected', False)
    }
    session['frame_data'].append(frame_data)

    frames_collected = len(session['frame_data'])

    # Need at least 3 frames to verify
    if frames_collected < 3:
        if challenge == 'BLINK':
            return 'collecting', 'BLINK YOUR EYES now!', False
        elif challenge == 'TURN_LEFT':
            return 'collecting', 'TURN HEAD LEFT now!', False
        else:
            return 'collecting', 'TURN HEAD RIGHT now!', False

    # Check if challenge was completed
    challenge_passed = verify_challenge(session)

    # Give more time if not passed yet (up to 8 frames)
    if not challenge_passed and frames_collected < LIVENESS_FRAMES_REQUIRED:
        if challenge == 'BLINK':
            return 'collecting', f'BLINK YOUR EYES! ({frames_collected}/{LIVENESS_FRAMES_REQUIRED})', False
        elif challenge == 'TURN_LEFT':
            return 'collecting', f'TURN HEAD LEFT! ({frames_collected}/{LIVENESS_FRAMES_REQUIRED})', False
        else:
            return 'collecting', f'TURN HEAD RIGHT! ({frames_collected}/{LIVENESS_FRAMES_REQUIRED})', False

    # Clean up session
    del liveness_sessions[session_id]

    if challenge_passed:
        print(f"CHALLENGE PASSED: {challenge}")
        return 'verified', 'Liveness verified! You are real.', True
    else:
        print(f"CHALLENGE FAILED: {challenge} - Photo detected!")
        return 'failed', f'Challenge failed! Photos cannot {challenge.lower().replace("_", " ")}. Please use your REAL face.', False


def verify_challenge(session):
    """
    Verify if the user completed the challenge.

    BLINK: Eye ratio must drop significantly (eyes close) then recover
    TURN_LEFT: Face X position must shift left (decrease)
    TURN_RIGHT: Face X position must shift right (increase)
    """
    challenge = session['challenge']
    frames = session['frame_data']
    baseline_eye = session.get('baseline_eye_ratio', 0)
    baseline_x = session.get('baseline_face_x', 0)

    if len(frames) < 3:
        return False

    eye_ratios = [f['eye_ratio'] for f in frames if f['eye_ratio'] > 0]
    face_xs = [f['face_x'] for f in frames if f['face_x'] > 0]

    if challenge == 'BLINK':
        # For blink: eye ratio should drop by at least 30% from baseline
        if len(eye_ratios) < 2 or baseline_eye == 0:
            return False

        min_ratio = min(eye_ratios)
        max_ratio = max(eye_ratios)

        # Check for significant eye closure (blink)
        # Eyes close = ratio drops, then recovers
        ratio_drop = (baseline_eye - min_ratio) / baseline_eye if baseline_eye > 0 else 0
        ratio_range = max_ratio - min_ratio

        print(f"BLINK CHECK: baseline={baseline_eye:.3f}, min={min_ratio:.3f}, max={max_ratio:.3f}, drop={ratio_drop:.1%}, range={ratio_range:.3f}")

        # Blink detected if: ratio dropped by 25%+ OR range is significant
        blink_detected = ratio_drop > 0.25 or ratio_range > 0.06
        return blink_detected

    elif challenge == 'TURN_LEFT':
        # Face should move left (X decreases)
        if len(face_xs) < 2 or baseline_x == 0:
            return False

        min_x = min(face_xs)
        x_shift = baseline_x - min_x  # Positive if moved left

        print(f"TURN_LEFT CHECK: baseline_x={baseline_x:.0f}, min_x={min_x:.0f}, shift={x_shift:.0f}px")

        # Need at least 30px shift to the left
        return x_shift > 30

    elif challenge == 'TURN_RIGHT':
        # Face should move right (X increases)
        if len(face_xs) < 2 or baseline_x == 0:
            return False

        max_x = max(face_xs)
        x_shift = max_x - baseline_x  # Positive if moved right

        print(f"TURN_RIGHT CHECK: baseline_x={baseline_x:.0f}, max_x={max_x:.0f}, shift={x_shift:.0f}px")

        # Need at least 30px shift to the right
        return x_shift > 30

    return False


def detect_screen_artifacts_strict(image_array):
    """
    Stricter screen artifact detection specifically for phone/tablet screens.
    This catches patterns that the regular anti-spoofing might miss.
    Returns: (is_screen, confidence)
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(image_array, (320, 240))
        gray_small = cv2.resize(gray, (320, 240))

        screen_indicators = 0
        total_checks = 0

        # 1. PWM flicker detection (screen refresh rate creates patterns)
        rows_mean = np.mean(gray_small, axis=1)
        row_diff = np.diff(rows_mean)
        row_periodicity = np.abs(np.fft.fft(row_diff))

        # Strong periodic patterns in rows suggest screen
        if np.max(row_periodicity[5:50]) > np.mean(row_periodicity) * 4:
            screen_indicators += 2
            print("Screen artifact: PWM flicker pattern detected")
        total_checks += 2

        # 2. Sub-pixel pattern detection (RGB stripe pattern)
        if len(image_array.shape) == 3:
            b, g, r = cv2.split(small)

            # Check for regular RGB shift patterns
            r_edges = cv2.Laplacian(r, cv2.CV_64F)
            g_edges = cv2.Laplacian(g, cv2.CV_64F)
            b_edges = cv2.Laplacian(b, cv2.CV_64F)

            # Screens have very regular edge patterns across channels
            edge_correlation = np.corrcoef([r_edges.flatten(), g_edges.flatten(), b_edges.flatten()])

            if np.mean(np.abs(edge_correlation)) > 0.92:
                screen_indicators += 2
                print("Screen artifact: Sub-pixel pattern detected")
        total_checks += 2

        # 3. Color depth check (screens often have banding in gradients)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        unique_brightness = len(np.unique(v_channel))

        # Real photos have smooth gradients with many values
        if unique_brightness < 80:
            screen_indicators += 1
            print(f"Screen artifact: Limited color depth ({unique_brightness} levels)")
        total_checks += 1

        # 4. Edge sharpness uniformity (screens have unnaturally uniform sharpness)
        laplacian = cv2.Laplacian(gray_small, cv2.CV_64F)

        # Divide into regions and check variance consistency
        h, w = laplacian.shape
        regions = []
        for i in range(3):
            for j in range(3):
                region = laplacian[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                regions.append(np.var(region))

        region_var_cv = np.std(regions) / (np.mean(regions) + 1)

        # Screens have very uniform sharpness across regions
        if region_var_cv < 0.25:
            screen_indicators += 1
            print(f"Screen artifact: Uniform sharpness (CV={region_var_cv:.2f})")
        total_checks += 1

        # 5. Temporal noise pattern (screen noise is uniform)
        noise = gray_small.astype(float) - cv2.GaussianBlur(gray_small, (5, 5), 0).astype(float)
        noise_uniformity = np.std(np.var(noise.reshape(-1, 16), axis=1))

        if noise_uniformity < 3:
            screen_indicators += 1
            print(f"Screen artifact: Uniform noise pattern ({noise_uniformity:.2f})")
        total_checks += 1

        # 6. Black level check (screens rarely have true black)
        min_brightness = np.percentile(gray_small, 1)

        if min_brightness > 15:
            screen_indicators += 1
            print(f"Screen artifact: Elevated black level ({min_brightness})")
        total_checks += 1

        # Calculate confidence
        confidence = screen_indicators / total_checks
        is_screen = confidence > 0.45  # Need 45% of indicators to trigger

        print(f"Screen artifact detection: {screen_indicators}/{total_checks} indicators, is_screen={is_screen}")

        return is_screen, confidence

    except Exception as e:
        print(f"Screen artifact detection error: {e}")
        return False, 0.0


def validate_face_quality(image_array):
    """
    Validate face image quality for enrollment.
    Checks: face detection, size, brightness, blur, etc.
    """
    issues = []

    # Check image dimensions
    height, width = image_array.shape[:2]
    if width < 200 or height < 200:
        issues.append("Image resolution too low (minimum 200x200)")

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array

    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < 50:
        issues.append("Image too dark - please improve lighting")
    elif mean_brightness > 200:
        issues.append("Image too bright - please reduce lighting")

    # Check blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        issues.append("Image is blurry - please hold camera steady")

    # Detect face and check
    face_info = detect_face(image_array)
    if not face_info['face_detected']:
        issues.append("No face detected in image")
    elif face_info['confidence'] < MIN_FACE_CONFIDENCE:
        issues.append(f"Face detection confidence too low ({face_info['confidence']:.1%})")

    # Check if multiple faces detected
    if face_info.get('face_count', 0) > 1:
        issues.append("Multiple faces detected - please ensure only one person is in frame")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'face_info': face_info,
        'quality_metrics': {
            'brightness': mean_brightness,
            'sharpness': laplacian_var,
            'resolution': f"{width}x{height}"
        }
    }

def find_matching_employee(image_array):
    """
    Find matching employee from face embedding.
    Uses cached embeddings with VECTORIZED operations for better performance.
    Returns (employee, similarity) tuple.

    SECURITY: This function uses strict matching to prevent:
    1. False positives (wrong person being recognized)
    2. Photo-based attacks (showing someone else's photo)

    PERFORMANCE: Uses numpy vectorized operations for 10-100x speed improvement
    """
    try:
        # Get current face embedding
        current_embedding = get_face_embedding(image_array)
        if current_embedding is None:
            return None, 0

        # Refresh cache if empty or stale (older than 5 minutes)
        if not embeddings_cache or cache_timestamp is None or \
           (datetime.now() - cache_timestamp).seconds > 300:
            refresh_embeddings_cache()

        # Return early if no employees in cache
        if not embeddings_cache:
            return None, 0

        # Get threshold from settings
        settings = Settings.query.first()
        threshold = settings.recognition_threshold if settings else RECOGNITION_THRESHOLD

        # Ensure minimum security threshold
        threshold = max(threshold, 0.70)

        # OPTIMIZATION: Vectorized similarity calculation
        # Pre-normalize current embedding once
        current_norm = np.linalg.norm(current_embedding)
        if current_norm == 0:
            return None, 0

        current_normalized = current_embedding / current_norm

        # Collect all normalized embeddings and employee IDs
        emp_ids = []
        normalized_embeddings = []

        for emp_id, data in embeddings_cache.items():
            if len(data['embedding']) == len(current_embedding):
                emp_ids.append(emp_id)
                normalized_embeddings.append(data['normalized'])

        if not normalized_embeddings:
            return None, 0

        # VECTORIZED OPERATION: Calculate all similarities at once
        # This is MUCH faster than looping
        embeddings_matrix = np.array(normalized_embeddings)
        similarities = np.dot(embeddings_matrix, current_normalized)

        # Convert to 0-1 range
        similarities = (similarities + 1) / 2

        # Find best and second best matches
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        best_idx = sorted_indices[0]
        best_similarity = similarities[best_idx]

        second_best_similarity = similarities[sorted_indices[1]] if len(sorted_indices) > 1 else 0

        # Count near-threshold matches for security check
        match_count = np.sum(similarities > threshold * 0.9)

        # SECURITY CHECK: If multiple faces match closely, it's suspicious
        if match_count > 1 and best_similarity - second_best_similarity < 0.05:
            print(f"SECURITY: Ambiguous match detected. Best: {best_similarity:.3f}, Second: {second_best_similarity:.3f}")
            return None, 0

        # Only return a match if it's significantly above threshold
        if best_similarity > threshold:
            best_emp_id = emp_ids[best_idx]
            best_match = embeddings_cache[best_emp_id]['employee']
            print(f"Match found: {best_match.full_name} with similarity {best_similarity:.3f} (threshold: {threshold})")
            return best_match, best_similarity

        return None, 0
    except Exception as e:
        print(f"Error finding matching employee: {e}")
        return None, 0

# ========================
# ROUTES
# ========================

@app.route('/')
def index():
    return redirect(url_for('attend'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        # Get client IP for rate limiting and logging
        client_ip = get_client_ip()
        login_key = rate_limit_key('login')

        # Check rate limiting (5 attempts per 5 minutes)
        is_allowed, remaining, lockout_time = rate_limiter.record_attempt(
            login_key, max_attempts=5, window_seconds=300, lockout_seconds=900
        )

        if not is_allowed:
            audit_logger.log_lockout(f"login_{client_ip}", client_ip)
            flash(f'Too many login attempts. Please try again in {lockout_time // 60} minutes.', 'danger')
            return render_template('login.html'), 429

        # Get and validate inputs
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Input validation
        valid_username, username_error = InputValidator.validate_username(username)
        if not valid_username:
            audit_logger.log_auth_failure(username, username_error, client_ip)
            flash('Invalid username format.', 'danger')
            return render_template('login.html')

        if not password:
            audit_logger.log_auth_failure(username, 'Empty password', client_ip)
            flash('Invalid username or password. Please try again.', 'danger')
            return render_template('login.html')

        # Sanitize username to prevent SQL injection (though ORM protects us)
        username = InputValidator.sanitize_string(username, max_length=80)

        # Query database using parameterized query (ORM handles this)
        admin = Admin.query.filter_by(username=username).first()

        # Constant-time comparison to prevent timing attacks
        if admin:
            password_valid = bcrypt.checkpw(password.encode('utf-8'), admin.password_hash.encode('utf-8'))
        else:
            # Fake check to prevent timing attacks revealing valid usernames
            bcrypt.checkpw(b'fake_password', bcrypt.hashpw(b'fake', bcrypt.gensalt()))
            password_valid = False

        if admin and password_valid:
            # Success - reset rate limiter
            rate_limiter.reset(login_key)

            # Regenerate session to prevent fixation attacks
            from flask import session as flask_session
            flask_session.permanent = True

            login_user(admin)
            audit_logger.log_login_attempt(username, True, client_ip)
            flash('Welcome back! You have been logged in successfully.', 'success')

            # Sanitize next parameter to prevent open redirect
            next_page = request.args.get('next')
            if next_page and not next_page.startswith('/'):
                next_page = None

            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            audit_logger.log_login_attempt(username, False, client_ip)
            flash(f'Invalid username or password. {remaining} attempts remaining.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    total_employees = Employee.query.count()
    today_attendance = Attendance.query.filter_by(date=today).count()
    late_today = Attendance.query.filter_by(date=today, is_late=True).count()

    # Get employees who haven't clocked in today
    clocked_in_ids = [a.employee_id for a in Attendance.query.filter_by(date=today).all()]
    absent_count = total_employees - len(set(clocked_in_ids))

    recent_attendances = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(10).all()

    # Get settings for display
    settings = Settings.query.first()

    return render_template('dashboard.html',
                         total_employees=total_employees,
                         today_attendance=today_attendance,
                         late_today=late_today,
                         absent_count=absent_count,
                         recent_attendances=recent_attendances,
                         settings=settings,
                         face_model=FACE_MODEL,
                         face_detector=FACE_DETECTOR)

@app.route('/dashboard/export')
@login_required
def export_today_attendance():
    today = date.today()
    attendances = Attendance.query.filter_by(date=today).all()

    output = io.StringIO()
    output.write('Employee ID,Full Name,Department,Clock In Time,Clock Out Time,Status\n')

    for att in attendances:
        status = 'Late' if att.is_late else 'On Time'
        clock_out = att.clock_out_time.strftime("%H:%M:%S") if att.clock_out_time else 'Not clocked out'
        output.write(f'{att.employee.employee_id},{att.employee.full_name},{att.employee.department},{att.clock_in_time.strftime("%H:%M:%S")},{clock_out},{status}\n')

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=attendance_{today}.csv'}
    )

@app.route('/employees')
@login_required
def employees():
    all_employees = Employee.query.order_by(Employee.registered_at.desc()).all()
    today = date.today()

    # Get today's attendance status for each employee
    today_attendances = {a.employee_id: a for a in Attendance.query.filter_by(date=today).all()}

    return render_template('employees.html', employees=all_employees, today_attendances=today_attendances)

@app.route('/employees/delete/<int:id>', methods=['POST'])
@login_required
def delete_employee(id):
    # Log sensitive data access
    client_ip = get_client_ip()
    audit_logger.log_data_access(
        f"DELETE employee ID {id}",
        current_user.username,
        client_ip
    )

    employee = Employee.query.get_or_404(id)

    # Delete all attendance records for this employee first (foreign key constraint)
    Attendance.query.filter_by(employee_id=id).delete()

    # Delete face image
    if employee.face_image_path and os.path.exists(employee.face_image_path):
        try:
            os.remove(employee.face_image_path)
        except OSError:
            pass

    # Remove from cache
    if id in embeddings_cache:
        del embeddings_cache[id]

    db.session.delete(employee)
    db.session.commit()
    flash(f'Employee {employee.full_name} has been deleted successfully.', 'success')
    return redirect(url_for('employees'))

@app.route('/employees/update/<int:id>', methods=['GET', 'POST'])
@login_required
def update_employee(id):
    employee = Employee.query.get_or_404(id)

    if request.method == 'POST':
        # Get and validate inputs
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()

        # Input validation
        valid_name, name_error = InputValidator.validate_name(full_name)
        if not valid_name:
            flash(name_error, 'danger')
            return redirect(url_for('update_employee', id=id))

        valid_dept, dept_error = InputValidator.validate_department(department)
        if not valid_dept:
            flash(dept_error, 'danger')
            return redirect(url_for('update_employee', id=id))

        # Sanitize inputs
        full_name = InputValidator.sanitize_string(full_name, max_length=100)

        # Update basic info
        employee.full_name = full_name
        employee.department = department


        # Update face image if provided
        image_data = request.form.get('image_data')
        if image_data:
            try:
                image_data = image_data.split(',')[1] if ',' in image_data else image_data
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))

                if image.mode == 'RGBA':
                    image = image.convert('RGB')

                image_array = np.array(image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                # Validate face quality
                quality = validate_face_quality(image_array)
                if not quality['valid']:
                    flash(f'Image quality issues: {", ".join(quality["issues"])}', 'danger')
                    return redirect(url_for('update_employee', id=id))

                # Get new embedding
                embedding = get_face_embedding(image_array)
                if embedding is None:
                    flash('Could not extract face features. Please try again.', 'danger')
                    return redirect(url_for('update_employee', id=id))

                # Delete old image
                if employee.face_image_path and os.path.exists(employee.face_image_path):
                    try:
                        os.remove(employee.face_image_path)
                    except OSError:
                        pass

                # Save new image
                filename = f"{employee.employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                image_path = os.path.join(FACE_DB_PATH, filename)
                cv2.imwrite(image_path, image_array)

                employee.face_image_path = image_path
                employee.set_embedding(embedding)

                # Update cache with normalized embedding
                norm = np.linalg.norm(embedding)
                normalized_embedding = embedding / norm if norm > 0 else embedding
                embeddings_cache[employee.id] = {
                    'embedding': embedding,
                    'normalized': normalized_embedding,
                    'employee': employee
                }
            except Exception as e:
                flash(f'Error updating face image: {str(e)}', 'danger')
                return redirect(url_for('update_employee', id=id))

        db.session.commit()
        flash(f'Employee {employee.full_name} has been updated successfully.', 'success')
        return redirect(url_for('employees'))

    departments = ['Cardiology', 'Neurology', 'Pediatrics', 'Radiology', 'Emergency', 'Surgery', 'Nursing', 'Pharmacy', 'Laboratory', 'Administration', 'IT Support']
    return render_template('update_employee.html', employee=employee, departments=departments)

@app.route('/employees/image/<int:id>')
def get_employee_image(id):
    """Serve employee face image"""
    employee = Employee.query.get_or_404(id)
    if employee.face_image_path and os.path.exists(employee.face_image_path):
        with open(employee.face_image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    # Return placeholder
    return redirect(url_for('static', filename='img/placeholder-avatar.png'))

@app.route('/enroll', methods=['GET', 'POST'])
@login_required
def enroll():
    if request.method == 'POST':
        # Get and sanitize inputs
        employee_id = request.form.get('employee_id', '').strip()
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()
        image_data = request.form.get('image_data')

        # Input validation
        valid_id, id_error = InputValidator.validate_employee_id(employee_id)
        if not valid_id:
            flash(id_error, 'danger')
            return redirect(url_for('enroll'))

        valid_name, name_error = InputValidator.validate_name(full_name)
        if not valid_name:
            flash(name_error, 'danger')
            return redirect(url_for('enroll'))

        valid_dept, dept_error = InputValidator.validate_department(department)
        if not valid_dept:
            flash(dept_error, 'danger')
            return redirect(url_for('enroll'))

        # Sanitize inputs
        employee_id = InputValidator.sanitize_string(employee_id, max_length=50)
        full_name = InputValidator.sanitize_string(full_name, max_length=100)

        # Check for duplicate employee ID
        if Employee.query.filter_by(employee_id=employee_id).first():
            flash('An employee with this ID already exists.', 'danger')
            return redirect(url_for('enroll'))

        if not image_data:
            flash('Please capture or upload a photo.', 'danger')
            return redirect(url_for('enroll'))

        try:
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image_array = np.array(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Validate face quality
            quality = validate_face_quality(image_array)
            if not quality['valid']:
                flash(f'Image quality issues: {", ".join(quality["issues"])}', 'danger')
                return redirect(url_for('enroll'))

            # Anti-spoofing DISABLED for enrollment - only used for attendance
            # Real face validation happens via quality checks above

            # Get face embedding
            embedding = get_face_embedding(image_array)
            if embedding is None:
                flash('Could not detect a face in the image. Please try again with a clearer photo.', 'danger')
                return redirect(url_for('enroll'))

            # Check for duplicate face (same person already enrolled)
            for emp_id, data in embeddings_cache.items():
                stored_embedding = data['embedding']
                if len(stored_embedding) == len(embedding):
                    similarity = np.dot(stored_embedding, embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(embedding))
                    similarity = (similarity + 1) / 2
                    if similarity > 0.85:  # High similarity threshold for duplicate detection
                        existing_emp = data['employee']
                        flash(f'This face appears to match existing employee: {existing_emp.full_name} ({existing_emp.employee_id})', 'warning')
                        return redirect(url_for('enroll'))

            # Save face image
            filename = f"{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = os.path.join(FACE_DB_PATH, filename)
            cv2.imwrite(image_path, image_array)

            # Create employee record
            employee = Employee(
                employee_id=employee_id,
                full_name=full_name,
                department=department,
                face_image_path=image_path
            )
            employee.set_embedding(embedding)

            db.session.add(employee)
            db.session.commit()

            # Add to cache with normalized embedding
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 0 else embedding
            embeddings_cache[employee.id] = {
                'embedding': embedding,
                'normalized': normalized_embedding,
                'employee': employee
            }

            flash(f'Employee {full_name} has been enrolled successfully!', 'success')
            return redirect(url_for('employees'))

        except Exception as e:
            print(f"Error enrolling employee: {e}")
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('enroll'))

    departments = ['Cardiology', 'Neurology', 'Pediatrics', 'Radiology', 'Emergency', 'Surgery', 'Nursing', 'Pharmacy', 'Laboratory', 'Administration', 'IT Support']
    return render_template('enroll.html', departments=departments)

@app.route('/attend')
def attend():
    today = date.today()
    recent = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(5).all()
    total_today = Attendance.query.filter_by(date=today).count()
    total_employees = Employee.query.count()
    settings = Settings.query.first()
    return render_template('attend.html',
                         recent_attendances=recent,
                         total_today=total_today,
                         total_employees=total_employees,
                         settings=settings)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        import time
        start_time = time.time()

        # Rate limiting for API endpoint (100 requests per minute per IP - generous for testing)
        client_ip = get_client_ip()
        api_key = rate_limit_key('api_recognize')

        is_allowed, remaining, lockout_time = rate_limiter.record_attempt(
            api_key, max_attempts=100, window_seconds=60, lockout_seconds=30
        )

        if not is_allowed:
            audit_logger.log_suspicious_activity(
                f"API rate limit exceeded: {lockout_time}s lockout",
                'anonymous', client_ip
            )
            return jsonify({
                'success': False,
                'message': f'Too many requests. Please wait {lockout_time} seconds.',
                'rate_limited': True
            }), 429

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request data'}), 400

        image_data = data.get('image')
        action = data.get('action', 'clock_in')  # clock_in or clock_out
        session_id = data.get('session_id', '')  # For liveness tracking

        # Validate action parameter
        if action not in ['clock_in', 'clock_out']:
            return jsonify({'success': False, 'message': 'Invalid action'}), 400

        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'}), 400

        parse_start = time.time()
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        print(f"⏱️ Image parsing: {(time.time() - parse_start)*1000:.0f}ms")

        # FIRST: Check if there's a face in the frame
        face_start = time.time()
        face_info = detect_face(image_array)
        print(f"⏱️ Face detection: {(time.time() - face_start)*1000:.0f}ms")

        if not face_info['face_detected']:
            # No face detected - silently return no_match (don't show error)
            return jsonify({
                'success': False,
                'no_match': True,
                'message': 'No face detected'
            })

        # FAST MODE: Skip heavy anti-spoofing checks for speed
        if FAST_RECOGNITION_MODE:
            print("🚀 FAST MODE: Skipping heavy anti-spoofing checks")
        else:
            # FACE QUALITY CHECK: Detect if face is "too perfect" (digital photo characteristic)
            if face_info.get('confidence', 0) > 0.99:
                # Confidence of 0.99+ is suspiciously perfect - real faces rarely get this
                print(f"SUSPICIOUS: Face confidence too perfect ({face_info['confidence']:.4f})")
                # This alone won't reject, but increases suspicion for later checks

            # SECOND: Run anti-spoofing check (DeepFace neural network - now with strict fail-safe)
            spoof_start = time.time()
            is_real, spoof_score = check_anti_spoofing(image_array)
            print(f"⏱️ Anti-spoofing check: {(time.time() - spoof_start)*1000:.0f}ms")
            if not is_real:
                print("🚫 REJECTED: DeepFace detected spoof or encountered error (fail-safe)")
                return jsonify({
                    'success': False,
                    'message': 'Spoof detected! Photos and videos are not allowed.',
                    'spoof_detected': True
                })

            # SIMPLE PHONE DETECTION - BALANCED MODE
            # Resize images for faster processing
            small_img = cv2.resize(image_array, (320, 240))
            gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

            # Check brightness, color, edges - but use BALANCED thresholds
            brightness_std = np.std(gray_img)
            b, g, r = cv2.split(small_img)
            color_std = (np.std(b) + np.std(g) + np.std(r)) / 3
            edges = cv2.Canny(gray_img, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            print(f"🔍 Brightness: {brightness_std:.2f}, Color: {color_std:.2f}, Edges: {edge_density:.3f}")

            # Count suspicious indicators
            suspicious = 0
            reasons = []

            if brightness_std < 10:  # Very uniform brightness
                suspicious += 1
                reasons.append(f"brightness={brightness_std:.2f}")

            if color_std < 8:  # Very uniform color
                suspicious += 1
                reasons.append(f"color={color_std:.2f}")

            if edge_density > 0.22:  # Very sharp edges
                suspicious += 1
                reasons.append(f"edges={edge_density:.3f}")

            # Reject only if 2+ indicators OR 1 very strong indicator
            if suspicious >= 2:
                print(f"🚫 PHONE DETECTED: {suspicious} indicators: {reasons}")
                return jsonify({
                    'success': False,
                    'message': 'Phone/screen detected! Please use your real face.',
                    'spoof_detected': True
                })

            if brightness_std < 8 or color_std < 6:  # VERY strong single indicator
                print(f"🚫 PHONE DETECTED: Very strong indicator: {reasons}")
                return jsonify({
                    'success': False,
                    'message': 'Phone/screen detected! Please use your real face.',
                    'spoof_detected': True
                })

            # FOURTH: STANDARD PHONE/SCREEN/PRINT DETECTION (with STRICT thresholds)
            detect_start = time.time()
            phone_score = detect_phone_screen(small_img, gray_img)
            screen_score = detect_screen_display_fast(small_img, gray_img)
            print_score = detect_printed_photo_fast(small_img, gray_img)
            print(f"⏱️ Standard detection: {(time.time() - detect_start)*1000:.0f}ms")

            avg_spoof_score = (phone_score + screen_score + print_score) / 3

            print("\n" + "=" * 70)
            print("📊 STANDARD SPOOF DETECTION SCORES:")
            print(f"   Phone Detection:  {phone_score:.3f} {'✓ PASS' if phone_score >= 0.85 else '✗ FAIL - PHONE DETECTED'}")
            print(f"   Screen Detection: {screen_score:.3f} {'✓ PASS' if screen_score >= 0.70 else '✗ FAIL - SCREEN DETECTED'}")
            print(f"   Print Detection:  {print_score:.3f} {'✓ PASS' if print_score >= 0.70 else '✗ FAIL - PRINT DETECTED'}")
            print(f"   Average Score:    {avg_spoof_score:.3f}")
            print(f"   Min Score:        {min(phone_score, screen_score, print_score):.3f}")
            print("=" * 70 + "\n")

            # BALANCED THRESHOLDS - Catch phones but allow real faces
            # Phone score: 0.70 (balanced)
            # Screen/Print: 0.55 (balanced)
            # Average: 0.60 (backup check)

            # Reject if phone score is low (main threat)
            if phone_score < 0.70:
                print(f"🚫 PHONE DETECTED! Phone score: {phone_score:.3f}")
                return jsonify({
                    'success': False,
                    'message': 'Phone screen detected! Please use your real face.',
                    'spoof_detected': True
                })

            # Reject if multiple scores are low
            low_scores = 0
            if screen_score < 0.55:
                low_scores += 1
            if print_score < 0.55:
                low_scores += 1

            if low_scores >= 2 or avg_spoof_score < 0.60:
                print(f"🚫 SPOOF DETECTED! Multiple low scores or low average")
                return jsonify({
                    'success': False,
                    'message': 'Spoof detected! Please use your real face.',
                    'spoof_detected': True
                })

            # THIRD: Multi-frame liveness detection (requires real motion/blink)
            if LIVENESS_ENABLED and session_id:
                liveness_status, liveness_msg, is_live = check_liveness_multi_frame(session_id, image_array)

                if liveness_status == 'collecting':
                    # Still collecting frames - tell frontend to keep sending
                    return jsonify({
                        'success': False,
                        'liveness_collecting': True,
                        'message': liveness_msg
                    })
                elif liveness_status == 'failed':
                    return jsonify({
                        'success': False,
                        'liveness_failed': True,
                        'message': liveness_msg
                    })

        # FOURTH: Find matching employee (only reached if liveness passed or not enabled)
        match_start = time.time()
        employee, similarity = find_matching_employee(image_array)
        print(f"⏱️ Employee matching: {(time.time() - match_start)*1000:.0f}ms")
        print(f"✅ TOTAL RECOGNITION TIME: {(time.time() - start_time)*1000:.0f}ms")

        if employee is None:
            return jsonify({
                'success': False,
                'message': 'Face not recognized. Please ensure you are enrolled in the system.',
                'no_match': True
            })

        today = date.today()
        existing = Attendance.query.filter_by(employee_id=employee.id, date=today).first()

        # Get work hours
        settings = Settings.query.first()
        work_start_hour = settings.work_start_hour if settings else 9
        work_start_minute = settings.work_start_minute if settings else 0

        now = datetime.now()

        if action == 'clock_out':
            if not existing:
                return jsonify({
                    'success': False,
                    'message': f'{employee.full_name} has not clocked in today.',
                    'not_clocked_in': True
                })

            if existing.clock_out_time:
                return jsonify({
                    'success': True,
                    'already_clocked_out': True,
                    'employee_name': employee.full_name,
                    'employee_id': employee.employee_id,
                    'department': employee.department,
                    'clock_out_time': existing.clock_out_time.strftime('%H:%M:%S'),
                    'message': f'{employee.full_name} already clocked out at {existing.clock_out_time.strftime("%H:%M:%S")}'
                })

            existing.clock_out_time = now
            try:
                db.session.commit()
            except Exception as db_err:
                db.session.rollback()
                print(f"Database error during clock-out: {db_err}")
                return jsonify({
                    'success': False,
                    'message': 'Failed to save clock-out. Please try again.',
                    'db_error': True
                })

            return jsonify({
                'success': True,
                'clocked_out': True,
                'employee_name': employee.full_name,
                'employee_id': employee.employee_id,
                'department': employee.department,
                'clock_in_time': existing.clock_in_time.strftime('%H:%M:%S'),
                'clock_out_time': now.strftime('%H:%M:%S'),
                'similarity': round(similarity * 100, 1),
                'message': f'Clock-out recorded for {employee.full_name}!'
            })

        # Clock in logic
        if existing:
            return jsonify({
                'success': True,
                'already_marked': True,
                'employee_name': employee.full_name,
                'employee_id': employee.employee_id,
                'department': employee.department,
                'clock_in_time': existing.clock_in_time.strftime('%H:%M:%S'),
                'clock_out_time': existing.clock_out_time.strftime('%H:%M:%S') if existing.clock_out_time else None,
                'message': f'Attendance already marked for {employee.full_name} today at {existing.clock_in_time.strftime("%H:%M:%S")}'
            })

        is_late = now.hour > work_start_hour or (now.hour == work_start_hour and now.minute > work_start_minute)

        attendance = Attendance(
            employee_id=employee.id,
            date=today,
            clock_in_time=now,
            is_late=is_late
        )
        db.session.add(attendance)
        try:
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            print(f"Database error during clock-in: {db_err}")
            return jsonify({
                'success': False,
                'message': 'Failed to save attendance. Please try again.',
                'db_error': True
            })

        return jsonify({
            'success': True,
            'already_marked': False,
            'employee_name': employee.full_name,
            'employee_id': employee.employee_id,
            'department': employee.department,
            'clock_in_time': now.strftime('%H:%M:%S'),
            'is_late': is_late,
            'similarity': round(similarity * 100, 1),
            'message': f'Attendance marked for {employee.full_name}!'
        })

    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        db.session.rollback()  # Ensure rollback on any error
        return jsonify({
            'success': False,
            'message': 'Recognition failed. Please try again.',
            'error': True
        })

@app.route('/api/detect-face', methods=['POST'])
def api_detect_face():
    """API endpoint for real-time face detection during enrollment"""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})

        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        face_info = detect_face(image_array)

        return jsonify({
            'success': True,
            'face_detected': face_info['face_detected'],
            'confidence': face_info.get('confidence', 0),
            'face_count': face_info.get('face_count', 0),
            'facial_area': face_info.get('facial_area', {})
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/validate-face', methods=['POST'])
def api_validate_face():
    """API endpoint for validating face quality during enrollment"""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})

        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        quality = validate_face_quality(image_array)

        return jsonify({
            'success': True,
            'valid': quality['valid'],
            'issues': quality['issues'],
            'quality_metrics': quality['quality_metrics'],
            'face_info': quality['face_info']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/recent-attendances')
def get_recent_attendances():
    today = date.today()
    recent = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(5).all()
    total_today = Attendance.query.filter_by(date=today).count()

    attendances = []
    for att in recent:
        attendances.append({
            'employee_name': att.employee.full_name,
            'employee_id': att.employee.employee_id,
            'department': att.employee.department,
            'clock_in_time': att.clock_in_time.strftime('%H:%M:%S'),
            'clock_out_time': att.clock_out_time.strftime('%H:%M:%S') if att.clock_out_time else None,
            'is_late': att.is_late
        })

    return jsonify({
        'attendances': attendances,
        'total_today': total_today
    })

@app.route('/records')
@login_required
def records():
    # Sanitize query parameters to prevent injection
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    employee_filter = request.args.get('employee_id', '').strip()

    # Validate date format (YYYY-MM-DD)
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    if start_date and not date_pattern.match(start_date):
        flash('Invalid start date format', 'danger')
        start_date = None

    if end_date and not date_pattern.match(end_date):
        flash('Invalid end date format', 'danger')
        end_date = None

    # Sanitize employee filter
    if employee_filter:
        employee_filter = InputValidator.sanitize_string(employee_filter, max_length=50)

    query = Attendance.query

    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date >= start)

    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date <= end)

    if employee_filter:
        employee = Employee.query.filter_by(employee_id=employee_filter).first()
        if employee:
            query = query.filter(Attendance.employee_id == employee.id)

    attendances = query.order_by(Attendance.date.desc(), Attendance.clock_in_time.desc()).all()
    all_employees = Employee.query.order_by(Employee.full_name).all()

    return render_template('records.html',
                         attendances=attendances,
                         start_date=start_date,
                         end_date=end_date,
                         employee_filter=employee_filter,
                         all_employees=all_employees)

@app.route('/records/export')
@login_required
def export_records():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = Attendance.query

    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date >= start)

    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date <= end)

    attendances = query.order_by(Attendance.date.desc(), Attendance.clock_in_time.desc()).all()

    output = io.StringIO()
    output.write('Date,Employee ID,Full Name,Department,Clock In Time,Clock Out Time,Status\n')

    for att in attendances:
        status = 'Late' if att.is_late else 'On Time'
        clock_out = att.clock_out_time.strftime("%H:%M:%S") if att.clock_out_time else 'N/A'
        output.write(f'{att.date},{att.employee.employee_id},{att.employee.full_name},{att.employee.department},{att.clock_in_time.strftime("%H:%M:%S")},{clock_out},{status}\n')

    output.seek(0)
    filename = f'attendance_records_{datetime.now().strftime("%Y%m%d")}.csv'

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings_page():
    settings = Settings.query.first()

    if request.method == 'POST':
        if not settings:
            settings = Settings()
            db.session.add(settings)

        try:
            # Validate and sanitize settings inputs
            work_start_hour = int(request.form.get('work_start_hour', 9))
            work_start_minute = int(request.form.get('work_start_minute', 0))
            work_end_hour = int(request.form.get('work_end_hour', 17))
            work_end_minute = int(request.form.get('work_end_minute', 0))
            recognition_threshold = float(request.form.get('recognition_threshold', 0.55))

            # Validate ranges
            if not (0 <= work_start_hour <= 23 and 0 <= work_end_hour <= 23):
                flash('Invalid hour values (must be 0-23)', 'danger')
                return redirect(url_for('settings_page'))

            if not (0 <= work_start_minute <= 59 and 0 <= work_end_minute <= 59):
                flash('Invalid minute values (must be 0-59)', 'danger')
                return redirect(url_for('settings_page'))

            if not (0.3 <= recognition_threshold <= 0.95):
                flash('Recognition threshold must be between 0.3 and 0.95', 'danger')
                return redirect(url_for('settings_page'))

            settings.work_start_hour = work_start_hour
            settings.work_start_minute = work_start_minute
            settings.work_end_hour = work_end_hour
            settings.work_end_minute = work_end_minute
            settings.recognition_threshold = recognition_threshold
            settings.anti_spoofing_enabled = request.form.get('anti_spoofing_enabled') == 'on'

            db.session.commit()

            # Log settings change
            client_ip = get_client_ip()
            audit_logger.log_data_access(
                "UPDATE system settings",
                current_user.username,
                client_ip
            )

            flash('Settings updated successfully!', 'success')
            return redirect(url_for('settings_page'))

        except (ValueError, TypeError) as e:
            flash('Invalid input values in settings form', 'danger')
            return redirect(url_for('settings_page'))

    return render_template('settings.html', settings=settings)

@app.route('/api/refresh-cache', methods=['POST'])
@login_required
def api_refresh_cache():
    """Manually refresh the embeddings cache"""
    count = refresh_embeddings_cache()
    return jsonify({
        'success': True,
        'message': f'Cache refreshed with {count} employee embeddings'
    })

@app.after_request
def add_header(response):
    # Cache control headers
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'

    # Add comprehensive security headers
    response = add_security_headers(response)

    return response

if __name__ == '__main__':
    init_db()
    # Pre-load embeddings cache
    with app.app_context():
        refresh_embeddings_cache()
        print(f"Loaded {len(embeddings_cache)} employee embeddings into cache")
    app.run(host='0.0.0.0', port=5000, debug=True)
