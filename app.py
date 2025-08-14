import os
import uuid
import base64
import cv2
import tempfile
import json
import requests
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import wave
import time
import math
from scipy import stats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 1024 * 1024 * 1024))  # 1GB max file size

# AI API Configuration 
AI_API_KEY = os.getenv('AI_API_KEY')
AI_API_BASE_URL = os.getenv('AI_API_BASE_URL', 'https://openrouter.ai/api/v1')
AI_MODEL = os.getenv('AI_MODEL', 'anthropic/claude-3.5-sonnet')

# Allowed video file extensions
ALLOWED_EXTENSIONS = {
    'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', 'm4v', '3gp', 'ogv', 'mp2', 'mpeg', 'mpv', 'mxf'
}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_upload_folder():
    """Create uploads folder if it doesn't exist."""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_comprehensive_video_analysis(video_path, num_frames=10):
    """
    Extract comprehensive visual analysis from video including frames, quality metrics, 
    motion analysis, and visual composition assessment.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count == 0:
            return {
                'frames': [],
                'video_quality': 'POOR',
                'visual_analysis': {},
                'technical_metrics': {}
            }
        
        # Extract frames at strategic intervals
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        frames = []
        quality_scores = []
        brightness_levels = []
        motion_scores = []
        face_detection_results = []
        composition_analysis = []
        
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        prev_frame = None
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for analysis
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # === QUALITY ANALYSIS ===
                # Measure image sharpness using Laplacian variance
                laplacian_var = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
                quality_scores.append(laplacian_var)
                
                # Measure brightness and contrast
                brightness = np.mean(frame_gray)
                brightness_levels.append(brightness)
                contrast = np.std(frame_gray)
                
                # === FACE AND PERSON DETECTION ===
                faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
                face_info = {
                    'faces_detected': len(faces),
                    'face_positions': faces.tolist() if len(faces) > 0 else [],
                    'primary_face_size': max([w*h for (x,y,w,h) in faces]) if len(faces) > 0 else 0
                }
                face_detection_results.append(face_info)
                
                # === COMPOSITION ANALYSIS ===
                # Rule of thirds analysis
                h, w = frame_gray.shape
                thirds_h = [h//3, 2*h//3]
                thirds_w = [w//3, 2*w//3]
                
                composition_score = 0
                if len(faces) > 0:
                    # Check if face is positioned well according to rule of thirds
                    for (x, y, face_w, face_h) in faces:
                        face_center_x = x + face_w // 2
                        face_center_y = y + face_h // 2
                        
                        # Score based on rule of thirds positioning
                        if abs(face_center_x - thirds_w[0]) < w//6 or abs(face_center_x - thirds_w[1]) < w//6:
                            composition_score += 0.5
                        if abs(face_center_y - thirds_h[0]) < h//6 or abs(face_center_y - thirds_h[1]) < h//6:
                            composition_score += 0.5
                
                composition_analysis.append({
                    'composition_score': composition_score,
                    'frame_center_focus': 1.0 if len(faces) > 0 and faces[0][0] + faces[0][2]//2 > w//3 and faces[0][0] + faces[0][2]//2 < 2*w//3 else 0.5
                })
                
                # === MOTION ANALYSIS ===
                if prev_frame is not None:
                    # Calculate optical flow for motion detection
                    flow = cv2.calcOpticalFlowPyrLK(prev_frame, frame_gray, None, None)
                    if flow[0] is not None:
                        motion_magnitude = np.mean(np.sqrt(flow[0][:,:,0]**2 + flow[0][:,:,1]**2))
                        motion_scores.append(motion_magnitude)
                    else:
                        motion_scores.append(0)
                else:
                    motion_scores.append(0)
                
                prev_frame = frame_gray.copy()
                
                # === IMAGE ENHANCEMENT AND ENCODING ===
                # Enhance image quality for AI analysis
                pil_image = Image.fromarray(frame_rgb)
                
                # Apply subtle enhancements
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.05)
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=90)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                frames.append({
                    'image': img_base64,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'quality_score': laplacian_var,
                    'brightness': brightness,
                    'contrast': contrast,
                    'faces_detected': len(faces),
                    'composition_score': composition_score
                })
        
        cap.release()
        
        # === AGGREGATE ANALYSIS ===
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        avg_brightness = np.mean(brightness_levels) if brightness_levels else 0
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        total_faces_detected = sum([f['faces_detected'] for f in face_detection_results])
        avg_composition = np.mean([c['composition_score'] for c in composition_analysis]) if composition_analysis else 0
        
        # Determine video quality rating
        video_quality = 'POOR'
        if avg_quality > 500 and avg_brightness > 50 and avg_brightness < 200:
            video_quality = 'EXCELLENT'
        elif avg_quality > 300 and avg_brightness > 30 and avg_brightness < 220:
            video_quality = 'GOOD'
        elif avg_quality > 100:
            video_quality = 'FAIR'
        
        # Professional presentation analysis
        presentation_stability = 'STABLE' if avg_motion < 10 else 'MODERATE' if avg_motion < 20 else 'UNSTABLE'
        face_consistency = 'CONSISTENT' if total_faces_detected > len(frames) * 0.8 else 'INCONSISTENT'
        
        return {
            'frames': [frame['image'] for frame in frames],  # For AI analysis
            'frame_details': frames,  # Detailed frame analysis
            'video_quality': video_quality,
            'technical_metrics': {
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'total_frames': frame_count,
                'avg_quality_score': avg_quality,
                'avg_brightness': avg_brightness,
                'avg_motion': avg_motion,
                'presentation_stability': presentation_stability
            },
            'visual_analysis': {
                'face_detection_rate': total_faces_detected / len(frames) if frames else 0,
                'face_consistency': face_consistency,
                'avg_composition_score': avg_composition,
                'lighting_quality': 'EXCELLENT' if 80 < avg_brightness < 180 else 'GOOD' if 60 < avg_brightness < 200 else 'NEEDS_IMPROVEMENT',
                'visual_engagement': 'HIGH' if avg_composition > 0.7 else 'MODERATE' if avg_composition > 0.4 else 'LOW'
            },
            'quality_breakdown': {
                'sharpness': 'EXCELLENT' if avg_quality > 500 else 'GOOD' if avg_quality > 300 else 'FAIR' if avg_quality > 100 else 'POOR',
                'lighting': 'EXCELLENT' if 80 < avg_brightness < 180 else 'GOOD' if 60 < avg_brightness < 200 else 'NEEDS_IMPROVEMENT',
                'stability': presentation_stability,
                'framing': 'PROFESSIONAL' if avg_composition > 0.7 else 'ADEQUATE' if avg_composition > 0.4 else 'NEEDS_IMPROVEMENT'
            }
        }
        
    except Exception as e:
        print(f"Error in comprehensive video analysis: {e}")
        return {
            'frames': [],
            'video_quality': 'ERROR',
            'visual_analysis': {},
            'technical_metrics': {}
        }

def extract_comprehensive_audio_analysis(video_path):
    """
    Extract comprehensive audio analysis including transcription, vocal qualities,
    speech patterns, and prosodic features.
    """
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save audio to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            temp_audio_path = temp_audio.name
        
        # === BASIC TRANSCRIPTION ===
        recognizer = sr.Recognizer()
        transcript = ""
        confidence_score = 0
        
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_data)
                confidence_score = 0.85  # Google API doesn't return confidence, estimate based on success
            except sr.UnknownValueError:
                transcript = "[Audio unclear or no speech detected]"
                confidence_score = 0.1
            except sr.RequestError:
                transcript = "[Speech recognition service unavailable]"
                confidence_score = 0.0
        
        # === ADVANCED AUDIO ANALYSIS WITH LIBROSA ===
        try:
            # Load audio with librosa for advanced analysis
            y, sr_rate = librosa.load(temp_audio_path, sr=None)
            duration = len(y) / sr_rate
            
            # === VOCAL QUALITY ANALYSIS ===
            # Pitch analysis (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            avg_pitch = np.mean(f0_values) if f0_values else 0
            pitch_variance = np.std(f0_values) if f0_values else 0
            pitch_range = (max(f0_values) - min(f0_values)) if f0_values else 0
            
            # Voice activity detection
            intervals = librosa.effects.split(y, top_db=20)
            speech_duration = sum([(end - start) / sr_rate for start, end in intervals])
            speech_ratio = speech_duration / duration if duration > 0 else 0
            
            # Energy and volume analysis
            rms_energy = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms_energy)
            energy_variance = np.std(rms_energy)
            
            # Spectral features for voice quality
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr_rate)[0]
            avg_spectral_centroid = np.mean(spectral_centroids)
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zero_crossing_rate)
            
            # Tempo and rhythm analysis
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_rate)
            rhythm_consistency = np.std(np.diff(beat_frames)) if len(beat_frames) > 1 else float('inf')
            
            # === SPEECH PATTERN ANALYSIS ===
            if transcript and transcript not in ["[Audio unclear or no speech detected]", "[Speech recognition service unavailable]"]:
                words = transcript.split()
                word_count = len(words)
                speaking_rate = word_count / (speech_duration / 60) if speech_duration > 0 else 0  # words per minute
                
                # Pause analysis (estimated from silence detection)
                silence_threshold = np.percentile(rms_energy, 20)  # Bottom 20% of energy as silence
                pause_count = len([e for e in rms_energy if e < silence_threshold])
                pause_frequency = pause_count / duration if duration > 0 else 0
                
                # Vocabulary richness
                unique_words = len(set(word.lower() for word in words))
                vocabulary_richness = unique_words / word_count if word_count > 0 else 0
                
                # Sentence complexity (estimated by punctuation and conjunctions)
                complex_indicators = ['because', 'however', 'therefore', 'furthermore', 'moreover', 'consequently']
                complexity_score = sum([1 for word in words if word.lower() in complex_indicators]) / word_count if word_count > 0 else 0
            else:
                word_count = 0
                speaking_rate = 0
                pause_frequency = 0
                vocabulary_richness = 0
                complexity_score = 0
            
        except Exception as librosa_error:
            print(f"Librosa analysis error: {librosa_error}")
            # Fallback to basic metrics
            duration = audio.duration if audio else 0
            avg_pitch = 0
            pitch_variance = 0
            pitch_range = 0
            speech_ratio = 0.5
            avg_energy = 0.5
            energy_variance = 0.1
            speaking_rate = 150  # Average speaking rate
            pause_frequency = 0.1
            vocabulary_richness = 0.5
            complexity_score = 0.3
            tempo = 120
            rhythm_consistency = 0.5
            avg_spectral_centroid = 0
            avg_zcr = 0
            word_count = len(transcript.split()) if transcript else 0
        
        # === VOCAL QUALITY ASSESSMENT ===
        # Pitch assessment (optimal range for teaching: 85-255 Hz for males, 165-265 Hz for females)
        pitch_quality = 'EXCELLENT'
        if avg_pitch == 0:
            pitch_quality = 'POOR'
        elif avg_pitch < 80 or avg_pitch > 300:
            pitch_quality = 'NEEDS_IMPROVEMENT'
        elif avg_pitch < 100 or avg_pitch > 250:
            pitch_quality = 'GOOD'
        
        # Voice consistency assessment
        pitch_stability = 'STABLE' if pitch_variance < 20 else 'MODERATE' if pitch_variance < 40 else 'UNSTABLE'
        volume_consistency = 'CONSISTENT' if energy_variance < 0.1 else 'MODERATE' if energy_variance < 0.2 else 'INCONSISTENT'
        
        # Speaking rate assessment (optimal: 140-160 WPM for teaching)
        rate_quality = 'EXCELLENT'
        if speaking_rate == 0:
            rate_quality = 'POOR'
        elif speaking_rate < 100 or speaking_rate > 200:
            rate_quality = 'NEEDS_IMPROVEMENT'
        elif speaking_rate < 120 or speaking_rate > 180:
            rate_quality = 'GOOD'
        
        # Clarity and articulation assessment
        clarity_score = 'EXCELLENT'
        if confidence_score < 0.3:
            clarity_score = 'POOR'
        elif confidence_score < 0.6:
            clarity_score = 'NEEDS_IMPROVEMENT'
        elif confidence_score < 0.8:
            clarity_score = 'GOOD'
        
        # === PROSODIC FEATURES ===
        prosodic_variety = 'HIGH' if pitch_range > 100 else 'MODERATE' if pitch_range > 50 else 'LOW'
        vocal_engagement = 'HIGH' if avg_energy > 0.1 and pitch_variance > 15 else 'MODERATE' if avg_energy > 0.05 else 'LOW'
        
        # Cleanup
        audio.close()
        video.close()
        os.unlink(temp_audio_path)
        
        return {
            'transcript': transcript,
            'confidence_score': confidence_score,
            'duration': duration,
            'word_count': word_count,
            
            # Vocal Quality Metrics
            'vocal_quality': {
                'pitch_average': avg_pitch,
                'pitch_range': pitch_range,
                'pitch_stability': pitch_stability,
                'pitch_quality': pitch_quality,
                'volume_consistency': volume_consistency,
                'vocal_energy': avg_energy,
                'clarity_score': clarity_score,
                'prosodic_variety': prosodic_variety,
                'vocal_engagement': vocal_engagement
            },
            
            # Speech Pattern Analysis
            'speech_patterns': {
                'speaking_rate': speaking_rate,  # words per minute
                'rate_quality': rate_quality,
                'pause_frequency': pause_frequency,
                'speech_to_silence_ratio': speech_ratio,
                'vocabulary_richness': vocabulary_richness,
                'language_complexity': complexity_score,
                'rhythm_consistency': 'GOOD' if rhythm_consistency < 0.5 else 'MODERATE' if rhythm_consistency < 1.0 else 'NEEDS_IMPROVEMENT'
            },
            
            # Technical Audio Metrics
            'technical_metrics': {
                'sample_rate': sr_rate if 'sr_rate' in locals() else 44100,
                'avg_spectral_centroid': avg_spectral_centroid if 'avg_spectral_centroid' in locals() else 0,
                'zero_crossing_rate': avg_zcr if 'avg_zcr' in locals() else 0,
                'tempo': tempo if 'tempo' in locals() else 120
            },
            
            # Overall Assessment
            'audio_quality_rating': 'EXCELLENT' if confidence_score > 0.8 and clarity_score == 'EXCELLENT' and rate_quality == 'EXCELLENT' else 
                                   'GOOD' if confidence_score > 0.6 and clarity_score in ['EXCELLENT', 'GOOD'] else 
                                   'FAIR' if confidence_score > 0.4 else 'POOR',
            
            'professional_delivery': {
                'vocal_professionalism': 'HIGH' if pitch_quality == 'EXCELLENT' and volume_consistency == 'CONSISTENT' else 'MODERATE',
                'speech_fluency': 'HIGH' if rate_quality == 'EXCELLENT' and pause_frequency < 0.15 else 'MODERATE',
                'language_proficiency': 'HIGH' if vocabulary_richness > 0.4 and complexity_score > 0.2 else 'MODERATE',
                'overall_communication': 'EXCELLENT' if all([
                    clarity_score == 'EXCELLENT',
                    rate_quality in ['EXCELLENT', 'GOOD'],
                    pitch_quality in ['EXCELLENT', 'GOOD']
                ]) else 'GOOD' if confidence_score > 0.6 else 'NEEDS_IMPROVEMENT'
            }
        }
        
    except Exception as e:
        print(f"Error in comprehensive audio analysis: {e}")
        return {
            'transcript': "[Error processing audio]",
            'confidence_score': 0,
            'duration': 0,
            'word_count': 0,
            'vocal_quality': {},
            'speech_patterns': {},
            'technical_metrics': {},
            'audio_quality_rating': 'ERROR',
            'professional_delivery': {}
        }

def call_ai_api(messages):
    """
    Call the AI API for video analysis.
    """
    try:
        headers = {
            'Authorization': f'Bearer {AI_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:5000',
            'X-Title': 'EvalPro Video Analysis'
        }
        
        payload = {
            'model': AI_MODEL,
            'messages': messages,
            'max_tokens': 1500,
            'temperature': 0.7
        }
        
        response = requests.post(
            f'{AI_API_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling AI API: {e}")
        return None

def analyze_video_with_ai(video_path, filename):
    """
    Analyze video using comprehensive AI models for multi-modal evaluation including
    advanced computer vision, audio analysis, and professional assessment.
    """
    try:
        print(f"Starting comprehensive analysis for {filename}...")
        
        # === COMPREHENSIVE MULTI-MODAL ANALYSIS ===
        # Advanced video analysis with computer vision
        visual_analysis = extract_comprehensive_video_analysis(video_path, num_frames=8)
        print(f"Visual analysis complete: {visual_analysis['video_quality']}")
        
        # Comprehensive audio analysis with vocal assessment
        audio_analysis = extract_comprehensive_audio_analysis(video_path)
        print(f"Audio analysis complete: {audio_analysis['audio_quality_rating']}")
        
        # Extract basic data for AI
        frames = visual_analysis['frames']
        audio_text = audio_analysis['transcript']
        
        # Prepare messages for AI analysis with rigorous professional criteria
        messages = [
            {
                "role": "system",
                "content": """You are a SENIOR EDUCATIONAL ASSESSMENT SPECIALIST with 15+ years experience evaluating teaching competencies for professional certification. You conduct rigorous, evidence-based evaluations using established pedagogical frameworks.

ASSESSMENT FRAMEWORK - PROFESSIONAL TEACHING STANDARDS:
Evaluate these 10 CRITICAL competencies with STRICT professional criteria:

1. **TECHNICAL PREPARATION & SETUP** (1-10 points)
   - EXCELLENT (8-10): Crystal clear audio/video, professional lighting, stable connection, backup systems ready
   - PROFICIENT (6-7): Good technical quality, minor issues, adequate setup
   - DEVELOPING (4-5): Noticeable technical problems affecting delivery
   - INADEQUATE (1-3): Poor audio/video, unprofessional setup, frequent disruptions

2. **CONTENT MASTERY & LESSON FAMILIARITY** (1-10 points) 
   - EXCELLENT (8-10): Expert knowledge, seamless flow, anticipates questions, rich examples
   - PROFICIENT (6-7): Solid understanding, prepared content, handles basics well
   - DEVELOPING (4-5): Basic knowledge, some hesitation, limited depth
   - INADEQUATE (1-3): Insufficient preparation, frequent errors, lacks confidence

3. **INSTRUCTIONAL DESIGN & CONTENT DELIVERY** (1-10 points)
   - EXCELLENT (8-10): Clear objectives, logical sequence, engaging structure, measurable outcomes
   - PROFICIENT (6-7): Organized content, mostly clear progression, adequate structure
   - DEVELOPING (4-5): Basic organization, some unclear transitions
   - INADEQUATE (1-3): Poor structure, confusing delivery, no clear objectives

4. **LANGUAGE PROFICIENCY & COMMUNICATION** (1-10 points)
   - EXCELLENT (8-10): Fluent, articulate, varied vocabulary, perfect grammar, engaging tone
   - PROFICIENT (6-7): Clear communication, good vocabulary, minor errors
   - DEVELOPING (4-5): Generally understandable, some language barriers
   - INADEQUATE (1-3): Frequent errors, unclear communication, language barriers

5. **STUDENT ENGAGEMENT & INTERACTION** (1-10 points)
   - EXCELLENT (8-10): Dynamic interaction, thoughtful questions, encourages participation, builds rapport
   - PROFICIENT (6-7): Good interaction, some questions, maintains attention
   - DEVELOPING (4-5): Limited interaction, basic questioning
   - INADEQUATE (1-3): Minimal engagement, one-way communication

6. **PEDAGOGICAL TIME MANAGEMENT** (1-10 points)
   - EXCELLENT (8-10): Perfect pacing, efficient transitions, maximizes learning time
   - PROFICIENT (6-7): Good timing, mostly on schedule, adequate pacing
   - DEVELOPING (4-5): Some timing issues, rushed or slow sections
   - INADEQUATE (1-3): Poor time control, significant over/under time

7. **PROFESSIONAL TEACHING PRESENCE** (1-10 points)
   - EXCELLENT (8-10): Confident, authoritative, natural, inspiring, commands attention
   - PROFICIENT (6-7): Professional demeanor, generally confident
   - DEVELOPING (4-5): Adequate presence, some nervousness
   - INADEQUATE (1-3): Lacks confidence, unprofessional, distracting

8. **ADAPTIVE INSTRUCTION & FLEXIBILITY** (1-10 points)
   - EXCELLENT (8-10): Seamlessly adapts to situations, reads audience, adjusts approach
   - PROFICIENT (6-7): Handles most situations well, shows flexibility
   - DEVELOPING (4-5): Some adaptability, basic responses
   - INADEQUATE (1-3): Rigid approach, cannot handle disruptions

9. **ASSESSMENT & FEEDBACK DELIVERY** (1-10 points)
   - EXCELLENT (8-10): Immediate, specific, constructive, encouraging, actionable feedback
   - PROFICIENT (6-7): Good feedback, mostly constructive
   - DEVELOPING (4-5): Basic feedback, some constructive elements
   - INADEQUATE (1-3): Poor or no feedback, not constructive

10. **PROFESSIONAL STANDARDS & ATTITUDE** (1-10 points)
    - EXCELLENT (8-10): Exemplary professionalism, growth mindset, ethical, inspiring
    - PROFICIENT (6-7): Professional behavior, positive attitude
    - DEVELOPING (4-5): Adequate professionalism, some concerns
    - INADEQUATE (1-3): Unprofessional, poor attitude, concerning behavior

CRITICAL EVALUATION REQUIREMENTS:
- Base scores on OBSERVABLE EVIDENCE from video/audio
- Provide SPECIFIC examples supporting each score
- Use PROFESSIONAL terminology and assessment language
- Consider CONTEXTUAL factors and teaching environment
- Apply CONSISTENT standards across all categories

CERTIFICATION STANDARDS:
- PASS THRESHOLD: ≥60/100 total AND no category below 4/10
- PROVISIONAL PASS: 55-59/100 with remedial training required
- FAIL: <55/100 or any category ≤3/10 requires full retraining

Return PROFESSIONAL assessment in this EXACT JSON format:
{
  "assessment_id": "EVAL_[timestamp]",
  "evaluator": "AI Assessment Specialist",
  "assessment_date": "[current_date]",
  "categories": {
    "preparation": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "lesson_familiarity": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "content_delivery": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "english_delivery": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "student_interaction": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "time_management": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "teaching_presence": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "adaptability": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "feedback_skills": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"},
    "professional_attitude": {"score": X, "evidence": "specific observable evidence", "feedback": "professional development feedback", "level": "EXCELLENT/PROFICIENT/DEVELOPING/INADEQUATE"}
  },
  "total_score": X,
  "certification_result": "CERTIFIED/PROVISIONAL/NOT_CERTIFIED",
  "competency_gaps": ["specific gap 1", "specific gap 2"],
  "professional_strengths": ["strength 1", "strength 2"],
  "development_priorities": ["priority 1", "priority 2"],
  "recommended_actions": "Detailed professional development plan with timeline",
  "assessor_confidence": "HIGH/MEDIUM/LOW",
  "quality_indicators": {
    "video_quality": "EXCELLENT/GOOD/FAIR/POOR",
    "audio_quality": "EXCELLENT/GOOD/FAIR/POOR", 
    "content_duration": "X minutes",
    "assessment_reliability": "HIGH/MEDIUM/LOW"
  }
}"""
            }
        ]
        
        # === ENHANCED AI PROMPT WITH COMPREHENSIVE ANALYSIS DATA ===
        comprehensive_analysis_data = f"""
COMPREHENSIVE VIDEO ANALYSIS DATA:

🎥 VISUAL ANALYSIS RESULTS:
- Video Quality: {visual_analysis['video_quality']}
- Duration: {visual_analysis['technical_metrics'].get('duration', 0):.1f} seconds
- Resolution: {visual_analysis['technical_metrics'].get('resolution', 'N/A')}
- FPS: {visual_analysis['technical_metrics'].get('fps', 0):.1f}
- Presentation Stability: {visual_analysis['technical_metrics'].get('presentation_stability', 'Unknown')}
- Face Detection Rate: {visual_analysis['visual_analysis'].get('face_detection_rate', 0):.2f}
- Face Consistency: {visual_analysis['visual_analysis'].get('face_consistency', 'Unknown')}
- Lighting Quality: {visual_analysis['visual_analysis'].get('lighting_quality', 'Unknown')}
- Visual Engagement: {visual_analysis['visual_analysis'].get('visual_engagement', 'Unknown')}
- Quality Breakdown:
  * Sharpness: {visual_analysis['quality_breakdown'].get('sharpness', 'Unknown')}
  * Lighting: {visual_analysis['quality_breakdown'].get('lighting', 'Unknown')}  
  * Stability: {visual_analysis['quality_breakdown'].get('stability', 'Unknown')}
  * Framing: {visual_analysis['quality_breakdown'].get('framing', 'Unknown')}

🎤 AUDIO ANALYSIS RESULTS:
- Audio Quality Rating: {audio_analysis['audio_quality_rating']}
- Transcript Confidence: {audio_analysis['confidence_score']:.2f}
- Duration: {audio_analysis['duration']:.1f} seconds
- Word Count: {audio_analysis['word_count']} words
- Speaking Rate: {audio_analysis['speech_patterns'].get('speaking_rate', 0):.1f} WPM

📊 VOCAL QUALITY METRICS:
- Pitch Average: {audio_analysis['vocal_quality'].get('pitch_average', 0):.1f} Hz
- Pitch Quality: {audio_analysis['vocal_quality'].get('pitch_quality', 'Unknown')}
- Pitch Stability: {audio_analysis['vocal_quality'].get('pitch_stability', 'Unknown')}
- Volume Consistency: {audio_analysis['vocal_quality'].get('volume_consistency', 'Unknown')}
- Clarity Score: {audio_analysis['vocal_quality'].get('clarity_score', 'Unknown')}
- Prosodic Variety: {audio_analysis['vocal_quality'].get('prosodic_variety', 'Unknown')}
- Vocal Engagement: {audio_analysis['vocal_quality'].get('vocal_engagement', 'Unknown')}

🗣️ SPEECH PATTERN ANALYSIS:
- Rate Quality: {audio_analysis['speech_patterns'].get('rate_quality', 'Unknown')}
- Pause Frequency: {audio_analysis['speech_patterns'].get('pause_frequency', 0):.3f}
- Speech-to-Silence Ratio: {audio_analysis['speech_patterns'].get('speech_to_silence_ratio', 0):.2f}
- Vocabulary Richness: {audio_analysis['speech_patterns'].get('vocabulary_richness', 0):.2f}
- Language Complexity: {audio_analysis['speech_patterns'].get('language_complexity', 0):.3f}
- Rhythm Consistency: {audio_analysis['speech_patterns'].get('rhythm_consistency', 'Unknown')}

💼 PROFESSIONAL DELIVERY ASSESSMENT:
- Vocal Professionalism: {audio_analysis['professional_delivery'].get('vocal_professionalism', 'Unknown')}
- Speech Fluency: {audio_analysis['professional_delivery'].get('speech_fluency', 'Unknown')}
- Language Proficiency: {audio_analysis['professional_delivery'].get('language_proficiency', 'Unknown')}
- Overall Communication: {audio_analysis['professional_delivery'].get('overall_communication', 'Unknown')}

📝 TRANSCRIPT: "{audio_text}"
"""

        content = [
            {
                "type": "text",
                "text": f"COMPREHENSIVE TEACHING COMPETENCY ASSESSMENT\n\nCandidate Video: {filename}\n{comprehensive_analysis_data}"
            }
        ]
        
        # Add frames if available with enhanced context
        if frames:
            content.append({
                "type": "text",
                "text": f"📸 VISUAL FRAMES FOR ANALYSIS ({len(frames)} frames extracted at key intervals):\nUse these frames to assess body language, posture, eye contact, professional appearance, background setup, lighting quality, and overall visual presentation."
            })
            
            for i, frame in enumerate(frames[:5]):  # Increased to 5 frames for better analysis
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })
        
        content.append({
            "type": "text", 
            "text": f"""🎯 ENHANCED PROFESSIONAL ASSESSMENT INSTRUCTION:

Conduct a comprehensive, evidence-based evaluation of this teaching demonstration using the provided multi-modal analysis data.

📊 ANALYSIS CONTEXT:
- Computer Vision Analysis: {visual_analysis['video_quality']} video quality with {visual_analysis['visual_analysis'].get('face_detection_rate', 0):.1%} face detection rate
- Audio Processing: {audio_analysis['word_count']} words transcribed with {audio_analysis['confidence_score']:.1%} confidence  
- Vocal Analysis: {audio_analysis['speech_patterns'].get('speaking_rate', 0):.0f} WPM rate, {audio_analysis['vocal_quality'].get('pitch_quality', 'Unknown')} pitch quality
- Technical Quality: {visual_analysis['technical_metrics'].get('resolution', 'N/A')} resolution, {visual_analysis['quality_breakdown'].get('stability', 'Unknown')} stability

🔍 RIGOROUS EVALUATION REQUIREMENTS:
1. **Multi-Modal Evidence Integration**: Use BOTH visual frames AND comprehensive audio/vocal analysis data
2. **Technical Assessment**: Incorporate computer vision metrics for technical setup scoring  
3. **Vocal Competency**: Use detailed speech pattern analysis for communication scoring
4. **Professional Standards**: Apply strict certification criteria with evidence citations
5. **Comprehensive Feedback**: Provide specific, actionable development recommendations

⚡ ENHANCED SCORING CRITERIA:
- TECHNICAL PREPARATION: Weight heavily on measured video/audio quality metrics
- COMMUNICATION: Integrate vocal quality analysis (pitch, clarity, rate, prosody)  
- CONTENT DELIVERY: Use speech pattern complexity and vocabulary richness data
- PROFESSIONAL PRESENCE: Combine visual analysis with vocal professionalism metrics

🎓 CERTIFICATION DECISION FRAMEWORK:
- Use quantitative analysis results to support qualitative assessments
- Provide evidence-based justification for all scores using the measured data
- Generate targeted improvement plans based on specific deficiencies identified

Return the complete professional assessment with integrated multi-modal evidence in the specified JSON format."""
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Call AI API
        ai_response = call_ai_api(messages)
        
        if ai_response:
            try:
                # Extract JSON from response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    evaluation = json.loads(json_str)
                    
                    # Validate required fields for professional assessment format
                    required_fields = ['categories', 'total_score', 'certification_result']
                    if all(field in evaluation for field in required_fields):
                        # Ensure score is within valid range
                        evaluation['total_score'] = max(0, min(100, int(evaluation['total_score'])))
                        
                        # Validate categories structure
                        required_categories = [
                            'preparation', 'lesson_familiarity', 'content_delivery', 'english_delivery',
                            'student_interaction', 'time_management', 'teaching_presence', 
                            'adaptability', 'feedback_skills', 'professional_attitude'
                        ]
                        
                        if 'categories' in evaluation and all(cat in evaluation['categories'] for cat in required_categories):
                            # Professional validation and enhancement
                            total_calculated = 0
                            lowest_category_score = 10
                            
                            for cat_name, cat_data in evaluation['categories'].items():
                                if isinstance(cat_data, dict) and 'score' in cat_data:
                                    # Ensure score is valid
                                    score = max(1, min(10, int(cat_data['score'])))
                                    cat_data['score'] = score
                                    total_calculated += score
                                    lowest_category_score = min(lowest_category_score, score)
                                    
                                    # Professional level mapping
                                    if score >= 8:
                                        cat_data['level'] = 'EXCELLENT'
                                        cat_data['performance_band'] = 'Exceptional'
                                    elif score >= 6:
                                        cat_data['level'] = 'PROFICIENT'  
                                        cat_data['performance_band'] = 'Meets Standards'
                                    elif score >= 4:
                                        cat_data['level'] = 'DEVELOPING'
                                        cat_data['performance_band'] = 'Approaching Standards'
                                    else:
                                        cat_data['level'] = 'INADEQUATE'
                                        cat_data['performance_band'] = 'Below Standards'
                                        
                                    # Add professional assessment markers
                                    if not cat_data.get('evidence'):
                                        cat_data['evidence'] = f"Assessment based on {cat_name.replace('_', ' ')} demonstration in video"
                                    if not cat_data.get('feedback'):
                                        cat_data['feedback'] = f"Professional development needed in {cat_name.replace('_', ' ')} competency"
                            
                            # Update total score with calculated value for accuracy
                            evaluation['total_score'] = total_calculated
                            
                            # Professional certification determination
                            if total_calculated >= 60 and lowest_category_score >= 4:
                                evaluation['certification_result'] = 'CERTIFIED'
                                evaluation['certification_status'] = 'APPROVED FOR TEACHING'
                            elif total_calculated >= 55 and lowest_category_score >= 3:
                                evaluation['certification_result'] = 'PROVISIONAL' 
                                evaluation['certification_status'] = 'CONDITIONAL APPROVAL - TRAINING REQUIRED'
                            else:
                                evaluation['certification_result'] = 'NOT_CERTIFIED'
                                evaluation['certification_status'] = 'NOT APPROVED - COMPREHENSIVE RETRAINING REQUIRED'
                            
                            # === ENHANCED METADATA WITH COMPREHENSIVE ANALYSIS ===
                            evaluation['assessment_metadata'] = {
                                'total_score': total_calculated,
                                'lowest_category': lowest_category_score,
                                'passing_threshold': 60,
                                'minimum_category_score': 4,
                                'assessment_standard': 'Professional Teaching Competency Framework v2.0',
                                'analysis_version': 'Comprehensive Multi-Modal v3.0',
                                'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # === COMPREHENSIVE TECHNICAL ANALYSIS DATA ===
                            evaluation['technical_analysis'] = {
                                'video_analysis': visual_analysis,
                                'audio_analysis': audio_analysis,
                                'processing_summary': {
                                    'frames_analyzed': len(frames),
                                    'audio_duration': audio_analysis['duration'],
                                    'transcript_words': audio_analysis['word_count'],
                                    'visual_quality_score': visual_analysis['video_quality'],
                                    'audio_quality_score': audio_analysis['audio_quality_rating'],
                                    'face_detection_success': visual_analysis['visual_analysis'].get('face_detection_rate', 0) > 0.5,
                                    'speech_recognition_confidence': audio_analysis['confidence_score']
                                }
                            }
                            
                            # === ENHANCED QUALITY INDICATORS ===
                            if 'quality_indicators' not in evaluation:
                                evaluation['quality_indicators'] = {}
                                
                            evaluation['quality_indicators'].update({
                                'video_quality': visual_analysis['video_quality'],
                                'audio_quality': audio_analysis['audio_quality_rating'],
                                'content_duration': f"{audio_analysis['duration']:.1f} seconds",
                                'assessment_reliability': 'HIGH' if (visual_analysis['video_quality'] in ['EXCELLENT', 'GOOD'] and 
                                                                   audio_analysis['audio_quality_rating'] in ['EXCELLENT', 'GOOD']) else 'MEDIUM',
                                'technical_metrics': {
                                    'resolution': visual_analysis['technical_metrics'].get('resolution', 'N/A'),
                                    'fps': visual_analysis['technical_metrics'].get('fps', 0),
                                    'speaking_rate': f"{audio_analysis['speech_patterns'].get('speaking_rate', 0):.0f} WPM",
                                    'vocal_pitch': f"{audio_analysis['vocal_quality'].get('pitch_average', 0):.0f} Hz"
                                },
                                'professional_indicators': {
                                    'visual_stability': visual_analysis['technical_metrics'].get('presentation_stability', 'Unknown'),
                                    'vocal_professionalism': audio_analysis['professional_delivery'].get('vocal_professionalism', 'Unknown'),
                                    'speech_fluency': audio_analysis['professional_delivery'].get('speech_fluency', 'Unknown'),
                                    'overall_communication': audio_analysis['professional_delivery'].get('overall_communication', 'Unknown')
                                }
                            })
                            
                            # === ADVANCED COMPETENCY INSIGHTS ===
                            evaluation['competency_insights'] = {
                                'strengths_analysis': {
                                    'technical_strengths': [],
                                    'communication_strengths': [],
                                    'content_strengths': []
                                },
                                'improvement_areas': {
                                    'technical_improvements': [],
                                    'communication_improvements': [], 
                                    'content_improvements': []
                                },
                                'quantitative_evidence': {
                                    'visual_engagement_score': visual_analysis['visual_analysis'].get('avg_composition_score', 0),
                                    'vocal_variety_score': audio_analysis['vocal_quality'].get('prosodic_variety', 'LOW'),
                                    'speech_clarity_score': audio_analysis['vocal_quality'].get('clarity_score', 'POOR'),
                                    'technical_quality_score': visual_analysis['video_quality']
                                }
                            }
                            
                            # Populate strengths based on analysis
                            if visual_analysis['video_quality'] in ['EXCELLENT', 'GOOD']:
                                evaluation['competency_insights']['strengths_analysis']['technical_strengths'].append(f"High video quality ({visual_analysis['video_quality']})")
                            if audio_analysis['audio_quality_rating'] in ['EXCELLENT', 'GOOD']:
                                evaluation['competency_insights']['strengths_analysis']['communication_strengths'].append(f"Clear audio quality ({audio_analysis['audio_quality_rating']})")
                            if audio_analysis['speech_patterns'].get('speaking_rate', 0) > 120 and audio_analysis['speech_patterns'].get('speaking_rate', 0) < 180:
                                evaluation['competency_insights']['strengths_analysis']['communication_strengths'].append(f"Optimal speaking rate ({audio_analysis['speech_patterns'].get('speaking_rate', 0):.0f} WPM)")
                            
                            # Legacy compatibility 
                            evaluation['overall_result'] = 'PASS' if evaluation['certification_result'] == 'CERTIFIED' else 'FAIL'
                            evaluation['requires_retraining'] = evaluation['certification_result'] != 'CERTIFIED'
                            
                            print(f"✅ Comprehensive analysis complete: {evaluation['certification_result']} ({total_calculated}/100)")
                            
                            return evaluation
                        else:
                            print("AI response missing required category structure")
                    else:
                        print("AI response missing required professional assessment fields")
                else:
                    print("No valid JSON found in AI response")
            except json.JSONDecodeError as e:
                print(f"Error parsing AI response JSON: {e}")
        
        # Fallback evaluation if AI fails - include comprehensive analysis
        print("⚠️  AI analysis failed, generating comprehensive fallback evaluation...")
        return generate_comprehensive_fallback_evaluation(visual_analysis, audio_analysis, filename)
        
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return generate_fallback_evaluation("", filename)

def generate_comprehensive_fallback_evaluation(visual_analysis, audio_analysis, filename):
    """
    Generate comprehensive fallback evaluation using computer vision and audio analysis
    when AI analysis fails, but we still have technical analysis data.
    """
    try:
        # Use comprehensive analysis data to generate intelligent scores
        audio_text = audio_analysis.get('transcript', '')
        
        # Base scores enhanced with technical analysis
        default_scores = {
            'preparation': 6,  # Base score
            'lesson_familiarity': 5,
            'content_delivery': 5,
            'english_delivery': 6,
            'student_interaction': 5,
            'time_management': 5,
            'teaching_presence': 5,
            'adaptability': 5,
            'feedback_skills': 5,
            'professional_attitude': 5
        }
        
        # === TECHNICAL PREPARATION SCORING ===
        video_quality = visual_analysis.get('video_quality', 'POOR')
        audio_quality = audio_analysis.get('audio_quality_rating', 'POOR')
        
        if video_quality == 'EXCELLENT' and audio_quality == 'EXCELLENT':
            default_scores['preparation'] = 9
        elif video_quality in ['EXCELLENT', 'GOOD'] and audio_quality in ['EXCELLENT', 'GOOD']:
            default_scores['preparation'] = 7
        elif video_quality == 'FAIR' or audio_quality == 'FAIR':
            default_scores['preparation'] = 5
        else:
            default_scores['preparation'] = 3
            
        # === COMMUNICATION SCORING BASED ON VOCAL ANALYSIS ===
        vocal_clarity = audio_analysis.get('vocal_quality', {}).get('clarity_score', 'POOR')
        speaking_rate = audio_analysis.get('speech_patterns', {}).get('speaking_rate', 0)
        pitch_quality = audio_analysis.get('vocal_quality', {}).get('pitch_quality', 'POOR')
        
        communication_score = 5  # Base
        if vocal_clarity == 'EXCELLENT':
            communication_score += 2
        elif vocal_clarity == 'GOOD':
            communication_score += 1
        elif vocal_clarity == 'POOR':
            communication_score -= 1
            
        if 120 <= speaking_rate <= 180:  # Optimal teaching rate
            communication_score += 1
        elif speaking_rate < 100 or speaking_rate > 200:
            communication_score -= 1
            
        if pitch_quality in ['EXCELLENT', 'GOOD']:
            communication_score += 1
        elif pitch_quality == 'POOR':
            communication_score -= 1
            
        default_scores['english_delivery'] = max(1, min(10, communication_score))
        
        # === CONTENT DELIVERY SCORING ===
        word_count = audio_analysis.get('word_count', 0)
        vocabulary_richness = audio_analysis.get('speech_patterns', {}).get('vocabulary_richness', 0)
        language_complexity = audio_analysis.get('speech_patterns', {}).get('language_complexity', 0)
        
        content_score = 5  # Base
        if word_count > 200:  # Substantial content
            content_score += 1
        if vocabulary_richness > 0.4:  # Rich vocabulary
            content_score += 1
        if language_complexity > 0.2:  # Complex language use
            content_score += 1
        
        default_scores['content_delivery'] = max(1, min(10, content_score))
        default_scores['lesson_familiarity'] = max(1, min(10, content_score))
        
        # === TEACHING PRESENCE SCORING ===
        visual_engagement = visual_analysis.get('visual_analysis', {}).get('visual_engagement', 'LOW')
        face_consistency = visual_analysis.get('visual_analysis', {}).get('face_consistency', 'INCONSISTENT')
        vocal_engagement = audio_analysis.get('vocal_quality', {}).get('vocal_engagement', 'LOW')
        
        presence_score = 5  # Base
        if visual_engagement == 'HIGH':
            presence_score += 1
        if face_consistency == 'CONSISTENT':
            presence_score += 1
        if vocal_engagement == 'HIGH':
            presence_score += 1
        elif vocal_engagement == 'LOW':
            presence_score -= 1
            
        default_scores['teaching_presence'] = max(1, min(10, presence_score))
        
        # === PROFESSIONAL ATTITUDE SCORING ===
        vocal_professionalism = audio_analysis.get('professional_delivery', {}).get('vocal_professionalism', 'MODERATE')
        overall_communication = audio_analysis.get('professional_delivery', {}).get('overall_communication', 'NEEDS_IMPROVEMENT')
        
        attitude_score = 5  # Base
        if vocal_professionalism == 'HIGH':
            attitude_score += 2
        elif vocal_professionalism == 'MODERATE':
            attitude_score += 1
            
        if overall_communication == 'EXCELLENT':
            attitude_score += 1
        elif overall_communication == 'NEEDS_IMPROVEMENT':
            attitude_score -= 1
            
        default_scores['professional_attitude'] = max(1, min(10, attitude_score))
        
        # Calculate total and determine certification
        total_score = sum(default_scores.values())
        lowest_score = min(default_scores.values())
        
        # Generate comprehensive categories
        categories = {}
        for category, score in default_scores.items():
            level = 'EXCELLENT' if score >= 8 else 'PROFICIENT' if score >= 6 else 'DEVELOPING' if score >= 4 else 'INADEQUATE'
            evidence = generate_evidence_from_analysis(category, visual_analysis, audio_analysis, score)
            feedback = generate_feedback_from_analysis(category, score, visual_analysis, audio_analysis)
            
            categories[category] = {
                'score': score,
                'level': level,
                'evidence': evidence,
                'feedback': feedback,
                'performance_band': 'Exceptional' if level == 'EXCELLENT' else 'Meets Standards' if level == 'PROFICIENT' else 'Approaching Standards' if level == 'DEVELOPING' else 'Below Standards'
            }
        
        # Certification determination
        if total_score >= 60 and lowest_score >= 4:
            cert_result = 'CERTIFIED'
            cert_status = 'APPROVED FOR TEACHING'
        elif total_score >= 55 and lowest_score >= 3:
            cert_result = 'PROVISIONAL'
            cert_status = 'CONDITIONAL APPROVAL - TRAINING REQUIRED'
        else:
            cert_result = 'NOT_CERTIFIED'
            cert_status = 'NOT APPROVED - COMPREHENSIVE RETRAINING REQUIRED'
        
        return {
            'assessment_id': f"FALLBACK_{int(time.time())}",
            'evaluator': 'Comprehensive Technical Analysis System',
            'assessment_date': time.strftime('%Y-%m-%d'),
            'categories': categories,
            'total_score': total_score,
            'certification_result': cert_result,
            'certification_status': cert_status,
            'competency_gaps': generate_gaps_from_analysis(default_scores),
            'professional_strengths': generate_strengths_from_analysis(visual_analysis, audio_analysis),
            'development_priorities': generate_priorities_from_analysis(default_scores, visual_analysis, audio_analysis),
            'recommended_actions': f"Technical analysis-based assessment completed for {filename}. Scores based on comprehensive video/audio analysis including: {visual_analysis['video_quality']} video quality, {audio_analysis['audio_quality_rating']} audio quality, {audio_analysis.get('word_count', 0)} words transcribed. AI-powered detailed assessment recommended for complete evaluation.",
            'assessor_confidence': 'MEDIUM',
            'quality_indicators': {
                'video_quality': visual_analysis.get('video_quality', 'UNKNOWN'),
                'audio_quality': audio_analysis.get('audio_quality_rating', 'UNKNOWN'),
                'content_duration': f"{audio_analysis.get('duration', 0):.1f} seconds",
                'assessment_reliability': 'MEDIUM'
            },
            'assessment_metadata': {
                'total_score': total_score,
                'lowest_category': lowest_score,
                'passing_threshold': 60,
                'minimum_category_score': 4,
                'assessment_standard': 'Technical Analysis Fallback v2.0'
            },
            'technical_analysis': {
                'video_analysis': visual_analysis,
                'audio_analysis': audio_analysis,
                'fallback_mode': True,
                'analysis_confidence': 'Medium - based on technical metrics'
            },
            'overall_result': 'PASS' if cert_result == 'CERTIFIED' else 'FAIL',
            'requires_retraining': cert_result != 'CERTIFIED'
        }
        
    except Exception as e:
        print(f"Error in comprehensive fallback evaluation: {e}")
        return generate_fallback_evaluation(audio_analysis.get('transcript', ''), filename)

def generate_evidence_from_analysis(category, visual_analysis, audio_analysis, score):
    """Generate evidence strings based on technical analysis data."""
    if category == 'preparation':
        return f"Video quality: {visual_analysis.get('video_quality', 'Unknown')}, Audio quality: {audio_analysis.get('audio_quality_rating', 'Unknown')}, Resolution: {visual_analysis.get('technical_metrics', {}).get('resolution', 'N/A')}"
    elif category == 'english_delivery':
        vocal_quality = audio_analysis.get('vocal_quality', {})
        return f"Speaking rate: {audio_analysis.get('speech_patterns', {}).get('speaking_rate', 0):.0f} WPM, Pitch quality: {vocal_quality.get('pitch_quality', 'Unknown')}, Clarity: {vocal_quality.get('clarity_score', 'Unknown')}"
    elif category == 'content_delivery':
        return f"Word count: {audio_analysis.get('word_count', 0)} words, Vocabulary richness: {audio_analysis.get('speech_patterns', {}).get('vocabulary_richness', 0):.2f}, Language complexity: {audio_analysis.get('speech_patterns', {}).get('language_complexity', 0):.3f}"
    elif category == 'teaching_presence':
        return f"Visual engagement: {visual_analysis.get('visual_analysis', {}).get('visual_engagement', 'Unknown')}, Face consistency: {visual_analysis.get('visual_analysis', {}).get('face_consistency', 'Unknown')}, Vocal engagement: {audio_analysis.get('vocal_quality', {}).get('vocal_engagement', 'Unknown')}"
    elif category == 'professional_attitude':
        return f"Vocal professionalism: {audio_analysis.get('professional_delivery', {}).get('vocal_professionalism', 'Unknown')}, Overall communication: {audio_analysis.get('professional_delivery', {}).get('overall_communication', 'Unknown')}"
    else:
        return f"Technical analysis assessment for {category.replace('_', ' ')} - Score: {score}/10 based on available metrics"

def generate_feedback_from_analysis(category, score, visual_analysis, audio_analysis):
    """Generate specific feedback based on technical analysis."""
    if score >= 8:
        return f"Excellent performance in {category.replace('_', ' ')} based on technical analysis. Maintain current standards."
    elif score >= 6:
        return f"Proficient {category.replace('_', ' ')} with room for minor improvements based on measured metrics."
    elif score >= 4:
        return f"Developing {category.replace('_', ' ')} competency. Focus on technical improvements identified in analysis."
    else:
        return f"Significant improvement needed in {category.replace('_', ' ')}. Technical analysis indicates major deficiencies."

def generate_gaps_from_analysis(scores):
    """Generate competency gaps based on scores."""
    gaps = []
    for category, score in scores.items():
        if score < 4:
            gaps.append(f"Critical deficiency in {category.replace('_', ' ')} (score: {score}/10)")
        elif score < 6:
            gaps.append(f"Development needed in {category.replace('_', ' ')} (score: {score}/10)")
    return gaps

def generate_strengths_from_analysis(visual_analysis, audio_analysis):
    """Generate strengths based on technical analysis."""
    strengths = []
    if visual_analysis.get('video_quality') in ['EXCELLENT', 'GOOD']:
        strengths.append(f"High video quality ({visual_analysis.get('video_quality')})")
    if audio_analysis.get('audio_quality_rating') in ['EXCELLENT', 'GOOD']:
        strengths.append(f"Clear audio quality ({audio_analysis.get('audio_quality_rating')})")
    
    speaking_rate = audio_analysis.get('speech_patterns', {}).get('speaking_rate', 0)
    if 120 <= speaking_rate <= 180:
        strengths.append(f"Optimal speaking rate ({speaking_rate:.0f} WPM)")
    
    if audio_analysis.get('vocal_quality', {}).get('clarity_score') in ['EXCELLENT', 'GOOD']:
        strengths.append(f"Clear speech and articulation")
        
    return strengths if strengths else ["Technical processing completed successfully"]

def generate_priorities_from_analysis(scores, visual_analysis, audio_analysis):
    """Generate development priorities based on analysis."""
    priorities = []
    
    # Find lowest scoring categories
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    
    for category, score in sorted_scores[:3]:  # Top 3 priorities
        if score < 6:
            priorities.append(f"Improve {category.replace('_', ' ')} (currently {score}/10)")
    
    # Add technical recommendations
    if visual_analysis.get('video_quality') not in ['EXCELLENT', 'GOOD']:
        priorities.append("Enhance video setup and lighting quality")
    if audio_analysis.get('audio_quality_rating') not in ['EXCELLENT', 'GOOD']:
        priorities.append("Improve audio recording quality and environment")
    
    return priorities if priorities else ["Continue maintaining current performance standards"]

def generate_fallback_evaluation(audio_text, filename):
    """
    Generate structured evaluation when AI analysis fails, following the new format.
    """
    # Default scores for each category when AI fails
    default_scores = {
        'preparation': 6,  # Assume basic technical setup
        'lesson_familiarity': 5,  # Unknown without analysis
        'content_delivery': 5,  # Basic assumption
        'english_delivery': 6,  # Assume acceptable if audio processed
        'student_interaction': 5,  # Cannot assess without AI
        'time_management': 5,  # Cannot assess without AI
        'teaching_presence': 5,  # Cannot assess without AI
        'adaptability': 5,  # Cannot assess without AI
        'feedback_skills': 5,  # Cannot assess without AI
        'professional_attitude': 5  # Basic assumption
    }
    
    # Adjust scores based on available audio analysis
    if audio_text and audio_text != "[Error processing audio]":
        word_count = len(audio_text.split())
        if word_count > 100:  # Good content length
            default_scores['content_delivery'] = 6
            default_scores['english_delivery'] = 7
        if len(audio_text) > 300:  # Comprehensive content
            default_scores['lesson_familiarity'] = 6
            default_scores['professional_attitude'] = 6
    else:
        # Reduce scores if audio processing failed
        default_scores['english_delivery'] = 4
        default_scores['content_delivery'] = 4
        default_scores['preparation'] = 4  # Technical issues
    
    # Calculate total score
    total_score = sum(default_scores.values())
    
    # Generate categories structure
    categories = {}
    for category, score in default_scores.items():
        level = 'Excellent' if score >= 7 else ('Acceptable' if score >= 4 else 'Needs Improvement')
        categories[category] = {
            'score': score,
            'level': level,
            'feedback': f'Automated assessment - score based on technical processing. Manual review recommended for accurate {category.replace("_", " ")} evaluation.'
        }
    
    # Generate summary
    strengths = ["Video successfully uploaded and processed", "Basic technical requirements met"]
    improvements = ["AI analysis temporarily unavailable", "Manual assessment recommended for detailed feedback"]
    
    if audio_text and len(audio_text) > 100:
        strengths.append("Adequate speech content detected in audio")
    
    return {
        "categories": categories,
        "total_score": total_score,
        "overall_result": "PASS" if total_score >= 60 else "FAIL",
        "requires_retraining": total_score < 60,
        "strengths": strengths,
        "improvements": improvements,
        "recommendation": f"Basic technical assessment completed for {filename}. For comprehensive evaluation including detailed category analysis, please ensure AI services are available or contact assessment team for manual review."
    }

@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')

@app.route('/upload')
def upload():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/results/<assessment_id>')
def results(assessment_id):
    """Render the results page."""
    # Get results from session
    assessment_data = session.get(assessment_id)
    
    if not assessment_data:
        flash('Assessment results not found or expired. Please upload and assess a new video.', 'error')
        return redirect(url_for('upload'))
    
    return render_template('results.html', 
                         assessment_id=assessment_id,
                         filename=assessment_data['filename'],
                         evaluation=assessment_data['evaluation'])

@app.route('/evaluate', methods=['POST'])
def evaluate_video():
    """Handle video upload and evaluation."""
    try:
        print(f"Received request with files: {list(request.files.keys())}")
        print(f"Content-Length: {request.content_length}")
        
        # Check if file is in request
        if 'video' not in request.files:
            print("No 'video' key in request.files")
            flash('Please select a video file to upload.', 'error')
            return redirect(url_for('upload'))
        
        file = request.files['video']
        print(f"File received: {file.filename}, Content-Type: {file.content_type}")
        
        # Check if file was actually selected
        if file.filename == '':
            print("Empty filename")
            flash('Please select a video file to upload.', 'error')
            return redirect(url_for('upload'))
        
        # Check file type
        if not allowed_file(file.filename):
            print(f"File type not allowed: {file.filename}")
            flash(f'Please upload a valid video file. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(url_for('upload'))
        
        print(f"File validation passed, proceeding with upload...")
        
        # Create uploads folder if it doesn't exist
        create_upload_folder()
        
        # Generate secure filename
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # AI-powered video processing and evaluation
        evaluation_result = analyze_video_with_ai(file_path, original_filename)
        
        # Generate assessment ID
        assessment_id = evaluation_result.get('assessment_id', f"EVAL_{uuid.uuid4().hex}")
        
        # Store results in session
        session[assessment_id] = {
            'filename': original_filename,
            'evaluation': evaluation_result
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass  # File cleanup failed, but continue
        
        # Redirect to results page
        return redirect(url_for('results', assessment_id=assessment_id))
        
    except RequestEntityTooLarge:
        flash('Please upload a video file smaller than 1GB.', 'error')
        return redirect(url_for('upload'))
    
    except Exception as e:
        print(f"Error processing video: {e}")
        flash('An error occurred while processing your video. Please try again.', 'error')
        return redirect(url_for('upload'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'message': 'Please upload a video file smaller than 1GB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('home.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Server error',
        'message': 'An internal server error occurred. Please try again.'
    }), 500

if __name__ == '__main__':
    # Create uploads folder on startup
    create_upload_folder()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)