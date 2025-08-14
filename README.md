# EvalPro - Professional Teaching Competency Assessment Platform

## 🎓 Executive Summary

EvalPro is an enterprise-grade AI-powered assessment platform designed for evaluating teaching competencies in professional educational environments. The system leverages advanced machine learning models, computer vision, and natural language processing to provide rigorous, standardized assessments of teaching performance across multiple competency domains.

## 🏗️ System Architecture

### Overview
EvalPro follows a modern microservices-inspired architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  React-style Frontend │  RESTful API Gateway │  Dark UI Theme   │
│  • File Upload        │  • Request Routing   │  • Responsive    │
│  • Progress Tracking  │  • Error Handling    │  • Professional  │
│  • Results Display    │  • Rate Limiting     │  • Accessible    │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Flask Web Framework │  Assessment Engine    │  File Management │
│  • Route Handlers    │  • AI Orchestration   │  • Upload Logic  │
│  • Validation Logic  │  • Score Calculation  │  • File Security │
│  • Response Mapping  │  • Report Generation  │  • Cleanup Tasks │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  Video Analysis       │  Audio Processing     │  AI Integration  │
│  • Frame Extraction   │  • Speech Recognition │  • Claude 3.5    │
│  • Computer Vision    │  • Transcript Gen.    │  • Prompt Eng.   │
│  • Quality Assessment │  • Audio Quality      │  • Response Parse│
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
├─────────────────────────────────────────────────────────────────┤
│  File Storage         │  Assessment Data      │  Logging System  │
│  • Local File System │  • JSON Responses     │  • Error Tracking│
│  • Secure Upload Dir │  • Evaluation Results │  • Audit Trails  │
│  • Auto Cleanup      │  • Score Persistence  │  • Performance   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Features

- 🎯 **Professional Assessment Framework**: 10-domain competency evaluation model
- 🧠 **Advanced AI Analysis**: Claude 3.5 Sonnet with specialized educational prompting
- 📊 **Multi-Modal Evaluation**: Video, audio, and contextual analysis integration
- 🏆 **Certification Standards**: CERTIFIED/PROVISIONAL/NOT_CERTIFIED outcomes
- 🎨 **Enterprise UI/UX**: Dark professional theme with glassmorphism effects
- 🔒 **Enterprise Security**: Multi-layer validation and secure file handling
- 📱 **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- ⚡ **High-Performance Processing**: Supports up to 1GB video files
- 🌐 **API-First Architecture**: RESTful endpoints with comprehensive error handling
- 📈 **Professional Reporting**: Detailed assessment reports with evidence and recommendations

## 🚀 Installation & Setup

### Prerequisites

**System Requirements:**
- Python 3.9 or higher
- FFmpeg (for video processing)
- Internet connection (for AI API calls)
- Minimum 4GB RAM (8GB recommended)
- 10GB available storage space

**Required Accounts:**
- OpenRouter API account with credits
- Optional: Google Speech Recognition API key

### Quick Start Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd EvalPro
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install FFmpeg** (required for video processing)
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   
   # Windows (using chocolatey)
   choco install ffmpeg
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   # Required: OpenRouter API Configuration
   AI_API_KEY=your-openrouter-api-key-here
   AI_API_BASE_URL=https://openrouter.ai/api/v1
   AI_MODEL=anthropic/claude-3.5-sonnet
   
   # Application Settings
   SECRET_KEY=your-secure-random-secret-key-here
   FLASK_ENV=development
   
   # File Upload Settings
   UPLOAD_FOLDER=uploads
   MAX_CONTENT_LENGTH=1073741824  # 1GB in bytes
   
   # Optional: Custom Port
   PORT=5000
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```


#### Production Environment
```env
FLASK_ENV=production
FLASK_DEBUG=False
AI_API_KEY=sk-or-v1-your-production-key
SECRET_KEY=your-secure-production-secret-key
MAX_CONTENT_LENGTH=1073741824  # 1GB
```

## 🧠 AI Models & Assessment Approach

### Primary AI Model
- **Model**: Anthropic Claude 3.5 Sonnet
- **Provider**: OpenRouter AI API
- **Specialization**: Senior Educational Assessment Specialist (15+ years experience)
- **Context Window**: Extended for comprehensive multi-modal analysis
- **Temperature**: 0.7 (balanced creativity and consistency)

### Professional Assessment Framework

#### Multi-Modal Analysis Pipeline
```python
Video Upload → Frame Extraction → Audio Processing → AI Analysis → Score Calculation → Report Generation
     ↓              ↓                    ↓              ↓              ↓                ↓
File Validation  Computer Vision   Speech-to-Text   Claude 3.5    Professional    JSON Response
Security Check   Quality Analysis  Transcript Gen.   Assessment    Validation      & Frontend
```

#### 10-Domain Competency Model
```
Professional Teaching Competency Framework v2.0
├── Technical Competencies (30%)
│   ├── Technical Preparation & Setup (10%)
│   ├── Pedagogical Time Management (10%)
│   └── Adaptive Instruction & Flexibility (10%)
├── Content Competencies (40%)
│   ├── Content Mastery & Lesson Familiarity (15%)
│   ├── Instructional Design & Content Delivery (15%)
│   └── Language Proficiency & Communication (10%)
└── Professional Competencies (30%)
    ├── Professional Teaching Presence (10%)
    ├── Student Engagement & Interaction (10%)
    ├── Assessment & Feedback Delivery (5%)
    └── Professional Standards & Attitude (5%)
```

#### Certification Logic
```python
def calculate_certification_status(total_score, lowest_category_score):
    if total_score >= 60 and lowest_category_score >= 4:
        return "CERTIFIED"  # Ready for teaching
    elif total_score >= 55 and lowest_category_score >= 3:
        return "PROVISIONAL"  # Requires training
    else:
        return "NOT_CERTIFIED"  # Comprehensive retraining needed
```

## 🔧 Technical Specifications

### System Requirements

#### **Production Environment**
- **OS**: Linux (Ubuntu 20.04+ recommended) / macOS / Windows Server
- **Python**: 3.9+ with pip package manager
- **Memory**: Minimum 4GB RAM (8GB recommended for large files)
- **Storage**: 10GB+ available space for temporary file processing
- **Network**: High-speed internet for AI API calls

#### **Development Environment**
- **OS**: Any modern operating system
- **Python**: 3.9+ with virtual environment support
- **IDE**: VS Code, PyCharm, or similar with Python support
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

### Dependencies & Libraries

#### **Core Backend Dependencies**
```python
Flask==2.3.3              # Web framework
Werkzeug==2.3.7           # WSGI utilities
requests==2.31.0          # HTTP client for AI API
python-dotenv==1.0.0      # Environment variable management
```

#### **Media Processing Libraries**
```python
opencv-python==4.8.1.78   # Computer vision and frame extraction
moviepy==1.0.3            # Video processing and audio extraction
Pillow==10.0.1            # Image processing and manipulation
numpy==1.25.2             # Numerical computing for image arrays
speechrecognition==3.14.3 # Speech-to-text conversion
```

### File Processing Capabilities

#### **Supported Video Formats**
- **Primary**: MP4, AVI, MOV, WMV, FLV, WebM, MKV
- **Extended**: M4V, 3GP, OGV, MPEG, MP2, MXF
- **Maximum Size**: 1GB per file
- **Processing**: Automatic format detection and conversion

#### **Processing Pipeline**
1. **Upload Validation**: File type, size, and security checks
2. **Video Analysis**: Frame extraction (5 key frames) at optimal intervals
3. **Audio Extraction**: High-quality audio isolation using MoviePy
4. **Speech Processing**: Google Speech Recognition with error handling
5. **AI Analysis**: Multi-modal assessment using Claude 3.5 Sonnet
6. **Score Calculation**: Professional competency scoring algorithm
7. **Report Generation**: Comprehensive assessment report with recommendations

## 🚀 Installation & Setup

### Prerequisites

### Quick Start

1. **Clone or download the project files**
   ```bash
   # If you have the files, navigate to the EvalPro directory
   cd EvalPro
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you encounter issues with `pyaudio`, you may need to install system dependencies:
   - **macOS**: `brew install portaudio`
   - **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev`
   - **Windows**: PyAudio may require Visual Studio Build Tools

3. **Configure environment variables**
   Copy the example environment file and configure:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your settings:
   ```env
   AI_API_KEY=your-openrouter-api-key-here
   AI_API_BASE_URL=https://openrouter.ai/api/v1
   AI_MODEL=anthropic/claude-3.5-sonnet
   SECRET_KEY=your-secure-random-secret-key
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

### Production Deployment

For production deployment, consider:

1. **Environment Configuration**:
   - Set secure environment variables in `.env`
   - Use strong SECRET_KEY
   - Configure proper AI_API_KEY from OpenRouter
   - Set FLASK_ENV=production

2. **Use a production WSGI server** (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app --timeout 120
   ```

3. **Configure reverse proxy** (nginx/Apache) for better performance
4. **Set up proper logging and monitoring**
5. **Configure SSL/TLS certificates**
6. **Set up file cleanup for uploaded videos**

## Project Structure

```
EvalPro/
├── app.py                 # Main Flask application with AI integration
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── .env                  # Environment variables (not in git)
├── .env.example          # Environment template
├── templates/
│   └── index.html        # Frontend HTML template
└── uploads/              # Directory for uploaded videos (auto-created)
```


## Configuration

### AI Analysis Settings
- **AI Model**: Claude 3.5 Sonnet (configurable via AI_MODEL env var)
- **API Provider**: OpenRouter (supports multiple AI models)
- **Frame Analysis**: Extracts 5 key frames per video
- **Speech Recognition**: Google Speech Recognition API
- **Timeout**: 60 seconds for AI API calls

### File Upload Settings
- **Max file size**: 1GB (configurable via MAX_CONTENT_LENGTH environment variable)
- **Allowed formats**: MP4, AVI, MOV, WMV, FLV, WebM, MKV, M4V, 3GP, OGV, MPEG, MP2, MXF
- **Upload directory**: `uploads/` (auto-created with proper permissions)
- **Filename security**: Uses secure_filename() with UUID prefix for uniqueness
- **File validation**: Both MIME type and extension checking
- **Automatic cleanup**: Temporary files are automatically removed after processing

### Security Features
- Environment-based configuration
- Secure filename handling to prevent directory traversal
- File type validation (both MIME type and extension)
- Request size limiting
- API key protection via environment variables
- CSRF protection ready (Flask-WTF can be added)

## AI Analysis Features

### Current AI Integration

EvalPro now includes **real AI-powered analysis** using Claude 3.5 Sonnet:

1. **Visual Analysis**:
   - Extracts key frames from video using OpenCV
   - Analyzes body language, posture, and presentation skills
   - Evaluates visual elements like lighting and background

2. **Audio Analysis**:
   - Extracts audio track using MoviePy
   - Converts speech to text using Google Speech Recognition
   - Analyzes speech patterns, clarity, and content

3. **AI Evaluation**:
   - Sends visual frames and transcript to Claude 3.5 Sonnet
   - Receives comprehensive feedback on presentation skills
   - Generates scores and actionable improvement suggestions

### Supported AI Models

Via OpenRouter API, you can use different models by changing the `AI_MODEL` environment variable:
- `anthropic/claude-3.5-sonnet` (default, best for comprehensive analysis)
- `anthropic/claude-3-haiku` (faster, cost-effective)
- `openai/gpt-4-vision-preview` (alternative vision model)
- `google/gemini-pro-vision` (Google's multimodal model)

### Extending AI Capabilities

1. **Add more analysis features**:
   ```python
   # Emotion detection
   # Gesture analysis
   # Speaking pace analysis
   # Confidence scoring
   ```

2. **Custom prompts for specific training types**:
   - Sales presentations
   - Technical training
   - Public speaking
   - Interview preparation

3. **Consider async processing** for longer videos:
   ```bash
   pip install celery redis  # For background tasks
   ```

### Database Integration

To store evaluation results:

1. **Add database support**:
   ```bash
   pip install Flask-SQLAlchemy
   ```

2. **Create models for**:
   - Video metadata
   - Evaluation results
   - User sessions (if needed)

### Additional Features to Consider

- **User authentication**: Flask-Login
- **File management**: List/delete uploaded videos
- **Batch processing**: Multiple video evaluation
- **Export results**: PDF/Excel report generation
- **Video thumbnails**: Preview generation
- **Progress tracking**: WebSocket for real-time updates
