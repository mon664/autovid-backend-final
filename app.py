#!/usr/bin/env python3
"""
AutoVid Full Backend API - Flask 버전
ai-platform-clean 통합
InfiniCloud WebDAV 연동
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from webdav3.client import Client
import google.generativeai as genai
from google.cloud import texttospeech
import replicate
import os
import subprocess
import uuid
import tempfile
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
import io
import base64

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://ai-platform-clean.vercel.app"]}})

# ===== 환경 변수 설정 =====
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# InfiniCloud WebDAV 설정
WEBDAV_CONFIG = {
    'webdav_hostname': "https://rausu.infini-cloud.net/dav/",
    'webdav_login': "hhtsta",
    'webdav_password': "RXYf3uYhCbL9Ezwa"
}
webdav_client = Client(WEBDAV_CONFIG)

# Google API 설정
genai.configure(api_key=GOOGLE_API_KEY)
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ===== 8개 비디오 템플릿 =====
TEMPLATES = {
    "BLACK": {
        "id": "9fa9a756-3374-49fb-80db-e7f53178f547",
        "name": "BLACK DEFAULT",
        "background_color": "#FF000000",
        "top_height": 15.0,
        "bottom_height": 15.0
    },
    "WHITE": {
        "id": "9fa9a756-3374-49fb-80db-e7f53178f548",
        "name": "WHITE DEFAULT",
        "background_color": "#FFFFFFFF",
        "top_height": 15.0,
        "bottom_height": 15.0
    },
    "BEIGE_BROWN": {
        "id": "789b4b30-93a7-46ed-b528-f546017844f1",
        "name": "BeigeBrown",
        "background_color": "#FFFFFBE5",
        "top_height": 32.0,
        "bottom_height": 7.0
    },
    "BEIGE_RED": {"id": "789b4b30-93a7-46ed-b528-f546017844f2", "name": "BeigeRed"},
    "BLACK_PINK": {"id": "789b4b30-93a7-46ed-b528-f546017844f3", "name": "BlackPink"},
    "WHITE_BLUE": {"id": "789b4b30-93a7-46ed-b528-f546017844f4", "name": "WhiteBlue"},
    "WHITE_GREEN": {"id": "789b4b30-93a7-46ed-b528-f546017844f5", "name": "WhiteGreen"},
    "WHITE_RED": {"id": "789b4b30-93a7-46ed-b528-f546017844f6", "name": "WhiteRed"}
}

# ===== 55개 FFmpeg 전환 효과 =====
FFMPEG_TRANSITIONS = [
    "fade", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "pixelize", "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice", "hblur",
    "dissolve", "radial", "circle", "rect",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "zoomin", "zoomout", "rotate", "spin", "shrink", "grow",
    "swirl", "wind", "wave", "mosaic", "grid",
    "checkerboard", "blinds", "curtain", "crossfade", "mix",
    "add", "subtract", "multiply", "screen", "overlay",
    "softlight", "hardlight", "colordodge", "colorburn",
    "darken", "lighten", "difference", "exclusion"
]

# ===== 6개 AI 이미지 모델 =====
IMAGE_MODELS = {
    "animagine31": "cjwbw/animagine-xl-3.1",
    "chibitoon": "fofr/sdxl-chibi",
    "enna-sketch": "replicate/text-to-image",
    "flux-dark": "black-forest-labs/flux-schnell",
    "flux-realistic": "black-forest-labs/flux-dev",
    "flux-webtoon": "custom/flux-webtoon"
}

# ===== 유틸리티 함수 =====
def upload_to_webdav(file_content, filename):
    """InfiniCloud WebDAV에 파일 업로드"""
    try:
        file_object = io.BytesIO(file_content if isinstance(file_content, bytes) else file_content.encode())
        remote_path = f"/autovid/{filename}"
        webdav_client.upload(remote_path, file_object)
        return f"{WEBDAV_CONFIG['webdav_hostname'].rstrip('/')}{remote_path}"
    except Exception as e:
        print(f"WebDAV 업로드 실패: {e}")
        return None

def generate_script(subject, request_number=5):
    """Gemini로 스크립트 생성"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
당신은 전문 비디오 스크립트 작성자입니다.

주제: {subject}
장면 개수: {request_number}

다음 JSON 형식으로 정확히 응답하세요:
{{
  "title": "영상 제목",
  "openingSegment": {{
    "script": ["호기심 훅킹으로 시작하는 문장", "두 번째 문장"],
    "imageGenPrompt": "이미지 생성 프롬프트"
  }},
  "snippets": [
    {{
      "segmentTitle": "장면 1 제목",
      "rank": 1,
      "script": ["첫 번째 문장", "두 번째 문장"],
      "imageGenPrompt": "이미지 생성 프롬프트"
    }}
  ]
}}

마크다운이나 추가 텍스트 없이 순수 JSON만 출력하세요.
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # JSON 파싱
        if text.startswith("```"):
            text = text.split("```")[1].replace("json\n", "")
        
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

def generate_image(prompt, model="flux-realistic"):
    """Replicate로 이미지 생성"""
    try:
        model_url = IMAGE_MODELS.get(model, IMAGE_MODELS["flux-realistic"])
        output = replicate.run(
            model_url,
            input={"prompt": prompt}
        )
        
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
            # 이미지 다운로드 및 Base64 변환
            response = requests.get(image_url)
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content).decode()
                return f"data:image/png;base64,{base64_image}"
        
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        return {"error": str(e)}

def generate_tts(text, voice="ko-KR-Wavenet-A"):
    """Google Cloud TTS로 음성 생성"""
    try:
        from google.cloud import texttospeech
        
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name=voice,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.9,
            pitch=0.0
        )
        
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        
        audio_base64 = base64.b64encode(response.audio_content).decode()
        return f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        return {"error": str(e)}

# ===== API 엔드포인트 =====

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "AutoVid Full Backend API",
        "version": "1.3.6.0",
        "status": "running",
        "features": [
            "Gemini 스크립트 생성",
            "6개 AI 이미지 모델",
            "Google Cloud TTS",
            "8개 템플릿",
            "55개 전환 효과",
            "InfiniCloud WebDAV",
            "YouTube 업로드"
        ]
    })

@app.route("/api/autovid/script", methods=["POST"])
def create_script():
    """스크립트 생성"""
    data = request.get_json()
    subject = data.get("subject", "")
    request_number = data.get("requestNumber", 5)
    
    if not subject:
        return jsonify({"error": "주제 입력 필요"}), 400
    
    result = generate_script(subject, request_number)
    return jsonify(result)

@app.route("/api/autovid/image", methods=["POST"])
def create_image():
    """이미지 생성"""
    data = request.get_json()
    prompt = data.get("prompt", "")
    model = data.get("model", "flux-realistic")
    
    if not prompt:
        return jsonify({"error": "프롬프트 입력 필요"}), 400
    
    image_url = generate_image(prompt, model)
    
    if isinstance(image_url, dict) and "error" in image_url:
        return jsonify(image_url), 400
    
    return jsonify({
        "success": True,
        "imageUrl": image_url,
        "model": model
    })

@app.route("/api/autovid/tts", methods=["POST"])
def create_tts():
    """음성 생성"""
    data = request.get_json()
    text = data.get("text", "")
    voice = data.get("voice", "ko-KR-Wavenet-A")
    
    if not text:
        return jsonify({"error": "텍스트 입력 필요"}), 400
    
    audio_url = generate_tts(text, voice)
    
    if isinstance(audio_url, dict) and "error" in audio_url:
        return jsonify(audio_url), 400
    
    return jsonify({
        "success": True,
        "audioUrl": audio_url,
        "voice": voice
    })

@app.route("/api/autovid/templates", methods=["GET"])
def list_templates():
    """템플릿 목록"""
    return jsonify({
        "success": True,
        "templates": list(TEMPLATES.keys()),
        "total": len(TEMPLATES),
        "details": TEMPLATES
    })

@app.route("/api/autovid/transitions", methods=["GET"])
def list_transitions():
    """전환 효과 목록"""
    return jsonify({
        "success": True,
        "transitions": FFMPEG_TRANSITIONS,
        "total": len(FFMPEG_TRANSITIONS)
    })

@app.route("/api/autovid/models", methods=["GET"])
def list_models():
    """이미지 모델 목록"""
    return jsonify({
        "success": True,
        "models": list(IMAGE_MODELS.keys()),
        "total": len(IMAGE_MODELS),
        "details": IMAGE_MODELS
    })

@app.route("/api/health", methods=["GET"])
def health():
    """헬스 체크"""
    return jsonify({"status": "healthy"})

# ===== 에러 핸들러 =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
