"""
레시피 챗봇 설정 파일
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.absolute()

# API 설정
FOOD_SAFETY_API_KEY = "0662ae02bb6549ed8e0b"  # 여기에 실제 API 키를 입력하세요
FOOD_SAFETY_BASE_URL = "http://openapi.foodsafetykorea.go.kr/api"
RECIPE_SERVICE_ID = "COOKRCP01"

# 데이터 디렉토리
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 데이터 파일 경로
RAW_RECIPES_PATH = DATA_DIR / "raw_recipes.json"
PROCESSED_RECIPES_PATH = DATA_DIR / "processed_recipes.json"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.json"

# 모델 설정
MODEL_NAME = "beomi/kcbert-base"
TRAINED_MODEL_DIR = MODEL_DIR / "trained_model"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Flask 설정
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# 생성할 디렉토리들
REQUIRED_DIRS = [
    DATA_DIR,
    MODEL_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    STATIC_DIR / "css",
    STATIC_DIR / "js",
    STATIC_DIR / "images",
    TRAINED_MODEL_DIR,
    TRAINED_MODEL_DIR / "tokenizer"
]

# 디렉토리 생성
for dir_path in REQUIRED_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)