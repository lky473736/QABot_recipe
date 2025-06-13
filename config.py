"""
레시피 챗봇 설정 파일 - 농림축산식품 공공데이터 사용
"""
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.absolute()

# 농림축산식품 공공데이터 API 설정
MAFRA_API_KEY = "c43f9e43df898ac83c17fecf1abcd3e0af0bf29087be02128cf82a9e8679c90c"
MAFRA_BASE_URL = "http://211.237.50.150:7080/openapi"

# API 서비스 ID들
RECIPE_BASIC_SERVICE_ID = "Grid_20150827000000000226_1"      # 레시피 기본정보 (537개)
RECIPE_INGREDIENT_SERVICE_ID = "Grid_20150827000000000227_1"  # 레시피 재료정보 (6104개)
RECIPE_PROCESS_SERVICE_ID = "Grid_20150827000000000228_1"     # 레시피 과정정보 (3022개)

# 데이터 디렉토리
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 데이터 파일 경로
RAW_RECIPES_PATH = DATA_DIR / "raw_recipes.json"
PROCESSED_RECIPES_PATH = DATA_DIR / "processed_recipes.json"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.json"

# 새로운 API용 데이터 파일 경로
RECIPE_BASIC_PATH = DATA_DIR / "recipe_basic.json"
RECIPE_INGREDIENT_PATH = DATA_DIR / "recipe_ingredient.json"
RECIPE_PROCESS_PATH = DATA_DIR / "recipe_process.json"
RECIPE_INGREDIENT_MAP_PATH = DATA_DIR / "recipe_ingredient_map.json"

# 모델 설정
MODEL_NAME = "beomi/kcbert-base"
TRAINED_MODEL_DIR = MODEL_DIR / "trained_model"
MAX_LENGTH = 300
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