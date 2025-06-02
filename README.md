# 🍲 한국 전통요리 레시피 마스터

BERT 기반 질의응답 시스템으로 한국 전통요리에 대한 모든 질문에 답변하는 AI 서비스

## 📋 프로젝트 개요

한국 전통요리 레시피 마스터는 공공데이터와 AI 기술을 결합하여 사용자의 한국 요리 관련 질문에 정확하고 상세한 답변을 제공하는 웹 서비스입니다.

### 주요 특징

- **🤖 AI 기반 질의응답**: BERT 모델을 한국 요리 데이터로 파인튜닝
- **📊 실제 데이터 활용**: 식약처, 농촌진흥청 등 공공 API 데이터 수집
- **🔍 지능형 검색**: 재료, 조리법, 영양 정보 기반 검색
- **💬 자연어 처리**: 일상 언어로 질문하고 답변 받기
- **🌐 웹 인터페이스**: 직관적이고 반응형 웹 UI

### 지원하는 질문 유형

- **레시피 조리법**: "김치찌개는 어떻게 만들어요?"
- **재료 정보**: "불고기에 들어가는 재료는?"
- **영양 정보**: "고구마죽의 칼로리는?"
- **카테고리별**: "찌개 종류에는 어떤 것들이 있나요?"
- **조리법별**: "끓이는 요리를 추천해주세요"

## 🏗️ 시스템 아키텍처

```
데이터 수집 → 전처리 → 모델 훈련 → 웹 서비스
     ↓           ↓         ↓          ↓
  공공 API   QA 쌍 생성   BERT 파인튜닝  Flask 웹앱
```

### 데이터 소스

1. **식품의약품안전처**: 조리식품 레시피 DB
2. **농촌진흥청**: 농식품 영양 정보 
3. **농림축산식품부**: 우리 농산물 요리 정보

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/korean-recipe-master.git
cd korean-recipe-master
```

### 2. 자동 설정 (추천)

```bash
chmod +x setup.sh
./setup.sh
```

### 3. 수동 설정

#### 환경 요구사항
- Python 3.8+
- Java 8+ (KoNLPy용)
- 4GB+ RAM (모델 훈련용)

#### 가상환경 설정
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
```

#### 의존성 설치
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. 실행

#### 빠른 데모 (추천)
```bash
python run_pipeline.py --mode quick
python app.py
```

#### 전체 파이프라인
```bash
python run_pipeline.py --mode full
python app.py
```

웹 브라우저에서 `http://localhost:5000` 접속

## 📁 프로젝트 구조

```
korean-recipe-master/
├── data_collector.py      # 데이터 수집기
├── data_preprocessor.py   # 데이터 전처리기
├── model_trainer.py       # BERT 모델 훈련기
├── app.py                 # Flask 웹 애플리케이션
├── run_pipeline.py        # 전체 파이프라인 실행기
├── setup.sh              # 자동 설정 스크립트
├── requirements.txt       # Python 의존성
├── templates/
│   └── index.html        # 웹 인터페이스
├── recipe_data/          # 데이터 저장소
├── recipe_qa_model/      # 훈련된 모델
└── logs/                 # 로그 파일
```

## 🔧 상세 사용법

### 데이터 수집

#### 공공 API 키 설정 (선택사항)
`.env` 파일에서 API 키 설정:
```env
FOOD_SAFETY_API_KEY=your_actual_api_key
RURAL_DEV_API_KEY=your_actual_api_key
```

#### 데이터 수집 실행
```bash
# 모의 데이터만 사용 (기본)
python run_pipeline.py --mode data-only

# 실제 API 데이터 수집
python run_pipeline.py --mode data-only --api-key-food YOUR_KEY
```

### 모델 훈련

```bash
# 기본 설정으로 훈련
python run_pipeline.py --mode train-only

# 강제 재훈련
python run_pipeline.py --mode train-only --force-retrain
```

### 웹 서비스 API

#### 질문 API
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "김치찌개 만드는 법"}'
```

#### 레시피 검색 API
```bash
curl "http://localhost:5000/api/recipes?q=김치&page=1&per_page=5"
```

#### 통계 API
```bash
curl "http://localhost:5000/api/stats"
```

## 🧪 테스트

### 모델 성능 테스트
```bash
python -c "
from model_trainer import RecipeQATrainer
trainer = RecipeQATrainer()
trainer.test_model()
"
```

### 웹 API 테스트
```python
import requests

# 질문 테스트
response = requests.post('http://localhost:5000/api/ask', 
                        json={'question': '김치찌개는 어떻게 만들어요?'})
print(response.json())
```

## 📊 데이터셋

### 기본 제공 데이터
- **레시피 수**: 50+ 개 (모의 데이터)
- **QA 쌍**: 500+ 개 (자동 생성)
- **카테고리**: 찌개, 죽, 구이, 나물 등

### 실제 API 데이터 (API 키 필요)
- **식약처 레시피**: 1,000+ 개
- **농식품 정보**: 2,000+ 개
- **영양 정보**: 상세 영양성분 데이터

## 🔄 CI/CD 및 배포

### Docker 배포
```bash
# Dockerfile 생성 예정
docker build -t recipe-master .
docker run -p 5000:5000 recipe-master
```

### 클라우드 배포
- **Heroku**: `Procfile` 설정
- **AWS EC2**: 인스턴스 설정 가이드
- **Google Cloud**: App Engine 배포

## 🛠️ 커스터마이징

### 새로운 데이터 소스 추가

```python
# data_collector.py에서
def add_new_data_source(self):
    # 새로운 API 호출 로직
    pass
```

### 모델 파라미터 조정

```python
# run_pipeline.py에서
trainer = RecipeQATrainer(
    model_name='beomi/kcbert-base',  # 다른 모델 사용
    batch_size=8,                   # 배치 크기 조정
    num_epochs=5,                   # 에포크 수 조정
    learning_rate=2e-5              # 학습률 조정
)
```

### 새로운 질문 유형 추가

```python
# data_preprocessor.py에서
self.qa_templates = {
    'new_category': [
        "새로운 질문 템플릿 {name}",
        # ...
    ]
}
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 메모리 부족
```bash
# 배치 크기 줄이기
python run_pipeline.py --mode train-only --batch-size 2
```

#### 2. Java 관련 오류 (KoNLPy)
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk

# macOS
brew install openjdk@8

# Windows
# Oracle Java 8 다운로드 설치
```

#### 3. 토큰화 오류
```bash
# transformers 재설치
pip uninstall transformers
pip install transformers==4.35.0
```

#### 4. CUDA 오류
```bash
# CPU 버전으로 PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 로그 확인

```bash
# 파이프라인 로그
tail -f pipeline.log

# Flask 로그
tail -f logs/app.log
```

## 📈 성능 최적화

### 모델 최적화
- **양자화**: 모델 크기 reduction
- **프루닝**: 불필요한 가중치 제거
- **지식 증류**: 작은 모델로 성능 유지

### 서버 최적화
- **캐싱**: Redis를 통한 답변 캐싱
- **로드 밸런싱**: Gunicorn + Nginx
- **비동기 처리**: Celery 작업 큐

## 🤝 기여하기

### 개발 환경 설정
```bash
git clone https://github.com/your-repo/korean-recipe-master.git
cd korean-recipe-master
pip install -e .
pre-commit install
```

### 코드 스타일
- **Black**: 코드 포매터
- **isort**: Import 정렬
- **flake8**: 린터

### 테스트 실행
```bash
python -m pytest tests/
```

### Pull Request
1. Fork 저장소
2. Feature 브랜치 생성
3. 변경사항 커밋
4. 테스트 실행
5. Pull Request 생성

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 👥 기여자

- **메인 개발자**: [Your Name]
- **데이터 수집**: 공공데이터포털 API
- **모델**: Hugging Face Transformers

## 📞 지원 및 문의

- **이슈 신고**: [GitHub Issues](https://github.com/your-repo/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **이메일**: your-email@example.com

## 🗺️ 로드맵

### v1.0 (현재)
- [x] 기본 QA 시스템
- [x] 웹 인터페이스
- [x] 공공 데이터 연동

### v1.1 (계획)
- [ ] 사용자 피드백 시스템
- [ ] 레시피 북마크 기능
- [ ] 모바일 앱

### v2.0 (장기)
- [ ] 이미지 인식 기반 요리 분석
- [ ] 개인화 추천 시스템
- [ ] 다국어 지원

---

**한국 전통요리 레시피 마스터**로 우리나라 음식 문화를 AI와 함께 탐험해보세요! 🇰🇷