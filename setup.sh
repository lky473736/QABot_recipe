#!/bin/bash

# 한국 전통요리 레시피 마스터 설정 스크립트

echo "========================================"
echo "한국 전통요리 레시피 마스터 설정 시작"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 에러 처리
set -e
trap 'log_error "설정 중 오류가 발생했습니다. 라인 $LINENO"' ERR

# Python 버전 확인
check_python() {
    log_info "Python 버전 확인 중..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python3 발견: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        log_success "Python 발견: $PYTHON_VERSION"
        PYTHON_CMD="python"
    else
        log_error "Python이 설치되어 있지 않습니다."
        exit 1
    fi
    
    # Python 3.8+ 확인
    MIN_VERSION="3.8"
    if ! $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8 이상이 필요합니다. 현재 버전: $PYTHON_VERSION"
        exit 1
    fi
}

# 가상환경 설정
setup_virtual_env() {
    log_info "가상환경 설정 중..."
    
    if [ ! -d "venv" ]; then
        log_info "가상환경 생성 중..."
        $PYTHON_CMD -m venv venv
        log_success "가상환경 생성 완료"
    else
        log_info "기존 가상환경 발견"
    fi
    
    # 가상환경 활성화
    log_info "가상환경 활성화 중..."
    source venv/bin/activate || source venv/Scripts/activate
    log_success "가상환경 활성화 완료"
    
    # pip 업그레이드
    log_info "pip 업그레이드 중..."
    pip install --upgrade pip
}

# 의존성 설치
install_dependencies() {
    log_info "Python 패키지 설치 중..."
    
    # requirements.txt가 있는지 확인
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 파일을 찾을 수 없습니다."
        exit 1
    fi
    
    # PyTorch 설치 (CPU 버전)
    log_info "PyTorch 설치 중..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 나머지 의존성 설치
    log_info "기타 의존성 설치 중..."
    pip install -r requirements.txt
    
    log_success "패키지 설치 완료"
}

# KoNLPy 설정 (한국어 자연어 처리)
setup_konlpy() {
    log_info "KoNLPy 설정 확인 중..."
    
    # Java 확인
    if ! command -v java &> /dev/null; then
        log_warning "Java가 설치되어 있지 않습니다."
        log_info "KoNLPy 사용을 위해 Java가 필요합니다."
        
        # OS별 Java 설치 안내
        case "$OSTYPE" in
            linux*)
                log_info "Ubuntu/Debian: sudo apt-get install openjdk-8-jdk"
                log_info "CentOS/RHEL: sudo yum install java-1.8.0-openjdk-devel"
                ;;
            darwin*)
                log_info "macOS: brew install openjdk@8"
                ;;
            msys*|win32*)
                log_info "Windows: https://www.oracle.com/java/technologies/javase-downloads.html"
                ;;
        esac
    else
        log_success "Java 발견: $(java -version 2>&1 | head -n 1)"
    fi
}

# 디렉토리 구조 생성
create_directories() {
    log_info "디렉토리 구조 생성 중..."
    
    # 필요한 디렉토리들
    directories=(
        "recipe_data"
        "recipe_qa_model"
        "logs"
        "templates"
        "static/css"
        "static/js"
        "static/images"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "디렉토리 생성: $dir"
        fi
    done
    
    log_success "디렉토리 구조 생성 완료"
}

# 환경 변수 설정
setup_environment() {
    log_info "환경 변수 설정 중..."
    
    # .env 파일 생성
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# 한국 전통요리 레시피 마스터 환경 변수

# Flask 설정
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True

# 데이터 설정
DATA_DIR=recipe_data
MODEL_DIR=recipe_qa_model

# API 키 (공공데이터포털에서 발급받아 설정)
FOOD_SAFETY_API_KEY=YOUR_FOOD_SAFETY_API_KEY
RURAL_DEV_API_KEY=YOUR_RURAL_DEV_API_KEY
AGRI_FOOD_API_KEY=YOUR_AGRI_FOOD_API_KEY

# 모델 설정
MODEL_NAME=beomi/kcbert-base
MAX_SEQ_LENGTH=512
BATCH_SIZE=4
LEARNING_RATE=3e-5
NUM_EPOCHS=3

# 로깅
LOG_LEVEL=INFO
EOF
        log_success ".env 파일 생성 완료"
    else
        log_info ".env 파일이 이미 존재합니다"
    fi
}

# 샘플 실행
run_sample() {
    log_info "샘플 실행 중..."
    
    # 빠른 설정으로 데모 데이터 생성
    $PYTHON_CMD run_pipeline.py --mode quick
    
    log_success "샘플 데이터 생성 완료"
}

# 사용법 안내
show_usage() {
    echo ""
    echo "========================================"
    echo "설정 완료!"
    echo "========================================"
    echo ""
    echo "다음 명령어로 시스템을 사용할 수 있습니다:"
    echo ""
    echo "1. 가상환경 활성화:"
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate     # Windows"
    echo ""
    echo "2. 전체 파이프라인 실행:"
    echo "   python run_pipeline.py --mode full"
    echo ""
    echo "3. 빠른 데모 실행:"
    echo "   python run_pipeline.py --mode quick"
    echo ""
    echo "4. 웹 서버 시작:"
    echo "   python app.py"
    echo ""
    echo "5. API 키 설정 (선택사항):"
    echo "   .env 파일에서 API 키를 실제 값으로 변경"
    echo ""
    echo "웹 브라우저에서 http://localhost:5000 접속"
    echo ""
    echo "========================================"
}

# 메인 실행
main() {
    log_info "설정 시작..."
    
    # 1. Python 확인
    check_python
    
    # 2. 가상환경 설정
    setup_virtual_env
    
    # 3. 의존성 설치
    install_dependencies
    
    # 4. KoNLPy 설정 확인
    setup_konlpy
    
    # 5. 디렉토리 생성
    create_directories
    
    # 6. 환경 변수 설정
    setup_environment
    
    # 7. 샘플 실행
    run_sample
    
    # 8. 사용법 안내
    show_usage
    
    log_success "모든 설정이 완료되었습니다!"
}

# 스크립트 시작
main "$@"s