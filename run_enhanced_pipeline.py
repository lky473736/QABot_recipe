"""
전체 파이프라인 실행 스크립트
- 데이터 수집부터 모델 훈련까지 자동화
- 각 단계별 실행 및 진행상황 확인
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import *

def print_step(step_num: int, title: str, description: str = ""):
    """단계 출력"""
    print(f"\n{'='*80}")
    print(f"🚀 STEP {step_num}: {title}")
    if description:
        print(f"📝 {description}")
    print(f"{'='*80}")

def check_file_exists(filepath: Path, name: str) -> bool:
    """파일 존재 확인"""
    if filepath.exists():
        print(f"✅ {name} 존재: {filepath}")
        return True
    else:
        print(f"❌ {name} 없음: {filepath}")
        return False

def run_script(script_path: str, description: str) -> bool:
    """스크립트 실행"""
    print(f"\n🔧 실행 중: {description}")
    print(f"📂 스크립트: {script_path}")
    
    try:
        # Python 스크립트 실행
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print(f"✅ 성공: {description}")
            print(f"📊 출력:\n{result.stdout}")
            return True
        else:
            print(f"❌ 실패: {description}")
            print(f"❌ 오류:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        return False

def check_api_key():
    """API 키 확인"""
    if FOOD_SAFETY_API_KEY == "0662ae02bb6549ed8e0b" or FOOD_SAFETY_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ API 키가 기본값입니다. config.py에서 올바른 API 키를 설정해주세요.")
        print("📍 식품안전처 공공데이터포털에서 API 키를 발급받으세요.")
        return False
    return True

def main():
    """메인 실행 함수"""
    print("🚀 개선된 레시피 챗봇 전체 파이프라인 시작!")
    print("🎯 목표: 30,000개+ 고품질 레시피 데이터로 챗봇 구축")
    
    start_time = time.time()
    
    # 0단계: 사전 준비
    print_step(0, "사전 준비 및 환경 확인")
    
    # API 키 확인
    if not check_api_key():
        print("\n❌ API 키를 먼저 설정해주세요.")
        return
    
    # 필요한 디렉토리 생성
    for dir_path in REQUIRED_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 디렉토리 확인: {dir_path}")
    
    print("✅ 사전 준비 완료")
    
    # 1단계: 대용량 데이터 수집
    print_step(1, "대용량 레시피 데이터 수집", 
               "식품안전처 API에서 30,000개+ 레시피 수집 및 중복 제거")
    
    if not run_script("data/enhanced_data_collector.py", "대용량 데이터 수집"):
        print("❌ 데이터 수집 실패. 중단합니다.")
        return
    
    # 수집된 데이터 확인
    if not check_file_exists(RAW_RECIPES_PATH, "원본 레시피 데이터"):
        print("❌ 원본 데이터가 없습니다. 중단합니다.")
        return
    
    # 2단계: 데이터 전처리
    print_step(2, "레시피 데이터 전처리", 
               "데이터 정제, 정규화, 카테고리화")
    
    if not run_script("data/enhanced_data_processor.py", "데이터 전처리"):
        print("❌ 데이터 전처리 실패. 중단합니다.")
        return
    
    # 처리된 데이터 확인
    if not check_file_exists(PROCESSED_RECIPES_PATH, "처리된 레시피 데이터"):
        print("❌ 처리된 데이터가 없습니다. 중단합니다.")
        return
    
    # 3단계: QA 데이터셋 생성
    print_step(3, "QA 데이터셋 생성", 
               "다양한 질문-답변 쌍 생성 (수만 개)")
    
    if not run_script("data/enhanced_qa_generator.py", "QA 데이터셋 생성"):
        print("❌ QA 생성 실패. 중단합니다.")
        return
    
    # QA 데이터 확인
    if not check_file_exists(QA_DATASET_PATH, "QA 데이터셋"):
        print("❌ QA 데이터가 없습니다. 중단합니다.")
        return
    
    # 4단계: 모델 훈련
    print_step(4, "챗봇 모델 훈련", 
               "KcBERT 기반 QA 모델 훈련")
    
    if not run_script("model/enhanced_model_trainer.py", "모델 훈련"):
        print("⚠️ 모델 훈련에 문제가 있었지만 계속 진행합니다.")
    
    # 5단계: 챗봇 테스트
    print_step(5, "챗봇 성능 테스트", 
               "훈련된 모델로 챗봇 테스트")
    
    # 간단한 테스트
    try:
        print("🧪 챗봇 테스트 중...")
        
        # 모델 클래스 임포트 및 테스트
        sys.path.append(str(project_root / "model"))
        from enhanced_chatbot_model import EnhancedRecipeChatbot
        
        # 챗봇 초기화
        chatbot = EnhancedRecipeChatbot(TRAINED_MODEL_DIR if TRAINED_MODEL_DIR.exists() else None)
        
        # 테스트 질문들
        test_questions = [
            "안녕하세요",
            "감자로 뭐 만들 수 있어?",
            "김치찌개 만드는 법",
            "불고기 재료가 뭐야?",
            "계란말이 칼로리"
        ]
        
        print("\n🎯 테스트 결과:")
        for i, question in enumerate(test_questions, 1):
            try:
                response = chatbot.generate_response(question)
                print(f"\n{i}. Q: {question}")
                print(f"   A: {response[:100]}{'...' if len(response) > 100 else ''}")
            except Exception as e:
                print(f"   A: 오류 - {e}")
        
        print("\n✅ 챗봇 테스트 완료")
        
    except Exception as e:
        print(f"❌ 챗봇 테스트 실패: {e}")
        print("⚠️ 하지만 기본 모델로 동작할 수 있습니다.")
    
    # 완료 보고
    end_time = time.time()
    total_time = end_time - start_time
    
    print_step("✅", "파이프라인 완료!", 
               f"총 소요시간: {total_time/60:.1f}분")
    
    # 최종 상태 확인
    print("\n📊 최종 결과:")
    
    # 파일 크기 확인
    files_to_check = [
        (RAW_RECIPES_PATH, "원본 레시피 데이터"),
        (PROCESSED_RECIPES_PATH, "처리된 레시피 데이터"),
        (QA_DATASET_PATH, "QA 데이터셋"),
        (TRAINED_MODEL_DIR, "훈련된 모델")
    ]
    
    for filepath, name in files_to_check:
        if filepath.exists():
            if filepath.is_file():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {name}: {size_mb:.1f}MB")
            else:
                # 디렉토리인 경우
                total_size = sum(f.stat().st_size for f in filepath.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"✅ {name}: {size_mb:.1f}MB (디렉토리)")
        else:
            print(f"❌ {name}: 없음")
    
    print(f"\n🎉 개선된 레시피 챗봇 구축 완료!")
    print(f"🚀 app.py를 실행하여 웹 인터페이스를 시작하세요!")
    print(f"💡 또는 enhanced_chatbot_model.py를 직접 사용하세요!")

if __name__ == "__main__":
    main()
