"""
농림축산식품 공공데이터 기반 레시피 챗봇 전체 파이프라인 실행 스크립트
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
        if filepath.is_file():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {name} 존재: {filepath} ({size_mb:.1f}MB)")
        else:
            total_size = sum(f.stat().st_size for f in filepath.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"✅ {name} 존재: {filepath} ({size_mb:.1f}MB 디렉토리)")
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
            # 출력이 너무 길면 마지막 1000자만 표시
            stdout = result.stdout
            if len(stdout) > 1000:
                print(f"📊 출력 (마지막 1000자):\n...{stdout[-1000:]}")
            else:
                print(f"📊 출력:\n{stdout}")
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
    if not MAFRA_API_KEY or len(MAFRA_API_KEY) < 10:
        print("⚠️ 농림축산식품 API 키가 설정되지 않았습니다.")
        print("📍 config.py에서 MAFRA_API_KEY를 올바르게 설정해주세요.")
        return False
    print(f"✅ API 키 확인됨: {MAFRA_API_KEY[:10]}...")
    return True

def print_data_summary():
    """데이터 요약 정보 출력"""
    print(f"\n📊 농림축산식품 공공데이터 정보:")
    print(f"   🍳 레시피 기본정보: 약 537개")
    print(f"   🥕 레시피 재료정보: 약 6,104개") 
    print(f"   👨‍🍳 레시피 과정정보: 약 3,022개")
    print(f"   🔗 데이터 연결: 레시피 코드 기반 조인")
    print(f"   📈 예상 통합 레시피: 약 500+ 개")

def main():
    """메인 실행 함수"""
    print("🚀 농림축산식품 공공데이터 기반 레시피 챗봇 전체 파이프라인 시작!")
    print("🎯 목표: 농림축산식품부 공식 데이터로 정확한 레시피 챗봇 구축")
    
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
    
    # 데이터 요약 정보 출력
    print_data_summary()
    
    print("✅ 사전 준비 완료")
    
    # 1단계: 농림축산식품 데이터 수집
    print_step(1, "농림축산식품 공공데이터 수집", 
               "기본정보, 재료정보, 과정정보 3개 테이블 수집 및 무결성 검증")
    
    if not run_script("data/enhanced_data_collector.py", "농림축산식품 데이터 수집"):
        print("❌ 데이터 수집 실패. 중단합니다.")
        return
    
    # 수집된 데이터 확인
    required_data_files = [RECIPE_BASIC_PATH, RECIPE_INGREDIENT_PATH, RECIPE_PROCESS_PATH]
    missing_data = [f for f in required_data_files if not check_file_exists(f, f.name)]
    
    if missing_data:
        print("❌ 필요한 데이터 파일이 없습니다. 중단합니다.")
        return
    
    # 호환성을 위한 통합 데이터 확인
    if not check_file_exists(RAW_RECIPES_PATH, "통합 원본 데이터"):
        print("❌ 통합 데이터가 없습니다. 중단합니다.")
        return
    
    # 2단계: 데이터 전처리 및 통합
    print_step(2, "레시피 데이터 전처리 및 통합", 
               "3개 테이블 조인, 데이터 정제, 정규화, 카테고리화")
    
    if not run_script("data/enhanced_data_processor.py", "데이터 전처리 및 통합"):
        print("❌ 데이터 전처리 실패. 중단합니다.")
        return
    
    # 처리된 데이터 확인
    if not check_file_exists(PROCESSED_RECIPES_PATH, "처리된 레시피 데이터"):
        print("❌ 처리된 데이터가 없습니다. 중단합니다.")
        return
    
    # 3단계: QA 데이터셋 생성
    print_step(3, "QA 데이터셋 생성", 
               "다양한 질문-답변 쌍 생성 (농림축산식품 데이터 기반)")
    
    if not run_script("data/enhanced_qa_generator.py", "QA 데이터셋 생성"):
        print("❌ QA 생성 실패. 중단합니다.")
        return
    
    # QA 데이터 확인
    if not check_file_exists(QA_DATASET_PATH, "QA 데이터셋"):
        print("❌ QA 데이터가 없습니다. 중단합니다.")
        return
    
    # 4단계: 모델 훈련 (선택적)
    print_step(4, "챗봇 모델 훈련 (선택사항)", 
               "KcBERT 기반 QA 모델 훈련")
    
    user_input = input("\n모델 훈련을 진행하시겠습니까? (y/N): ").strip().lower()
    if user_input == 'y' or user_input == 'yes':
        if not run_script("model/enhanced_model_trainer.py", "모델 훈련"):
            print("⚠️ 모델 훈련에 문제가 있었지만 계속 진행합니다.")
    else:
        print("⏭️ 모델 훈련을 건너뜁니다. 사전 훈련된 모델을 사용합니다.")
    
    # 5단계: 챗봇 테스트
    print_step(5, "농림축산식품 챗봇 성능 테스트", 
               "훈련된 모델로 챗봇 테스트")
    
    # 간단한 테스트
    try:
        print("🧪 농림축산식품 챗봇 테스트 중...")
        
        # 모델 클래스 임포트 및 테스트
        sys.path.append(str(project_root / "model"))
        from enhanced_chatbot_model import EnhancedRecipeChatbot
        
        # 챗봇 초기화
        chatbot = EnhancedRecipeChatbot(TRAINED_MODEL_DIR if TRAINED_MODEL_DIR.exists() else None)
        
        # 농림축산식품 데이터 특화 테스트 질문들
        test_questions = [
            "안녕하세요",
            "쇠고기로 뭐 만들 수 있어?",
            "된장찌개 만드는 법",
            "김치볶음밥 재료가 뭐야?",
            "쉬운 요리 추천해줘",
            "볶음 요리 뭐가 있어?",
            "불고기 만들기 어려워?",
            "밑반찬 추천해줘"
        ]
        
        print("\n🎯 농림축산식품 데이터 기반 테스트 결과:")
        for i, question in enumerate(test_questions, 1):
            try:
                response = chatbot.generate_response(question)
                print(f"\n{i}. Q: {question}")
                print(f"   A: {response[:150]}{'...' if len(response) > 150 else ''}")
            except Exception as e:
                print(f"   A: 오류 - {e}")
        
        print("\n✅ 농림축산식품 챗봇 테스트 완료")
        
    except Exception as e:
        print(f"❌ 챗봇 테스트 실패: {e}")
        print("⚠️ 하지만 기본 모델로 동작할 수 있습니다.")
    
    # 완료 보고
    end_time = time.time()
    total_time = end_time - start_time
    
    print_step("✅", "농림축산식품 파이프라인 완료!", 
               f"총 소요시간: {total_time/60:.1f}분")
    
    # 최종 상태 확인
    print("\n📊 최종 결과:")
    
    # 파일 크기 확인
    files_to_check = [
        (RECIPE_BASIC_PATH, "농림축산식품 기본정보"),
        (RECIPE_INGREDIENT_PATH, "농림축산식품 재료정보"),
        (RECIPE_PROCESS_PATH, "농림축산식품 과정정보"),
        (RAW_RECIPES_PATH, "통합 원본 데이터"),
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
    
    # 데이터 통계 출력
    try:
        import json
        
        # 처리된 레시피 통계
        if PROCESSED_RECIPES_PATH.exists():
            with open(PROCESSED_RECIPES_PATH, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                
            if 'statistics' in processed_data:
                stats = processed_data['statistics']
                print(f"\n📈 레시피 통계:")
                
                if 'categories' in stats:
                    print(f"   카테고리 분포:")
                    for cat, count in list(stats['categories'].items())[:5]:
                        print(f"     • {cat}: {count}개")
                
                if 'top_ingredients' in stats:
                    print(f"   인기 재료 (상위 5개):")
                    for ing, count in list(stats['top_ingredients'].items())[:5]:
                        print(f"     • {ing}: {count}개 레시피")
        
        # QA 통계
        if QA_DATASET_PATH.exists():
            with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
                
            if 'statistics' in qa_data:
                qa_stats = qa_data['statistics']
                print(f"\n💬 QA 통계:")
                
                if 'type_distribution' in qa_stats:
                    print(f"   QA 유형 분포:")
                    for qa_type, count in list(qa_stats['type_distribution'].items())[:5]:
                        print(f"     • {qa_type}: {count}개")
                        
    except Exception as e:
        print(f"⚠️ 통계 출력 중 오류: {e}")
    
    print(f"\n🎉 농림축산식품 공공데이터 기반 레시피 챗봇 구축 완료!")
    print(f"🚀 app.py를 실행하여 웹 인터페이스를 시작하세요!")
    print(f"💡 또는 enhanced_chatbot_model.py를 직접 사용하세요!")
    print(f"\n📋 다음 명령어로 웹 서버를 시작할 수 있습니다:")
    print(f"   python app.py")
    print(f"\n🌐 그러면 http://localhost:5000 에서 챗봇을 사용할 수 있습니다.")

if __name__ == "__main__":
    main()