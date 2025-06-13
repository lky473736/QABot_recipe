import os
import sys
import time
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import *

sys.path.append(str(project_root / "model"))
from enhanced_chatbot_model import EnhancedRecipeChatbot  
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