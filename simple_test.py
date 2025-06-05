from pathlib import Path
import sys

project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from model.enhanced_chatbot_model import EnhancedRecipeChatbot

trained_model_dir = project_root / "model" / "trained_model"

chatbot = EnhancedRecipeChatbot(trained_model_dir)

test_questions = [
    "안녕하세요",
    "감자로 뭐 만들 수 있어?",
    "김치찌개 만드는 법",
    "불고기 재료가 뭐야?",
    "계란말이 칼로리"
]

print("\n🎯 챗봇 응답 테스트:")
for i, question in enumerate(test_questions, 1):
    try:
        response = chatbot.generate_response(question)
        print(f"\n{i}. Q: {question}")
        print(f"   A: {response}")
    except Exception as e:
        print(f"   A: 오류 발생 - {e}")

print("\n✅ 테스트 완료")
