"""
레시피 챗봇 Flask 애플리케이션
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from model.chatbot_model import RecipeChatbot

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'recipe-chatbot-secret-key'
CORS(app)

# 챗봇 인스턴스 초기화
print("챗봇을 초기화하는 중...")
try:
    chatbot = RecipeChatbot(TRAINED_MODEL_DIR if TRAINED_MODEL_DIR.exists() else None)
    print("챗봇 초기화 완료!")
except Exception as e:
    print(f"챗봇 초기화 실패: {e}")
    chatbot = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """채팅 페이지"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """채팅 API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': '메시지가 비어있습니다.'
            })
        
        if not chatbot:
            return jsonify({
                'success': False,
                'error': '챗봇이 초기화되지 않았습니다.'
            })
        
        # 챗봇 응답 생성
        response = chatbot.generate_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        print(f"채팅 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/api/recipes/search', methods=['GET'])
def api_search_recipes():
    """레시피 검색 API"""
    try:
        query = request.args.get('q', '').strip()
        search_type = request.args.get('type', 'ingredient')  # ingredient, name, category
        
        if not query:
            return jsonify({
                'success': False,
                'error': '검색어가 비어있습니다.'
            })
        
        if not chatbot:
            return jsonify({
                'success': False,
                'error': '챗봇이 초기화되지 않았습니다.'
            })
        
        # 검색 수행
        if search_type == 'ingredient':
            recipes = chatbot.search_recipes_by_ingredient(query)
        elif search_type == 'name':
            recipes = chatbot.search_recipes_by_name(query)
        else:
            recipes = []
        
        return jsonify({
            'success': True,
            'recipes': recipes,
            'count': len(recipes)
        })
        
    except Exception as e:
        print(f"레시피 검색 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/api/recipes/<recipe_id>')
def api_get_recipe(recipe_id):
    """특정 레시피 조회 API"""
    try:
        if not chatbot:
            return jsonify({
                'success': False,
                'error': '챗봇이 초기화되지 않았습니다.'
            })
        
        recipe = chatbot.get_recipe_by_id(recipe_id)
        
        if not recipe:
            return jsonify({
                'success': False,
                'error': '레시피를 찾을 수 없습니다.'
            })
        
        return jsonify({
            'success': True,
            'recipe': recipe
        })
        
    except Exception as e:
        print(f"레시피 조회 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/api/health')
def api_health():
    """헬스 체크 API"""
    return jsonify({
        'status': 'healthy',
        'chatbot_loaded': chatbot is not None,
        'recipes_count': len(chatbot.recipes) if chatbot else 0,
        'qa_count': len(chatbot.qa_dataset) if chatbot else 0
    })

@app.errorhandler(404)
def not_found(error):
    """404 오류 처리"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 오류 처리"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # 디렉토리 확인
    print(f"템플릿 디렉토리: {TEMPLATES_DIR}")
    print(f"정적 파일 디렉토리: {STATIC_DIR}")
    
    # 데이터 확인
    if chatbot:
        print(f"로드된 레시피 수: {len(chatbot.recipes)}")
        print(f"로드된 QA 수: {len(chatbot.qa_dataset)}")
    
    # Flask 앱 실행
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )