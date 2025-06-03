"""
개선된 레시피 챗봇 Flask 애플리케이션
- 새로운 enhanced 모델 사용
- 향상된 성능 및 응답 품질
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced-recipe-chatbot-secret-key'
CORS(app)

# 챗봇 인스턴스 초기화
print("🤖 개선된 레시피 챗봇을 초기화하는 중...")
try:
    # 개선된 모델 임포트
    from model.enhanced_chatbot_model import EnhancedRecipeChatbot
    
    # 훈련된 모델이 있으면 사용, 없으면 사전 훈련된 모델 사용
    model_path = TRAINED_MODEL_DIR if TRAINED_MODEL_DIR.exists() else None
    chatbot = EnhancedRecipeChatbot(model_path)
    
    print("✅ 개선된 챗봇 초기화 완료!")
    
except Exception as e:
    print(f"❌ 개선된 챗봇 초기화 실패: {e}")
    print("🔄 기본 모델로 폴백...")
    
    try:
        # 기존 모델로 폴백
        from model.chatbot_model import RecipeChatbot
        chatbot = RecipeChatbot(TRAINED_MODEL_DIR if TRAINED_MODEL_DIR.exists() else None)
        print("✅ 기본 챗봇 초기화 완료!")
    except Exception as e2:
        print(f"❌ 기본 챗봇도 초기화 실패: {e2}")
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
    """채팅 API - 개선된 버전"""
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
                'error': '챗봇이 초기화되지 않았습니다. 서버를 다시 시작해주세요.'
            })
        
        # 챗봇 응답 생성
        print(f"📩 사용자 질문: {user_message}")
        response = chatbot.generate_response(user_message)
        print(f"🤖 챗봇 응답: {response[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        print(f"❌ 채팅 API 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
        })

@app.route('/api/recipes/search', methods=['GET'])
def api_search_recipes():
    """레시피 검색 API - 개선된 버전"""
    try:
        query = request.args.get('q', '').strip()
        search_type = request.args.get('type', 'ingredient')  # ingredient, name, category, cooking_method
        
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
        recipes = []
        if search_type == 'ingredient':
            recipes = chatbot.search_recipes_by_ingredient(query)
        elif search_type == 'name':
            recipes = chatbot.search_recipes_by_name(query)
        elif search_type == 'category':
            recipes = chatbot.search_recipes_by_category(query)
        elif search_type == 'cooking_method':
            recipes = chatbot.search_recipes_by_cooking_method(query)
        
        # 응답 포맷팅
        formatted_recipes = []
        for recipe in recipes:
            formatted_recipe = {
                'id': recipe.get('id', ''),
                'name': recipe.get('name', ''),
                'category': recipe.get('category', ''),
                'cooking_method': recipe.get('cooking_method', ''),
                'main_ingredients': recipe.get('main_ingredients', []),
                'summary': recipe.get('summary', ''),
                'main_image': recipe.get('main_image', '')
            }
            formatted_recipes.append(formatted_recipe)
        
        return jsonify({
            'success': True,
            'recipes': formatted_recipes,
            'count': len(formatted_recipes),
            'search_type': search_type,
            'query': query
        })
        
    except Exception as e:
        print(f"❌ 레시피 검색 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/api/recipes/<recipe_id>')
def api_get_recipe(recipe_id):
    """특정 레시피 조회 API - 개선된 버전"""
    try:
        if not chatbot:
            return jsonify({
                'success': False,
                'error': '챗봇이 초기화되지 않았습니다.'
            })
        
        # 레시피 검색 (개선된 방법)
        recipe = None
        for r in chatbot.recipes:
            if r.get('id') == recipe_id:
                recipe = r
                break
        
        if not recipe:
            return jsonify({
                'success': False,
                'error': '레시피를 찾을 수 없습니다.'
            })
        
        # 상세 정보 포함
        detailed_recipe = {
            'id': recipe.get('id', ''),
            'name': recipe.get('name', ''),
            'category': recipe.get('category', ''),
            'cooking_method': recipe.get('cooking_method', ''),
            'ingredients': recipe.get('ingredients', ''),
            'main_ingredients': recipe.get('main_ingredients', []),
            'steps': recipe.get('steps', []),
            'nutrition': recipe.get('nutrition', {}),
            'tip': recipe.get('tip', ''),
            'hashtag': recipe.get('hashtag', ''),
            'main_image': recipe.get('main_image', ''),
            'summary': recipe.get('summary', '')
        }
        
        return jsonify({
            'success': True,
            'recipe': detailed_recipe
        })
        
    except Exception as e:
        print(f"❌ 레시피 조회 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
        })

@app.route('/api/health')
def api_health():
    """헬스 체크 API - 개선된 버전"""
    try:
        health_info = {
            'status': 'healthy',
            'chatbot_loaded': chatbot is not None,
            'model_type': 'enhanced' if hasattr(chatbot, 'qa_embeddings') else 'basic',
            'recipes_count': len(chatbot.recipes) if chatbot else 0,
            'qa_count': len(chatbot.qa_dataset) if chatbot else 0,
        }
        
        # 추가 정보
        if chatbot:
            health_info.update({
                'has_trained_model': TRAINED_MODEL_DIR.exists(),
                'has_qa_embeddings': hasattr(chatbot, 'qa_embeddings') and chatbot.qa_embeddings is not None,
                'recipe_index_size': len(chatbot.recipe_index.get('by_ingredient', {})) if hasattr(chatbot, 'recipe_index') else 0
            })
        
        return jsonify(health_info)
        
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/stats')
def api_stats():
    """통계 정보 API"""
    try:
        if not chatbot:
            return jsonify({
                'success': False,
                'error': '챗봇이 초기화되지 않았습니다.'
            })
        
        stats = {
            'total_recipes': len(chatbot.recipes),
            'total_qa_pairs': len(chatbot.qa_dataset),
        }
        
        # 카테고리별 통계
        if chatbot.recipes:
            from collections import defaultdict
            categories = defaultdict(int)
            cooking_methods = defaultdict(int)
            
            for recipe in chatbot.recipes:
                category = recipe.get('category', '기타')
                method = recipe.get('cooking_method', '기타')
                categories[category] += 1
                cooking_methods[method] += 1
            
            stats['categories'] = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])
            stats['cooking_methods'] = dict(sorted(cooking_methods.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # QA 유형별 통계
        if chatbot.qa_dataset:
            qa_types = defaultdict(int)
            for qa in chatbot.qa_dataset:
                qa_type = qa.get('type', 'unknown')
                qa_types[qa_type] += 1
            
            stats['qa_types'] = dict(sorted(qa_types.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ 통계 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': '서버 오류가 발생했습니다.'
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
    print(f"📁 템플릿 디렉토리: {TEMPLATES_DIR}")
    print(f"📁 정적 파일 디렉토리: {STATIC_DIR}")
    
    # 데이터 확인
    if chatbot:
        print(f"📊 로드된 레시피 수: {len(chatbot.recipes)}")
        print(f"📊 로드된 QA 수: {len(chatbot.qa_dataset)}")
        
        if hasattr(chatbot, 'qa_embeddings') and chatbot.qa_embeddings is not None:
            print(f"🧠 QA 임베딩: {chatbot.qa_embeddings.shape}")
        
        if hasattr(chatbot, 'recipe_index'):
            print(f"🔍 레시피 인덱스: {len(chatbot.recipe_index.get('by_ingredient', {}))}개 재료")
    
    print(f"\n🚀 개선된 레시피 챗봇 서버 시작!")
    print(f"🌐 http://localhost:{FLASK_PORT} 에서 접속하세요!")
    
    # Flask 앱 실행
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )