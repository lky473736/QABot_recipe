from flask import Flask, request, render_template, jsonify
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import json
import os
import logging
from typing import Dict, List, Any
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SimpleRecipeMaster:
    def __init__(self):
        self.data_dir = 'recipe_data'
        self.recipes = []
        self.qa_database = []
        
        # 간단한 키워드 기반 답변 시스템
        self.keyword_responses = {
            '김치찌개': {
                'answer': '김치찌개 만드는 법: 1) 김치를 한입 크기로 자른다 2) 돼지고기를 볶아 기름을 낸다 3) 김치를 넣고 함께 볶는다 4) 물을 넣고 끓인 후 두부와 대파를 넣는다',
                'ingredients': '김치 200g, 돼지고기 100g, 두부 100g, 대파 20g'
            },
            '된장찌개': {
                'answer': '된장찌개 만드는 법: 1) 멸치로 육수를 우린다 2) 된장을 체에 걸러 육수에 푼다 3) 야채를 넣고 끓인다 4) 두부를 넣고 마저 끓인다',
                'ingredients': '된장 2큰술, 두부 100g, 호박 50g, 양파 30g'
            },
            '고구마죽': {
                'answer': '고구마죽 만드는 법: 1) 고구마를 삶아서 으깬다 2) 으깬 고구마와 찹쌀을 물과 함께 끓인다 3) 걸쭉해질 때까지 저어가며 끓인다',
                'ingredients': '고구마 200g, 찹쌀 30g, 물 500ml'
            },
            '불고기': {
                'answer': '불고기 만드는 법: 1) 소고기를 얇게 썰어 준비한다 2) 양파와 당근을 채썰고 마늘을 다진다 3) 간장, 설탕, 마늘로 양념장을 만든다 4) 팬에 고기와 야채를 넣고 양념장과 함께 볶는다',
                'ingredients': '소고기 200g, 양파 100g, 당근 50g, 간장 3큰술, 설탕 1큰술'
            },
            '잡채': {
                'answer': '잡채 만드는 법: 1) 당면을 끓는 물에 삶는다 2) 각종 채소를 채썰어 각각 볶는다 3) 소고기를 볶는다 4) 모든 재료를 한데 모아 간장과 참기름으로 무친다',
                'ingredients': '당면 100g, 시금치 50g, 당근 30g, 버섯 30g, 소고기 50g'
            }
        }
        
        self._initialize()
    
    def _initialize(self):
        """시스템 초기화"""
        logger.info("RecipeMaster 초기화 시작")
        
        # 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 데이터 로드
        self._load_data()
        
        logger.info("RecipeMaster 초기화 완료 (키워드 기반 시스템)")
    
    def _load_data(self):
        """레시피 데이터 로드"""
        try:
            # 기본 레시피 데이터 생성
            self.recipes = [
                {
                    "RCP_SEQ": "1",
                    "RCP_NM": "김치찌개",
                    "RCP_WAY2": "끓이기",
                    "RCP_PAT2": "찌개",
                    "INFO_ENG": "180kcal",
                    "RCP_PARTS_DTLS": "김치 200g, 돼지고기 100g, 두부 100g, 대파 20g",
                    "MANUAL01": "김치를 한입 크기로 자른다",
                    "MANUAL02": "돼지고기를 볶아 기름을 낸다",
                    "MANUAL03": "김치를 넣고 함께 볶는다",
                    "MANUAL04": "물을 넣고 끓인 후 두부와 대파를 넣는다"
                },
                {
                    "RCP_SEQ": "2",
                    "RCP_NM": "된장찌개",
                    "RCP_WAY2": "끓이기",
                    "RCP_PAT2": "찌개",
                    "INFO_ENG": "120kcal",
                    "RCP_PARTS_DTLS": "된장 2큰술, 두부 100g, 호박 50g, 양파 30g",
                    "MANUAL01": "멸치로 육수를 우린다",
                    "MANUAL02": "된장을 체에 걸러 육수에 푼다",
                    "MANUAL03": "야채를 넣고 끓인다",
                    "MANUAL04": "두부를 넣고 마저 끓인다"
                },
                {
                    "RCP_SEQ": "3",
                    "RCP_NM": "고구마죽",
                    "RCP_WAY2": "끓이기",
                    "RCP_PAT2": "죽",
                    "INFO_ENG": "150kcal",
                    "RCP_PARTS_DTLS": "고구마 200g, 찹쌀 30g, 물 500ml",
                    "MANUAL01": "고구마를 삶아서 으깬다",
                    "MANUAL02": "으깬 고구마와 찹쌀을 물과 함께 끓인다",
                    "MANUAL03": "걸쭉해질 때까지 저어가며 끓인다"
                },
                {
                    "RCP_SEQ": "4",
                    "RCP_NM": "불고기",
                    "RCP_WAY2": "볶기",
                    "RCP_PAT2": "구이",
                    "INFO_ENG": "280kcal",
                    "RCP_PARTS_DTLS": "소고기 200g, 양파 100g, 당근 50g, 간장 3큰술",
                    "MANUAL01": "소고기를 얇게 썰어 준비한다",
                    "MANUAL02": "양파와 당근을 채썰고 마늘을 다진다",
                    "MANUAL03": "간장, 설탕, 마늘로 양념장을 만든다",
                    "MANUAL04": "팬에 고기와 야채를 넣고 양념장과 함께 볶는다"
                },
                {
                    "RCP_SEQ": "5",
                    "RCP_NM": "잡채",
                    "RCP_WAY2": "볶기",
                    "RCP_PAT2": "나물",
                    "INFO_ENG": "200kcal",
                    "RCP_PARTS_DTLS": "당면 100g, 시금치 50g, 당근 30g, 버섯 30g",
                    "MANUAL01": "당면을 끓는 물에 삶는다",
                    "MANUAL02": "각종 채소를 채썰어 각각 볶는다",
                    "MANUAL03": "소고기를 볶는다",
                    "MANUAL04": "모든 재료를 한데 모아 간장과 참기름으로 무친다"
                }
            ]
            
            logger.info(f"레시피 데이터 로드: {len(self.recipes)}개")
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            self.recipes = []
    
    def get_recipe_data(self, query: str = "") -> List[Dict]:
        """레시피 데이터 검색"""
        if not query:
            return self.recipes
        
        # 키워드 기반 검색
        query_lower = query.lower()
        filtered_recipes = []
        
        for recipe in self.recipes:
            recipe_name = recipe.get('RCP_NM', '').lower()
            ingredients = recipe.get('RCP_PARTS_DTLS', '').lower()
            category = recipe.get('RCP_PAT2', '').lower()
            
            if (query_lower in recipe_name or 
                query_lower in ingredients or 
                query_lower in category):
                filtered_recipes.append(recipe)
        
        return filtered_recipes
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """키워드 기반 질문 답변"""
        question_lower = question.lower()
        
        # 키워드 매칭
        best_match = None
        best_score = 0
        
        for keyword, response in self.keyword_responses.items():
            if keyword in question_lower:
                score = question_lower.count(keyword)
                if score > best_score:
                    best_score = score
                    best_match = response
        
        # 질문 유형별 답변
        if best_match:
            if '재료' in question_lower or '성분' in question_lower:
                answer = f"재료: {best_match['ingredients']}"
            else:
                answer = best_match['answer']
            confidence = 0.9
        else:
            # 일반적인 답변
            if '끓이' in question_lower:
                answer = "끓이는 요리로는 김치찌개, 된장찌개, 고구마죽 등이 있습니다."
            elif '볶' in question_lower:
                answer = "볶는 요리로는 불고기, 잡채 등이 있습니다."
            elif '찌개' in question_lower:
                answer = "찌개 종류로는 김치찌개, 된장찌개 등이 있습니다."
            elif '칼로리' in question_lower:
                answer = "요리별 칼로리 정보를 확인하시려면 구체적인 요리명을 말씀해주세요."
            else:
                answer = "죄송합니다. 김치찌개, 된장찌개, 고구마죽, 불고기, 잡채에 대해 질문해주세요."
            confidence = 0.5
        
        return {
            "answer": answer,
            "confidence": confidence
        }
    
    def get_enhanced_answer(self, question: str) -> Dict[str, Any]:
        """향상된 답변 제공"""
        try:
            # 키워드 추출
            keywords = self._extract_keywords(question)
            
            # 관련 레시피 검색
            related_recipes = []
            for keyword in keywords:
                recipes = self.get_recipe_data(keyword)
                related_recipes.extend(recipes)
            
            # 중복 제거
            seen_ids = set()
            unique_recipes = []
            for recipe in related_recipes:
                recipe_id = recipe.get('RCP_SEQ')
                if recipe_id not in seen_ids:
                    seen_ids.add(recipe_id)
                    unique_recipes.append(recipe)
            
            # 답변 생성
            result = self.answer_question(question)
            
            return {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "related_recipes": unique_recipes[:3],
                "keywords": keywords,
                "total_recipes": len(self.recipes)
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "answer": "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "related_recipes": [],
                "keywords": [],
                "total_recipes": len(self.recipes)
            }
    
    def _extract_keywords(self, question: str) -> List[str]:
        """질문에서 키워드 추출"""
        keywords = []
        question_lower = question.lower()
        
        # 요리명 키워드
        food_names = ['김치찌개', '된장찌개', '고구마죽', '불고기', '잡채']
        for food in food_names:
            if food in question_lower:
                keywords.append(food)
        
        # 조리법 키워드
        cooking_methods = ['끓이기', '볶기', '찌기', '굽기']
        for method in cooking_methods:
            if method in question_lower or method[:-1] in question_lower:
                keywords.append(method)
        
        # 카테고리 키워드
        categories = ['찌개', '죽', '구이', '나물']
        for category in categories:
            if category in question_lower:
                keywords.append(category)
        
        return keywords

# 전역 RecipeMaster 인스턴스
recipe_master = SimpleRecipeMaster()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """질문 API 엔드포인트"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "질문을 입력해주세요.",
                "success": False
            }), 400
        
        if len(question) > 200:
            return jsonify({
                "error": "질문은 200자 이내로 입력해주세요.",
                "success": False
            }), 400
        
        # 답변 생성
        result = recipe_master.get_enhanced_answer(question)
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "related_recipes": result["related_recipes"],
            "keywords": result["keywords"],
            "total_recipes": result["total_recipes"],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}",
            "success": False
        }), 500

@app.route('/api/recipes')
def get_recipes():
    """레시피 목록 API"""
    try:
        query = request.args.get('q', '')
        recipes = recipe_master.get_recipe_data(query)
        
        return jsonify({
            "success": True,
            "recipes": recipes,
            "count": len(recipes)
        })
        
    except Exception as e:
        logger.error(f"레시피 조회 실패: {e}")
        return jsonify({
            "error": f"레시피 조회 실패: {str(e)}",
            "success": False
        }), 500

@app.route('/api/stats')
def get_statistics():
    """시스템 통계 API"""
    try:
        stats = {
            "total_recipes": len(recipe_master.recipes),
            "categories": {},
            "cooking_methods": {},
            "system_type": "keyword_based"
        }
        
        # 카테고리별 통계
        for recipe in recipe_master.recipes:
            category = recipe.get('RCP_PAT2', '기타')
            method = recipe.get('RCP_WAY2', '기타')
            
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["cooking_methods"][method] = stats["cooking_methods"].get(method, 0) + 1
        
        return jsonify({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({
            "error": f"통계 조회 실패: {str(e)}",
            "success": False
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "페이지를 찾을 수 없습니다.",
        "success": False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "내부 서버 오류가 발생했습니다.",
        "success": False
    }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("한국 전통요리 레시피 마스터 시작!")
    print("웹 브라우저에서 http://localhost:5000 접속")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)