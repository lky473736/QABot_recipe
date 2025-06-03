"""
개선된 QA 데이터셋 생성기
- 대용량 고품질 QA 생성
- 다양한 질문 패턴
- 챗봇 학습에 최적화된 구조
"""
import json
import random
from typing import List, Dict, Any, Tuple
import sys
import os
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class EnhancedQAGenerator:
    def __init__(self):
        # 확장된 질문 템플릿
        self.question_templates = {
            'recipe_search': [
                "{ingredient}로 뭐 만들 수 있어?",
                "{ingredient} 요리 레시피 알려줘",
                "{ingredient}를 사용한 음식 추천해줘", 
                "{ingredient} 넣어서 뭐 만들까?",
                "{ingredient}가 들어간 요리는?",
                "{ingredient}로 만드는 음식은?",
                "{ingredient} 활용 요리법",
                "{ingredient} 들어간 반찬",
                "{ingredient}로 간단한 요리",
                "{ingredient} 요리 종류"
            ],
            'cooking_method': [
                "{recipe_name} 어떻게 만들어?",
                "{recipe_name} 만드는 법 알려줘",
                "{recipe_name} 조리법이 궁금해",
                "{recipe_name} 레시피 가르쳐줘",
                "{recipe_name} 만들기 어려워?",
                "{recipe_name} 조리 과정",
                "{recipe_name} 요리 방법",
                "{recipe_name} 만드는 순서",
                "{recipe_name} 어떻게 요리해?",
                "{recipe_name} 조리 단계"
            ],
            'ingredients': [
                "{recipe_name}에 뭐가 들어가?",
                "{recipe_name} 재료가 뭐야?",
                "{recipe_name} 만들 때 필요한 재료",
                "{recipe_name}의 재료를 알고싶어",
                "{recipe_name} 재료 리스트",
                "{recipe_name} 주재료는?",
                "{recipe_name} 들어가는 재료",
                "{recipe_name} 필요한 것들",
                "{recipe_name} 재료 목록",
                "{recipe_name} 사용 재료"
            ],
            'nutrition': [
                "{recipe_name} 칼로리가 얼마야?",
                "{recipe_name} 영양정보 알려줘",
                "{recipe_name}의 영양성분이 궁금해",
                "{recipe_name} 열량은?",
                "{recipe_name} 건강에 어때?",
                "{recipe_name} 영양소",
                "{recipe_name} 칼로리 정보",
                "{recipe_name} 영양 분석",
                "{recipe_name} 다이어트에 좋아?",
                "{recipe_name} 영양가"
            ],
            'tips': [
                "{recipe_name} 만들 때 팁 있어?",
                "{recipe_name} 조리 팁 알려줘",
                "{recipe_name} 맛있게 만드는 비법",
                "{recipe_name} 요리할 때 주의사항",
                "{recipe_name} 실패하지 않으려면?",
                "{recipe_name} 요리 노하우",
                "{recipe_name} 잘 만드는 방법",
                "{recipe_name} 비법 있어?",
                "{recipe_name} 팁 좀 줘",
                "{recipe_name} 요리 꿀팁"
            ],
            'category': [
                "{category} 요리 추천해줘",
                "{category} 음식 뭐가 있어?",
                "{category} 레시피 알려줘",
                "오늘은 {category} 먹고싶어",
                "{category} 종류 알려줘",
                "{category} 메뉴 추천",
                "{category} 요리법",
                "{category} 만들기",
                "{category} 음식 종류",
                "{category} 뭐 해먹을까?"
            ],
            'cooking_method_search': [
                "{method} 요리 뭐가 있어?",
                "{method}으로 만드는 음식",
                "{method} 요리법 알려줘",
                "{method} 음식 추천",
                "{method} 요리 종류",
                "{method}으로 뭐 만들까?",
                "{method} 레시피",
                "{method} 음식들",
                "{method} 요리 목록",
                "{method} 메뉴"
            ],
            'difficulty': [
                "쉬운 요리 추천해줘",
                "간단한 레시피 알려줘", 
                "초보도 할 수 있는 요리",
                "어려운 요리는?",
                "복잡한 레시피",
                "시간 오래 걸리는 요리",
                "빠르게 만들 수 있는 음식",
                "10분 요리",
                "간단 요리법",
                "쉬운 반찬"
            ]
        }
        
        # 일반적인 대화 QA
        self.general_qa = [
            {
                'question': '안녕하세요',
                'answer': '안녕하세요! 레시피 챗봇입니다. 요리 레시피나 재료에 대해 궁금한 것이 있으시면 언제든 물어보세요! 🍳',
                'type': 'greeting'
            },
            {
                'question': '안녕',
                'answer': '안녕하세요! 오늘은 어떤 요리를 만들어보고 싶으신가요?',
                'type': 'greeting'
            },
            {
                'question': '뭐 해줄 수 있어?',
                'answer': '저는 다음과 같은 도움을 드릴 수 있어요:\n• 재료로 요리 추천\n• 레시피 조리법 안내\n• 요리 재료 정보\n• 영양정보 제공\n• 조리 팁 공유\n무엇을 도와드릴까요?',
                'type': 'help'
            },
            {
                'question': '도움말',
                'answer': '레시피 챗봇 사용법:\n\n1. "감자로 뭐 만들 수 있어?" - 재료로 요리 검색\n2. "김치찌개 만드는 법" - 특정 요리 레시피\n3. "불고기 재료가 뭐야?" - 요리 재료 확인\n4. "계란말이 칼로리" - 영양정보 확인\n\n편하게 물어보세요!',
                'type': 'help'
            },
            {
                'question': '오늘 뭐 먹을까?',
                'answer': '맛있는 요리를 추천해드릴게요! 어떤 재료가 있으신가요? 또는 어떤 종류의 음식을 드시고 싶으신지 알려주세요.',
                'type': 'recommendation'
            },
            {
                'question': '간단한 요리',
                'answer': '간단하게 만들 수 있는 요리들을 추천해드릴게요:\n• 계란말이\n• 김치볶음밥\n• 라면\n• 토스트\n• 샐러드\n어떤 재료로 만들고 싶으신가요?',
                'type': 'recommendation'
            }
        ]
        
    def load_enhanced_recipes(self, filepath: str) -> List[Dict[str, Any]]:
        """개선된 레시피 데이터 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 레시피 파일 로드 성공: {filepath}")
            
            recipes = []
            if isinstance(data, dict):
                if 'recipes' in data:
                    recipes = data['recipes']
                    if 'metadata' in data:
                        print(f"📈 메타데이터: {data['metadata']}")
                    if 'statistics' in data:
                        print(f"📊 통계 정보 포함")
                else:
                    # 기존 구조 지원
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict):
                                recipes = value
                                break
            elif isinstance(data, list):
                recipes = data
            
            # 유효한 레시피만 필터링
            valid_recipes = []
            for recipe in recipes:
                if isinstance(recipe, dict) and recipe.get('name') and recipe.get('main_ingredients'):
                    valid_recipes.append(recipe)
            
            print(f"🍳 유효한 레시피: {len(valid_recipes)}개")
            return valid_recipes
            
        except Exception as e:
            print(f"❌ 레시피 로드 실패: {e}")
            return []
    
    def generate_recipe_search_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """재료 기반 레시피 검색 QA 생성"""
        qa_pairs = []
        
        print(f"🔍 재료 검색 QA 생성 중...")
        
        # 재료별 레시피 매핑
        ingredient_recipes = defaultdict(list)
        for recipe in recipes:
            main_ingredients = recipe.get('main_ingredients', [])
            for ingredient in main_ingredients:
                if ingredient and len(ingredient) >= 2:
                    ingredient_recipes[ingredient].append(recipe)
        
        print(f"📊 발견된 재료: {len(ingredient_recipes)}개")
        
        # 각 재료에 대해 다양한 질문 생성
        for ingredient, recipe_list in ingredient_recipes.items():
            if len(recipe_list) >= 1:  # 최소 1개 이상의 레시피
                # 여러 템플릿 사용
                for template in self.question_templates['recipe_search']:
                    question = template.format(ingredient=ingredient)
                    
                    # 추천 레시피 선택 (최대 5개)
                    recommended = random.sample(recipe_list, min(5, len(recipe_list)))
                    answer_parts = [f"{ingredient}로 만들 수 있는 요리들을 추천해드릴게요:\n"]
                    
                    for i, recipe in enumerate(recommended, 1):
                        recipe_name = recipe.get('name', '알 수 없는 요리')
                        category = recipe.get('category', '')
                        cooking_method = recipe.get('cooking_method', '')
                        
                        if category and cooking_method:
                            answer_parts.append(f"{i}. {recipe_name} ({category}, {cooking_method})")
                        else:
                            answer_parts.append(f"{i}. {recipe_name}")
                    
                    answer = "\n".join(answer_parts)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'recipe_search',
                        'ingredient': ingredient,
                        'related_recipes': [r.get('id', '') for r in recommended]
                    })
        
        print(f"✅ 재료 검색 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_cooking_method_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """조리법 QA 생성"""
        qa_pairs = []
        
        print(f"👨‍🍳 조리법 QA 생성 중...")
        
        for recipe in recipes:
            recipe_name = recipe.get('name', '')
            steps = recipe.get('steps', [])
            
            if not recipe_name or not steps:
                continue
            
            # 여러 템플릿으로 질문 생성
            for template in self.question_templates['cooking_method']:
                question = template.format(recipe_name=recipe_name)
                
                # 조리법 답변 생성
                if steps:
                    steps_text = [f"{recipe_name} 만드는 방법을 알려드릴게요:\n"]
                    for i, step in enumerate(steps[:8], 1):  # 최대 8단계
                        if step.strip():
                            steps_text.append(f"{i}. {step.strip()}")
                    
                    # 추가 정보 포함
                    category = recipe.get('category', '')
                    cooking_method = recipe.get('cooking_method', '')
                    if category:
                        steps_text.append(f"\n카테고리: {category}")
                    if cooking_method:
                        steps_text.append(f"조리방법: {cooking_method}")
                    
                    answer = "\n".join(steps_text)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'cooking_method',
                        'recipe_name': recipe_name,
                        'related_recipes': [recipe.get('id', '')]
                    })
        
        print(f"✅ 조리법 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_ingredients_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """재료 QA 생성"""
        qa_pairs = []
        
        print(f"📋 재료 QA 생성 중...")
        
        for recipe in recipes:
            recipe_name = recipe.get('name', '')
            ingredients = recipe.get('ingredients', '')
            main_ingredients = recipe.get('main_ingredients', [])
            
            if not recipe_name or (not ingredients and not main_ingredients):
                continue
            
            for template in self.question_templates['ingredients']:
                question = template.format(recipe_name=recipe_name)
                
                # 재료 정보 구성
                answer_parts = [f"{recipe_name}의 재료는 다음과 같아요:\n"]
                
                if main_ingredients:
                    answer_parts.append("주요 재료:")
                    for ingredient in main_ingredients:
                        answer_parts.append(f"• {ingredient}")
                
                if ingredients and ingredients != ' '.join(main_ingredients):
                    answer_parts.append(f"\n상세 재료:\n{ingredients}")
                
                answer = "\n".join(answer_parts)
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'ingredients',
                    'recipe_name': recipe_name,
                    'related_recipes': [recipe.get('id', '')]
                })
        
        print(f"✅ 재료 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_nutrition_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """영양정보 QA 생성"""
        qa_pairs = []
        
        print(f"📊 영양정보 QA 생성 중...")
        
        for recipe in recipes:
            recipe_name = recipe.get('name', '')
            nutrition = recipe.get('nutrition', {})
            
            if not recipe_name or not nutrition:
                continue
            
            for template in self.question_templates['nutrition']:
                question = template.format(recipe_name=recipe_name)
                
                # 영양정보 구성
                nutrition_info = [f"{recipe_name}의 영양정보는 다음과 같아요:\n"]
                
                nutrition_labels = {
                    'calories': '칼로리',
                    'carbs': '탄수화물', 
                    'protein': '단백질',
                    'fat': '지방',
                    'sodium': '나트륨'
                }
                
                units = {
                    'calories': 'kcal',
                    'carbs': 'g',
                    'protein': 'g', 
                    'fat': 'g',
                    'sodium': 'mg'
                }
                
                for key, label in nutrition_labels.items():
                    if key in nutrition:
                        unit = units.get(key, '')
                        nutrition_info.append(f"• {label}: {nutrition[key]}{unit}")
                
                if len(nutrition_info) > 1:  # 실제 영양정보가 있는 경우만
                    answer = "\n".join(nutrition_info)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'nutrition',
                        'recipe_name': recipe_name,
                        'related_recipes': [recipe.get('id', '')]
                    })
        
        print(f"✅ 영양정보 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_category_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """카테고리별 요리 QA 생성"""
        qa_pairs = []
        
        print(f"🗂️ 카테고리 QA 생성 중...")
        
        # 카테고리별 레시피 매핑
        category_recipes = defaultdict(list)
        for recipe in recipes:
            category = recipe.get('category', '기타')
            if category:
                category_recipes[category].append(recipe)
        
        print(f"📊 발견된 카테고리: {len(category_recipes)}개")
        
        for category, recipe_list in category_recipes.items():
            if len(recipe_list) >= 2:  # 최소 2개 이상
                for template in self.question_templates['category']:
                    question = template.format(category=category)
                    
                    # 추천 레시피 선택
                    recommended = random.sample(recipe_list, min(7, len(recipe_list)))
                    answer_parts = [f"{category} 요리를 추천해드릴게요:\n"]
                    
                    for i, recipe in enumerate(recommended, 1):
                        recipe_name = recipe.get('name', '알 수 없는 요리')
                        cooking_method = recipe.get('cooking_method', '')
                        main_ingredients = recipe.get('main_ingredients', [])
                        
                        if cooking_method and main_ingredients:
                            ingredients_str = ', '.join(main_ingredients[:2])
                            answer_parts.append(f"{i}. {recipe_name} ({cooking_method}, {ingredients_str})")
                        else:
                            answer_parts.append(f"{i}. {recipe_name}")
                    
                    answer = "\n".join(answer_parts)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'category',
                        'category': category,
                        'related_recipes': [r.get('id', '') for r in recommended]
                    })
        
        print(f"✅ 카테고리 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_cooking_method_search_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """조리방법별 검색 QA 생성"""
        qa_pairs = []
        
        print(f"🔥 조리방법 검색 QA 생성 중...")
        
        # 조리방법별 레시피 매핑
        method_recipes = defaultdict(list)
        for recipe in recipes:
            method = recipe.get('cooking_method', '기타')
            if method and method != '기타':
                method_recipes[method].append(recipe)
        
        print(f"📊 발견된 조리방법: {len(method_recipes)}개")
        
        for method, recipe_list in method_recipes.items():
            if len(recipe_list) >= 2:
                for template in self.question_templates['cooking_method_search']:
                    question = template.format(method=method)
                    
                    recommended = random.sample(recipe_list, min(6, len(recipe_list)))
                    answer_parts = [f"{method} 요리들을 추천해드릴게요:\n"]
                    
                    for i, recipe in enumerate(recommended, 1):
                        recipe_name = recipe.get('name', '알 수 없는 요리')
                        category = recipe.get('category', '')
                        if category:
                            answer_parts.append(f"{i}. {recipe_name} ({category})")
                        else:
                            answer_parts.append(f"{i}. {recipe_name}")
                    
                    answer = "\n".join(answer_parts)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'cooking_method_search',
                        'cooking_method': method,
                        'related_recipes': [r.get('id', '') for r in recommended]
                    })
        
        print(f"✅ 조리방법 검색 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_all_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 유형의 QA 생성"""
        print(f"\n🚀 대용량 QA 생성을 시작합니다... (총 레시피: {len(recipes)}개)")
        
        all_qa = []
        
        try:
            # 각 유형별 QA 생성
            print("\n1️⃣ 재료 검색 QA 생성...")
            all_qa.extend(self.generate_recipe_search_qa(recipes))
            
            print("\n2️⃣ 조리법 QA 생성...")
            all_qa.extend(self.generate_cooking_method_qa(recipes))
            
            print("\n3️⃣ 재료 정보 QA 생성...")
            all_qa.extend(self.generate_ingredients_qa(recipes))
            
            print("\n4️⃣ 영양정보 QA 생성...")
            all_qa.extend(self.generate_nutrition_qa(recipes))
            
            print("\n5️⃣ 카테고리 QA 생성...")
            all_qa.extend(self.generate_category_qa(recipes))
            
            print("\n6️⃣ 조리방법 검색 QA 생성...")
            all_qa.extend(self.generate_cooking_method_search_qa(recipes))
            
            print("\n7️⃣ 일반 QA 추가...")
            all_qa.extend(self.general_qa)
            
        except Exception as e:
            print(f"❌ QA 생성 중 오류: {e}")
            return []
        
        # 중복 제거
        unique_qa = []
        seen_questions = set()
        
        for qa in all_qa:
            question = qa.get('question', '')
            if question and question not in seen_questions:
                unique_qa.append(qa)
                seen_questions.add(question)
        
        # 셔플
        random.shuffle(unique_qa)
        
        print(f"\n🎉 QA 생성 완료: {len(unique_qa)}개 (중복 제거 전: {len(all_qa)}개)")
        
        # 유형별 통계
        type_counts = defaultdict(int)
        for qa in unique_qa:
            qa_type = qa.get('type', 'unknown')
            type_counts[qa_type] += 1
        
        print(f"\n📊 유형별 분포:")
        for qa_type, count in sorted(type_counts.items()):
            print(f"   {qa_type}: {count}개")
        
        return unique_qa
    
    def save_enhanced_qa_dataset(self, qa_data: List[Dict[str, Any]], filepath: str):
        """개선된 QA 데이터셋 저장"""
        # 상세 통계 생성
        type_counts = defaultdict(int)
        question_lengths = []
        answer_lengths = []
        
        for qa in qa_data:
            qa_type = qa.get('type', 'unknown')
            type_counts[qa_type] += 1
            
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            question_lengths.append(len(question))
            answer_lengths.append(len(answer))
        
        # 메타데이터 생성
        metadata = {
            'generation_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'total_qa_pairs': len(qa_data),
            'generation_version': '3.0_enhanced',
            'features': [
                'multi_template_questions',
                'detailed_answers',
                'recipe_categorization',
                'ingredient_mapping',
                'nutrition_info',
                'cooking_methods'
            ],
            'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        }
        
        statistics = {
            'type_distribution': dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
            'question_length_stats': {
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0,
                'avg': metadata['avg_question_length']
            },
            'answer_length_stats': {
                'min': min(answer_lengths) if answer_lengths else 0,
                'max': max(answer_lengths) if answer_lengths else 0,
                'avg': metadata['avg_answer_length']
            }
        }
        
        # 최종 데이터 구조
        enhanced_qa_data = {
            'metadata': metadata,
            'statistics': statistics,
            'qa_pairs': qa_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_qa_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 QA 데이터셋 저장 완료: {filepath}")
        print(f"   총 QA: {len(qa_data)}개")
        print(f"   평균 질문 길이: {metadata['avg_question_length']:.1f}자")
        print(f"   평균 답변 길이: {metadata['avg_answer_length']:.1f}자")

def main():
    """메인 실행 함수"""
    print("🚀 개선된 QA 생성을 시작합니다...")
    
    if not PROCESSED_RECIPES_PATH.exists():
        print(f"❌ 처리된 레시피 파일을 찾을 수 없습니다: {PROCESSED_RECIPES_PATH}")
        print("먼저 enhanced_data_processor.py를 실행해주세요.")
        return
    
    generator = EnhancedQAGenerator()
    
    # 레시피 로드
    recipes = generator.load_enhanced_recipes(PROCESSED_RECIPES_PATH)
    
    if not recipes:
        print("❌ 유효한 레시피 데이터를 찾을 수 없습니다.")
        return
    
    # QA 생성
    qa_dataset = generator.generate_all_qa(recipes)
    
    if qa_dataset:
        # 저장
        generator.save_enhanced_qa_dataset(qa_dataset, QA_DATASET_PATH)
        
        # 샘플 출력
        print(f"\n📋 샘플 QA:")
        for i, qa in enumerate(qa_dataset[:5]):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            qa_type = qa.get('type', '')
            
            print(f"\n{i+1}. [{qa_type}]")
            print(f"   Q: {question}")
            print(f"   A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
    
    else:
        print("❌ QA 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
