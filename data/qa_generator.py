"""
수정된 레시피 QA 데이터셋 생성기 - 다양한 데이터 구조 지원
"""
import json
import random
from typing import List, Dict, Any, Tuple, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class FixedQAGenerator:
    def __init__(self):
        self.question_templates = {
            'recipe_search': [
                "{ingredient}로 만들 수 있는 요리가 뭐가 있어?",
                "{ingredient} 요리 레시피 알려줘",
                "{ingredient}를 사용한 음식 추천해줘",
                "{ingredient} 넣어서 뭐 만들 수 있을까?",
                "{ingredient}가 들어간 요리 뭐가 있지?",
            ],
            'cooking_method': [
                "{recipe_name} 어떻게 만들어?",
                "{recipe_name} 만드는 법 알려줘",
                "{recipe_name} 조리법이 궁금해",
                "{recipe_name} 레시피 가르쳐줘",
                "{recipe_name} 만들기 어려워?",
            ],
            'ingredients': [
                "{recipe_name}에 뭐가 들어가?",
                "{recipe_name} 재료가 뭐야?",
                "{recipe_name} 만들 때 필요한 재료 알려줘",
                "{recipe_name}의 재료를 알고싶어",
                "{recipe_name} 재료 리스트 줘",
            ],
            'nutrition': [
                "{recipe_name} 칼로리가 얼마야?",
                "{recipe_name} 영양정보 알려줘",
                "{recipe_name}의 영양성분이 궁금해",
                "{recipe_name} 열량은?",
                "{recipe_name} 건강에 어때?",
            ],
            'tips': [
                "{recipe_name} 만들 때 팁 있어?",
                "{recipe_name} 조리 팁 알려줘",
                "{recipe_name} 맛있게 만드는 비법은?",
                "{recipe_name} 요리할 때 주의사항은?",
                "{recipe_name} 실패하지 않으려면?",
            ],
            'category': [
                "{category} 요리 추천해줘",
                "{category} 음식 뭐가 있어?",
                "{category} 레시피 알려줘",
                "오늘은 {category} 먹고싶어",
                "{category} 종류 알려줘",
            ]
        }
    
    def load_recipes(self, filepath: str) -> List[Dict[str, Any]]:
        """레시피 데이터 로드 - 다양한 구조 지원"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 레시피 파일 로드 성공: {filepath}")
            
            # 데이터 구조 분석 및 추출
            recipes = []
            
            if isinstance(data, dict):
                if 'metadata' in data and 'recipes' in data:
                    # 메타데이터가 있는 구조
                    recipes = data['recipes']
                    print(f"✅ 메타데이터 구조에서 레시피 추출")
                else:
                    # 기타 딕셔너리 구조
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict):
                                recipes = value
                                print(f"✅ '{key}' 키에서 레시피 배열 추출")
                                break
                    
                    if not recipes:
                        recipes = [data]  # 단일 레시피
                        
            elif isinstance(data, list):
                recipes = data
                print(f"✅ 리스트 구조에서 레시피 추출")
            else:
                print(f"❌ 지원하지 않는 데이터 타입: {type(data)}")
                return []
            
            # 유효한 레시피만 필터링
            valid_recipes = []
            for i, recipe in enumerate(recipes):
                if isinstance(recipe, dict):
                    # 필수 필드 확인
                    if recipe.get('name') and recipe.get('id'):
                        valid_recipes.append(recipe)
                    else:
                        print(f"⚠️ 레시피 {i}: 필수 필드 누락")
                elif isinstance(recipe, str):
                    print(f"⚠️ 레시피 {i}: 문자열 형태 - JSON 파싱 시도")
                    try:
                        parsed_recipe = json.loads(recipe)
                        if isinstance(parsed_recipe, dict) and parsed_recipe.get('name'):
                            valid_recipes.append(parsed_recipe)
                    except:
                        print(f"❌ 레시피 {i}: JSON 파싱 실패")
                else:
                    print(f"❌ 레시피 {i}: 예상치 못한 타입 {type(recipe)}")
            
            print(f"🍳 유효한 레시피: {len(valid_recipes)}개")
            return valid_recipes
            
        except Exception as e:
            print(f"❌ 레시피 로드 실패: {e}")
            return []
    
    def extract_main_ingredients(self, ingredients_text: Union[str, None]) -> List[str]:
        """주요 재료 추출 - None 안전 처리"""
        if not ingredients_text or ingredients_text in ['None', '']:
            return []
        
        # 문자열로 변환
        ingredients_text = str(ingredients_text)
        
        # 일반적인 주요 재료들
        main_ingredients = []
        ingredients_list = ingredients_text.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
        
        # 주요 재료 키워드
        important_keywords = [
            '쇠고기', '돼지고기', '닭고기', '생선', '새우', '오징어', '두부', '계란', '달걀',
            '쌀', '면', '국수', '밀가루', '감자', '고구마', '양파', '마늘', '대파', '파',
            '배추', '무', '당근', '호박', '브로콜리', '시금치', '버섯', '김치',
            '콩나물', '미역', '다시마', '치즈', '우유', '요구르트', '연두부', '순두부'
        ]
        
        for item in ingredients_list:
            for keyword in important_keywords:
                if keyword in item and len(item) <= len(keyword) + 3:  # 너무 긴 단어 제외
                    main_ingredients.append(keyword)
                    break
        
        return list(set(main_ingredients))[:3]  # 중복 제거 후 최대 3개
    
    def generate_recipe_search_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """재료 기반 레시피 검색 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"🔍 재료 검색 QA 생성 중... (레시피 {len(recipes)}개)")
        
        # 재료별 레시피 매핑
        ingredient_recipes = {}
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            ingredients_text = recipe.get('ingredients', '')
            ingredients = self.extract_main_ingredients(ingredients_text)
            
            for ingredient in ingredients:
                if ingredient not in ingredient_recipes:
                    ingredient_recipes[ingredient] = []
                ingredient_recipes[ingredient].append(recipe)
        
        print(f"📊 발견된 재료: {len(ingredient_recipes)}개")
        
        # QA 생성
        for ingredient, recipe_list in ingredient_recipes.items():
            if len(recipe_list) >= 1:  # 최소 1개 이상의 레시피가 있는 재료
                for template in self.question_templates['recipe_search']:
                    question = template.format(ingredient=ingredient)
                    
                    # 추천할 레시피들 선택 (최대 3개)
                    recommended = random.sample(recipe_list, min(3, len(recipe_list)))
                    answer_parts = []
                    
                    for recipe in recommended:
                        recipe_name = recipe.get('name', '알 수 없는 요리')
                        answer_parts.append(f"• {recipe_name}")
                    
                    answer = f"{ingredient}로 만들 수 있는 요리들을 추천해드릴게요:\n" + "\n".join(answer_parts)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'recipe_search',
                        'related_recipes': [r.get('id', '') for r in recommended]
                    })
        
        print(f"✅ 재료 검색 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_cooking_method_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """조리법 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"👨‍🍳 조리법 QA 생성 중...")
        
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            recipe_name = recipe.get('name', '')
            steps = recipe.get('steps', [])
            
            if not recipe_name or not steps:
                continue
                
            for template in self.question_templates['cooking_method']:
                question = template.format(recipe_name=recipe_name)
                
                # 조리법 답변 생성
                if isinstance(steps, list) and steps:
                    steps_text = []
                    for i, step in enumerate(steps[:8], 1):  # 최대 8단계
                        if isinstance(step, str) and step.strip():
                            steps_text.append(f"{i}. {step.strip()}")
                    
                    if steps_text:
                        answer = f"{recipe_name} 만드는 방법을 알려드릴게요:\n\n" + "\n".join(steps_text)
                        
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'type': 'cooking_method',
                            'related_recipes': [recipe.get('id', '')]
                        })
        
        print(f"✅ 조리법 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_ingredients_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """재료 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"📋 재료 QA 생성 중...")
        
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            recipe_name = recipe.get('name', '')
            ingredients = recipe.get('ingredients', '')
            
            if not recipe_name or not ingredients:
                continue
                
            for template in self.question_templates['ingredients']:
                question = template.format(recipe_name=recipe_name)
                answer = f"{recipe_name}의 재료는 다음과 같아요:\n\n{ingredients}"
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'ingredients',
                    'related_recipes': [recipe.get('id', '')]
                })
        
        print(f"✅ 재료 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_nutrition_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """영양정보 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"📊 영양정보 QA 생성 중...")
        
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            recipe_name = recipe.get('name', '')
            if not recipe_name:
                continue
            
            # 영양정보가 있는 레시피만
            nutrition_info = []
            if recipe.get('calories'):
                nutrition_info.append(f"칼로리: {recipe['calories']}kcal")
            if recipe.get('carbs'):
                nutrition_info.append(f"탄수화물: {recipe['carbs']}g")
            if recipe.get('protein'):
                nutrition_info.append(f"단백질: {recipe['protein']}g")
            if recipe.get('fat'):
                nutrition_info.append(f"지방: {recipe['fat']}g")
            if recipe.get('sodium'):
                nutrition_info.append(f"나트륨: {recipe['sodium']}mg")
            
            if nutrition_info:
                for template in self.question_templates['nutrition']:
                    question = template.format(recipe_name=recipe_name)
                    answer = f"{recipe_name}의 영양정보는 다음과 같아요:\n\n" + "\n".join(nutrition_info)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'nutrition',
                        'related_recipes': [recipe.get('id', '')]
                    })
        
        print(f"✅ 영양정보 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_tips_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """조리 팁 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"💡 조리 팁 QA 생성 중...")
        
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            recipe_name = recipe.get('name', '')
            tip = recipe.get('tip', '')
            
            if not recipe_name or not tip:
                continue
                
            for template in self.question_templates['tips']:
                question = template.format(recipe_name=recipe_name)
                answer = f"{recipe_name} 조리 팁을 알려드릴게요:\n\n{tip}"
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'tips',
                    'related_recipes': [recipe.get('id', '')]
                })
        
        print(f"✅ 조리 팁 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_category_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """카테고리별 요리 QA 생성 - 안전 처리"""
        qa_pairs = []
        
        print(f"🗂️ 카테고리 QA 생성 중...")
        
        # 카테고리별 레시피 매핑
        category_recipes = {}
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
                
            category = recipe.get('category', '기타')
            if category and category != '':
                if category not in category_recipes:
                    category_recipes[category] = []
                category_recipes[category].append(recipe)
        
        print(f"📊 발견된 카테고리: {len(category_recipes)}개")
        
        # QA 생성
        for category, recipe_list in category_recipes.items():
            if len(recipe_list) >= 2:  # 최소 2개 이상의 레시피가 있는 카테고리만
                for template in self.question_templates['category']:
                    question = template.format(category=category)
                    
                    # 추천할 레시피들 선택 (최대 5개)
                    recommended = random.sample(recipe_list, min(5, len(recipe_list)))
                    answer_parts = []
                    
                    for recipe in recommended:
                        recipe_name = recipe.get('name', '알 수 없는 요리')
                        answer_parts.append(f"• {recipe_name}")
                    
                    answer = f"{category} 요리를 추천해드릴게요:\n" + "\n".join(answer_parts)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'category',
                        'related_recipes': [r.get('id', '') for r in recommended]
                    })
        
        print(f"✅ 카테고리 QA {len(qa_pairs)}개 생성")
        return qa_pairs
    
    def generate_general_qa(self) -> List[Dict[str, Any]]:
        """일반적인 QA 생성"""
        general_qa = [
            {
                'question': '안녕하세요',
                'answer': '안녕하세요! 레시피 챗봇입니다. 요리 레시피나 재료에 대해 궁금한 것이 있으시면 언제든 물어보세요!',
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
            }
        ]
        
        return general_qa
    
    def generate_all_qa(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 유형의 QA 생성 - 안전 처리"""
        print(f"\n🚀 QA 생성을 시작합니다... (총 레시피: {len(recipes)}개)")
        
        all_qa = []
        
        try:
            print("\n1️⃣ 레시피 검색 QA 생성...")
            all_qa.extend(self.generate_recipe_search_qa(recipes))
            
            print("\n2️⃣ 조리법 QA 생성...")
            all_qa.extend(self.generate_cooking_method_qa(recipes))
            
            print("\n3️⃣ 재료 QA 생성...")
            all_qa.extend(self.generate_ingredients_qa(recipes))
            
            print("\n4️⃣ 영양정보 QA 생성...")
            all_qa.extend(self.generate_nutrition_qa(recipes))
            
            print("\n5️⃣ 조리 팁 QA 생성...")
            all_qa.extend(self.generate_tips_qa(recipes))
            
            print("\n6️⃣ 카테고리 QA 생성...")
            all_qa.extend(self.generate_category_qa(recipes))
            
            print("\n7️⃣ 일반 QA 추가...")
            all_qa.extend(self.generate_general_qa())
            
        except Exception as e:
            print(f"❌ QA 생성 중 오류: {e}")
            return []
        
        # 중복 제거 및 셔플
        unique_qa = []
        seen_questions = set()
        
        for qa in all_qa:
            question = qa.get('question', '')
            if question and question not in seen_questions:
                unique_qa.append(qa)
                seen_questions.add(question)
        
        random.shuffle(unique_qa)
        
        print(f"\n🎉 QA 생성 완료: {len(unique_qa)}개")
        return unique_qa
    
    def save_qa_dataset(self, qa_data: List[Dict[str, Any]], filepath: str):
        """QA 데이터셋 저장 - 메타데이터 포함"""
        metadata = {
            'generation_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'total_qa_pairs': len(qa_data),
            'generation_version': '2.0_fixed'
        }
        
        data_with_metadata = {
            'metadata': metadata,
            'qa_pairs': qa_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ QA 데이터셋 저장 완료: {filepath}")

def main():
    """메인 실행 함수"""
    print("🔧 수정된 QA 생성을 시작합니다...")
    
    # 처리된 레시피 데이터 로드
    if not PROCESSED_RECIPES_PATH.exists():
        print(f"❌ 처리된 레시피 파일을 찾을 수 없습니다: {PROCESSED_RECIPES_PATH}")
        print("먼저 data_processor.py 또는 fixed_data_processor.py를 실행해주세요.")
        return
    
    generator = FixedQAGenerator()
    
    # 레시피 로드
    recipes = generator.load_recipes(PROCESSED_RECIPES_PATH)
    
    if not recipes:
        print("❌ 유효한 레시피 데이터를 찾을 수 없습니다.")
        return
    
    # QA 생성
    qa_dataset = generator.generate_all_qa(recipes)
    
    if qa_dataset:
        # QA 데이터셋 저장
        generator.save_qa_dataset(qa_dataset, QA_DATASET_PATH)
        
        # 유형별 통계 출력
        type_counts = {}
        for qa in qa_dataset:
            qa_type = qa.get('type', 'unknown')
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
        
        print("\n📊 유형별 QA 통계:")
        for qa_type, count in sorted(type_counts.items()):
            print(f"   {qa_type}: {count}개")
        
        # 샘플 출력
        print("\n📋 샘플 QA:")
        for i, qa in enumerate(qa_dataset[:3]):
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            qa_type = qa.get('type', '')
            
            print(f"\n{i+1}. 질문: {question}")
            print(f"   답변: {answer[:100]}...")
            print(f"   유형: {qa_type}")
    
    else:
        print("❌ QA 생성에 실패했습니다.")

if __name__ == "__main__":
    main()