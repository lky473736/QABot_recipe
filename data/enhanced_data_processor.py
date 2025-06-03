"""
개선된 레시피 데이터 전처리기
- 대용량 데이터 처리 최적화
- 고품질 데이터 정제
- 챗봇 학습에 최적화된 구조 생성
"""
import json
import re
from typing import List, Dict, Any, Union, Set
import sys
import os
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class EnhancedRecipeDataProcessor:
    def __init__(self):
        # 재료 정규화 매핑
        self.ingredient_mapping = {
            # 고기류
            '소고기': ['쇠고기', '소', '한우'],
            '돼지고기': ['돼지', '삼겹살', '목살', '등심'],
            '닭고기': ['닭', '치킨', '닭다리', '닭가슴살'],
            '생선': ['생선살', '흰살생선', '등푸른생선'],
            
            # 채소류
            '양파': ['양파'],
            '마늘': ['마늘', '다진마늘'],
            '대파': ['파', '대파', '쪽파'],
            '생강': ['생강', '다진생강'],
            '배추': ['배추', '절인배추'],
            '무': ['무', '무말랭이'],
            '당근': ['당근'],
            '감자': ['감자', '찐감자'],
            '고구마': ['고구마'],
            
            # 기본 재료
            '계란': ['계란', '달걀', '계란물'],
            '두부': ['두부', '연두부', '순두부', '된두부'],
            '버섯': ['버섯', '표고버섯', '느타리버섯', '팽이버섯'],
            '콩나물': ['콩나물'],
            '김치': ['김치', '배추김치', '묵은지'],
            
            # 조미료
            '간장': ['간장', '진간장', '양조간장'],
            '고춧가루': ['고춧가루', '고추가루'],
            '참기름': ['참기름'],
            '깨소금': ['깨소금', '깨'],
            '소금': ['소금', '천일염'],
            '설탕': ['설탕', '백설탕'],
            '식용유': ['식용유', '기름'],
        }
        
        # 조리방법 정규화
        self.cooking_method_mapping = {
            '볶음': ['볶기', '볶음', '볶은'],
            '찜': ['찜', '찌기', '찐'],
            '구이': ['구이', '굽기', '구운'],
            '조림': ['조림', '조리기', '조린'],
            '튀김': ['튀김', '튀기기', '튀긴'],
            '끓임': ['끓임', '끓이기', '끓인'],
            '무침': ['무침', '무치기', '무친'],
        }
        
        # 카테고리 정규화
        self.category_mapping = {
            '밑반찬': ['밑반찬', '반찬', '나물'],
            '메인반찬': ['메인반찬', '주반찬'],
            '국/탕': ['국', '탕', '찌개', '전골'],
            '밥/죽/면': ['밥', '죽', '면', '국수'],
            '후식': ['후식', '디저트', '간식'],
            '일품요리': ['일품', '일품요리'],
            '양념/소스': ['양념', '소스', '드레싱'],
        }
        
    def load_enhanced_data(self, filepath: str) -> List[Dict[str, Any]]:
        """개선된 데이터 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 파일 로드 성공: {filepath}")
            
            recipes = []
            if isinstance(data, dict):
                if 'recipes' in data:
                    recipes = data['recipes']
                    if 'metadata' in data:
                        print(f"📈 메타데이터: {data['metadata']}")
                    if 'statistics' in data:
                        print(f"📊 통계 정보: {len(data['statistics'])}개 항목")
                else:
                    # 기존 구조 지원
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict):
                                recipes = value
                                break
            elif isinstance(data, list):
                recipes = data
            
            print(f"🍳 로드된 레시피: {len(recipes)}개")
            return recipes
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []
    
    def normalize_ingredient(self, ingredient_text: str) -> str:
        """재료명 정규화"""
        if not ingredient_text:
            return ""
        
        # 기본 정리
        ingredient = re.sub(r'[^\w\s가-힣]', ' ', ingredient_text)
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        
        # 양 표시 제거
        ingredient = re.sub(r'\d+[gmlkg개큰술작은술컵]?\s*', '', ingredient)
        ingredient = re.sub(r'적당량|조금|많이|약간', '', ingredient)
        
        # 매핑을 통한 정규화
        for standard, variants in self.ingredient_mapping.items():
            for variant in variants:
                if variant in ingredient:
                    return standard
        
        return ingredient.strip()
    
    def extract_main_ingredients(self, ingredients_text: str) -> List[str]:
        """주요 재료 추출 (개선된 버전)"""
        if not ingredients_text:
            return []
        
        # 쉼표나 줄바꿈으로 분리
        ingredients_list = re.split(r'[,\n]', ingredients_text)
        
        main_ingredients = []
        for ingredient in ingredients_list:
            normalized = self.normalize_ingredient(ingredient)
            if normalized and len(normalized) >= 2:
                main_ingredients.append(normalized)
        
        # 중복 제거 및 상위 5개만 반환
        return list(dict.fromkeys(main_ingredients))[:5]
    
    def normalize_cooking_method(self, method: str) -> str:
        """조리방법 정규화"""
        if not method:
            return "기타"
        
        for standard, variants in self.cooking_method_mapping.items():
            for variant in variants:
                if variant in method:
                    return standard
        
        return method
    
    def normalize_category(self, category: str) -> str:
        """카테고리 정규화"""
        if not category:
            return "기타"
        
        for standard, variants in self.category_mapping.items():
            for variant in variants:
                if variant in category:
                    return standard
        
        return category
    
    def clean_cooking_steps(self, recipe: Dict[str, Any]) -> List[str]:
        """조리 순서 정리 (개선된 버전)"""
        steps = []
        
        # MANUAL01 ~ MANUAL20 추출
        for i in range(1, 21):
            manual_key = f"MANUAL{i:02d}"
            if manual_key in recipe:
                step_text = str(recipe[manual_key]).strip()
                
                # 의미있는 내용만 추가
                if step_text and step_text not in ["", "-", "None", "없음", "null"]:
                    # 번호 제거 및 정리
                    step_text = re.sub(r'^[\d]+\.\s*', '', step_text)
                    step_text = re.sub(r'\s+', ' ', step_text).strip()
                    
                    if len(step_text) >= 5:  # 너무 짧은 단계 제외
                        steps.append(step_text)
        
        return steps[:10]  # 최대 10단계까지
    
    def extract_nutrition_info(self, recipe: Dict[str, Any]) -> Dict[str, str]:
        """영양 정보 추출 및 정리"""
        nutrition = {}
        
        nutrition_fields = {
            'calories': 'INFO_ENG',
            'carbs': 'INFO_CAR', 
            'protein': 'INFO_PRO',
            'fat': 'INFO_FAT',
            'sodium': 'INFO_NA'
        }
        
        for key, field in nutrition_fields.items():
            value = recipe.get(field, '')
            if value and str(value).strip() not in ['', '-', '0', '0.0', 'None']:
                # 숫자만 추출
                numbers = re.findall(r'\d+\.?\d*', str(value))
                if numbers:
                    nutrition[key] = numbers[0]
        
        return nutrition
    
    def generate_recipe_summary(self, recipe: Dict[str, Any]) -> str:
        """레시피 요약 생성 (챗봇 학습용)"""
        name = recipe.get('name', '')
        category = recipe.get('category', '')
        cooking_method = recipe.get('cooking_method', '')
        main_ingredients = recipe.get('main_ingredients', [])
        
        summary_parts = []
        
        if name:
            summary_parts.append(f"{name}는")
        
        if category:
            summary_parts.append(f"{category} 종류의")
        
        if main_ingredients:
            ingredients_str = ', '.join(main_ingredients[:3])
            summary_parts.append(f"{ingredients_str}를 주재료로 하는")
        
        if cooking_method:
            summary_parts.append(f"{cooking_method} 요리입니다.")
        else:
            summary_parts.append("요리입니다.")
        
        return ' '.join(summary_parts)
    
    def process_single_recipe(self, raw_recipe: Dict[str, Any]) -> Dict[str, Any]:
        """단일 레시피 처리 (개선된 버전)"""
        try:
            # 필수 필드 확인
            recipe_id = raw_recipe.get('RCP_SEQ', '')
            recipe_name = str(raw_recipe.get('RCP_NM', '')).strip()
            
            if not recipe_name or len(recipe_name) < 2:
                return {}
            
            # 기본 정보 처리
            ingredients_text = str(raw_recipe.get('RCP_PARTS_DTLS', ''))
            main_ingredients = self.extract_main_ingredients(ingredients_text)
            
            # 조리법 및 카테고리 정규화
            cooking_method = self.normalize_cooking_method(raw_recipe.get('RCP_WAY2', ''))
            category = self.normalize_category(raw_recipe.get('RCP_PAT2', ''))
            
            # 조리 순서 정리
            cooking_steps = self.clean_cooking_steps(raw_recipe)
            
            # 영양 정보 추출
            nutrition = self.extract_nutrition_info(raw_recipe)
            
            # 처리된 레시피 구조
            processed = {
                'id': str(recipe_id) if recipe_id else f"recipe_{hash(recipe_name)}",
                'name': recipe_name,
                'category': category,
                'cooking_method': cooking_method,
                'ingredients': ingredients_text,
                'main_ingredients': main_ingredients,
                'steps': cooking_steps,
                'nutrition': nutrition,
                'tip': str(raw_recipe.get('RCP_NA_TIP', '')).strip(),
                'hashtag': str(raw_recipe.get('HASH_TAG', '')).strip(),
                'main_image': raw_recipe.get('ATT_FILE_NO_MAIN', ''),
            }
            
            # 레시피 요약 생성
            processed['summary'] = self.generate_recipe_summary(processed)
            
            # 빈 값들 정리
            cleaned_recipe = {}
            for key, value in processed.items():
                if value not in ['', None, [], {}, 'None']:
                    cleaned_recipe[key] = value
            
            # 최소 필수 조건 확인
            required_fields = ['id', 'name', 'main_ingredients']
            if all(field in cleaned_recipe for field in required_fields):
                return cleaned_recipe
            
            return {}
            
        except Exception as e:
            print(f"❌ 레시피 처리 중 오류: {e}")
            return {}
    
    def process_all_recipes(self, raw_recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 레시피 처리 (개선된 버전)"""
        processed_recipes = []
        
        print(f"🚀 {len(raw_recipes)}개 레시피 처리 시작...")
        
        # 진행률 표시를 위한 체크포인트
        checkpoints = [int(len(raw_recipes) * i / 10) for i in range(1, 11)]
        
        for i, recipe in enumerate(raw_recipes):
            try:
                processed = self.process_single_recipe(recipe)
                
                if processed:
                    processed_recipes.append(processed)
                
                # 진행률 표시
                if i + 1 in checkpoints:
                    progress = ((i + 1) / len(raw_recipes)) * 100
                    print(f"📈 진행률: {progress:.0f}% ({i + 1}/{len(raw_recipes)}) - 유효: {len(processed_recipes)}개")
                    
            except Exception as e:
                print(f"❌ 레시피 {i} 처리 실패: {e}")
                continue
        
        print(f"✅ 처리 완료: {len(processed_recipes)}개 유효 레시피")
        
        # 처리 결과 통계
        self.print_processing_statistics(processed_recipes)
        
        return processed_recipes
    
    def print_processing_statistics(self, recipes: List[Dict[str, Any]]):
        """처리 결과 통계 출력"""
        if not recipes:
            return
        
        categories = defaultdict(int)
        methods = defaultdict(int)
        ingredient_counts = defaultdict(int)
        
        for recipe in recipes:
            categories[recipe.get('category', '기타')] += 1
            methods[recipe.get('cooking_method', '기타')] += 1
            
            for ingredient in recipe.get('main_ingredients', []):
                ingredient_counts[ingredient] += 1
        
        print(f"\n📊 처리 결과 통계:")
        print(f"   총 레시피: {len(recipes)}개")
        
        print(f"\n🗂️ 카테고리 분포 (상위 5개):")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {cat}: {count}개")
        
        print(f"\n🍳 조리방법 분포 (상위 5개):")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {method}: {count}개")
        
        print(f"\n🥕 인기 재료 (상위 10개):")
        for ingredient, count in sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {ingredient}: {count}개 레시피")
    
    def save_processed_recipes(self, recipes: List[Dict[str, Any]], filepath: str):
        """처리된 레시피 저장 (개선된 메타데이터 포함)"""
        # 상세 메타데이터 생성
        categories = defaultdict(int)
        methods = defaultdict(int)
        ingredients = defaultdict(int)
        
        for recipe in recipes:
            categories[recipe.get('category', '기타')] += 1
            methods[recipe.get('cooking_method', '기타')] += 1
            for ingredient in recipe.get('main_ingredients', []):
                ingredients[ingredient] += 1
        
        metadata = {
            'processing_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'total_recipes': len(recipes),
            'processing_version': '3.0_enhanced',
            'features': [
                'ingredient_normalization',
                'category_standardization', 
                'cooking_method_classification',
                'nutrition_extraction',
                'recipe_summarization'
            ]
        }
        
        statistics = {
            'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            'cooking_methods': dict(sorted(methods.items(), key=lambda x: x[1], reverse=True)),
            'top_ingredients': dict(sorted(ingredients.items(), key=lambda x: x[1], reverse=True)[:20])
        }
        
        # 최종 데이터 구조
        enhanced_data = {
            'metadata': metadata,
            'statistics': statistics,
            'recipes': recipes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 저장 완료: {filepath}")
        print(f"   메타데이터: {len(metadata)}개 항목")
        print(f"   통계 정보: {len(statistics)}개 카테고리")

def main():
    """메인 실행 함수"""
    print("🚀 개선된 레시피 데이터 전처리 시작...")
    
    if not RAW_RECIPES_PATH.exists():
        print(f"❌ 원본 레시피 파일을 찾을 수 없습니다: {RAW_RECIPES_PATH}")
        print("먼저 enhanced_data_collector.py를 실행해주세요.")
        return
    
    # 데이터 로드
    processor = EnhancedRecipeDataProcessor()
    raw_recipes = processor.load_enhanced_data(RAW_RECIPES_PATH)
    
    if not raw_recipes:
        print("❌ 유효한 레시피 데이터를 찾을 수 없습니다.")
        return
    
    # 데이터 처리
    processed_recipes = processor.process_all_recipes(raw_recipes)
    
    if processed_recipes:
        # 처리된 데이터 저장
        processor.save_processed_recipes(processed_recipes, PROCESSED_RECIPES_PATH)
        
        # 샘플 출력
        print(f"\n📋 샘플 레시피:")
        for i, recipe in enumerate(processed_recipes[:3]):
            print(f"\n{i+1}. {recipe.get('name', 'N/A')}")
            print(f"   카테고리: {recipe.get('category', 'N/A')}")
            print(f"   조리방법: {recipe.get('cooking_method', 'N/A')}")
            print(f"   주재료: {', '.join(recipe.get('main_ingredients', []))}")
            print(f"   요약: {recipe.get('summary', 'N/A')}")
    else:
        print("❌ 처리된 레시피가 없습니다.")

if __name__ == "__main__":
    main()
