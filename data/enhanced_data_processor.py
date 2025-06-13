"""
농림축산식품 공공데이터 레시피 전처리기
- 기본정보, 재료정보, 과정정보 통합
- 데이터 정제 및 정규화
- 챗봇 학습용 구조 생성
"""
import json
import re
from typing import List, Dict, Any, Optional
import sys
import os
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class MafraRecipeDataProcessor:
    def __init__(self):
        # 재료 정규화 매핑 (농림축산식품 데이터용)
        self.ingredient_mapping = {
            # 기본 재료
            '쇠고기': ['쇠고기', '소고기', '한우', '우육'],
            '돼지고기': ['돼지고기', '돼지', '삼겹살', '목살', '등심', '돈육'],
            '닭고기': ['닭고기', '닭', '치킨', '닭다리', '닭가슴살', '계육'],
            '생선': ['생선', '생선살', '흰살생선', '등푸른생선', '어류'],
            '달걀': ['달걀', '계란', '계란물', '난'],
            '두부': ['두부', '연두부', '순두부', '된두부'],
            
            # 채소류
            '양파': ['양파', '백양파', '적양파'],
            '마늘': ['마늘', '다진마늘', '마늘종'],
            '대파': ['대파', '파', '쪽파', '실파'],
            '생강': ['생강', '다진생강'],
            '배추': ['배추', '절인배추', '배추김치'],
            '무': ['무', '무말랭이', '단무지'],
            '당근': ['당근', '홍당무'],
            '감자': ['감자', '찐감자', '감자전분'],
            '고구마': ['고구마', '군고구마'],
            '호박': ['호박', '단호박', '애호박'],
            
            # 조미료
            '간장': ['간장', '진간장', '양조간장', '국간장'],
            '된장': ['된장', '쌈장', '고추장'],
            '고춧가루': ['고춧가루', '고추가루', '굵은고춧가루'],
            '참기름': ['참기름', '들기름'],
            '깨소금': ['깨소금', '깨', '참깨'],
            '소금': ['소금', '천일염', '굵은소금'],
            '설탕': ['설탕', '백설탕', '흑설탕'],
            '식용유': ['식용유', '기름', '올리브오일'],
        }
        
        # 요리 분류 정규화
        self.category_mapping = {
            '밑반찬': ['밑반찬', '반찬', '나물', '무침', '장아찌'],
            '메인반찬': ['메인반찬', '주반찬', '주요리'],
            '국/탕/찌개': ['국', '탕', '찌개', '찜', '전골', '스프'],
            '밥/죽/면': ['밥', '죽', '면', '국수', '파스타', '라면'],
            '후식/간식': ['후식', '디저트', '간식', '과자', '음료'],
            '일품요리': ['일품', '일품요리', '특별요리'],
            '기타': ['기타', '양념', '소스', '드레싱'],
        }
        
        # 난이도 매핑
        self.difficulty_mapping = {
            '쉬움': ['쉬움', '초급', '1단계', '간단'],
            '보통': ['보통', '중급', '2단계', '일반'],
            '어려움': ['어려움', '고급', '3단계', '복잡'],
        }
    
    def load_mafra_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """농림축산식품 데이터 로드"""
        data = {'basic': [], 'ingredients': [], 'processes': []}
        
        try:
            # 기본정보 로드
            if RECIPE_BASIC_PATH.exists():
                with open(RECIPE_BASIC_PATH, 'r', encoding='utf-8') as f:
                    basic_data = json.load(f)
                    data['basic'] = basic_data.get('basic_info', [])
                    print(f"✅ 기본정보 로드: {len(data['basic'])}개")
            
            # 재료정보 로드
            if RECIPE_INGREDIENT_PATH.exists():
                with open(RECIPE_INGREDIENT_PATH, 'r', encoding='utf-8') as f:
                    ingredient_data = json.load(f)
                    data['ingredients'] = ingredient_data.get('ingredient_info', [])
                    print(f"✅ 재료정보 로드: {len(data['ingredients'])}개")
            
            # 과정정보 로드
            if RECIPE_PROCESS_PATH.exists():
                with open(RECIPE_PROCESS_PATH, 'r', encoding='utf-8') as f:
                    process_data = json.load(f)
                    data['processes'] = process_data.get('process_info', [])
                    print(f"✅ 과정정보 로드: {len(data['processes'])}개")
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return {'basic': [], 'ingredients': [], 'processes': []}
    
    def group_data_by_recipe_id(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """레시피 ID별로 데이터 그룹화"""
        print("🔗 레시피 ID별 데이터 그룹화 중...")
        
        grouped_data = defaultdict(lambda: {
            'basic': None,
            'ingredients': [],
            'processes': []
        })
        
        # 기본정보 그룹화
        for item in data['basic']:
            recipe_id = self.extract_recipe_id(item)
            if recipe_id:
                grouped_data[recipe_id]['basic'] = item
        
        # 재료정보 그룹화
        for item in data['ingredients']:
            recipe_id = self.extract_recipe_id(item)
            if recipe_id:
                grouped_data[recipe_id]['ingredients'].append(item)
        
        # 과정정보 그룹화
        for item in data['processes']:
            recipe_id = self.extract_recipe_id(item)
            if recipe_id:
                grouped_data[recipe_id]['processes'].append(item)
        
        # 완전한 데이터만 필터링 (기본정보가 있는 것만)
        complete_data = {}
        for recipe_id, recipe_data in grouped_data.items():
            if recipe_data['basic'] is not None:
                complete_data[recipe_id] = recipe_data
        
        print(f"✅ 그룹화 완료: {len(complete_data)}개 레시피")
        return complete_data
    
    def extract_recipe_id(self, item: Dict[str, Any]) -> Optional[str]:
        """아이템에서 레시피 ID 추출"""
        # 가능한 레시피 ID 필드명들
        id_fields = ['RECIPE_ID', '레시피일련번호', 'RECIPE_CODE', 'RCP_SEQ']
        
        for field in id_fields:
            if field in item and item[field]:
                return str(item[field])
        
        return None
    
    def normalize_ingredient_name(self, ingredient_name: str) -> str:
        """재료명 정규화"""
        if not ingredient_name:
            return ""
        
        # 기본 정리
        ingredient = re.sub(r'[^\w\s가-힣]', ' ', ingredient_name)
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        
        # 양 표시 제거
        ingredient = re.sub(r'\d+[gmlkg개큰술작은술컵마리개입]?\s*', '', ingredient)
        ingredient = re.sub(r'적당량|조금|많이|약간|소량|대량', '', ingredient)
        
        # 매핑을 통한 정규화
        for standard, variants in self.ingredient_mapping.items():
            for variant in variants:
                if variant in ingredient:
                    return standard
        
        return ingredient.strip()
    
    def extract_main_ingredients(self, ingredients_list: List[Dict[str, Any]]) -> List[str]:
        """주요 재료 추출"""
        main_ingredients = []
        
        for ingredient_item in ingredients_list:
            # 재료명 필드 찾기
            ingredient_name = (
                ingredient_item.get('IRDNT_NM') or 
                ingredient_item.get('재료명') or 
                ingredient_item.get('INGREDIENT_NAME') or
                str(ingredient_item.get('IRDNT_NM', ''))
            )
            
            if ingredient_name:
                normalized = self.normalize_ingredient_name(ingredient_name)
                if normalized and len(normalized) >= 2:
                    main_ingredients.append(normalized)
        
        # 중복 제거 및 상위 8개만 반환
        return list(dict.fromkeys(main_ingredients))[:8]
    
    def extract_cooking_steps(self, processes_list: List[Dict[str, Any]]) -> List[str]:
        """조리 과정 추출"""
        steps = []
        
        # 과정 순서별로 정렬
        sorted_processes = sorted(processes_list, key=lambda x: int(x.get('COOKING_NO', x.get('조리순서', 0)) or 0))
        
        for process_item in sorted_processes:
            # 조리 과정 텍스트 필드 찾기
            step_text = (
                process_item.get('COOKING_DC') or
                process_item.get('조리과정') or
                process_item.get('PROCESS_DESCRIPTION') or
                str(process_item.get('COOKING_DC', ''))
            )
            
            if step_text and step_text.strip():
                # 텍스트 정리
                step_text = re.sub(r'\s+', ' ', step_text).strip()
                
                # 의미있는 내용만 추가
                if len(step_text) >= 10:  # 너무 짧은 단계 제외
                    steps.append(step_text)
        
        return steps[:10]  # 최대 10단계
    
    def normalize_recipe_info(self, basic_info: Dict[str, Any]) -> Dict[str, str]:
        """레시피 기본정보 정규화"""
        # 레시피명
        recipe_name = (
            basic_info.get('RECIPE_NM_KO') or
            basic_info.get('요리명') or
            basic_info.get('RECIPE_NAME') or
            str(basic_info.get('RECIPE_NM_KO', ''))
        ).strip()
        
        # 요리 분류
        recipe_type = (
            basic_info.get('RECIPE_TY_NM') or
            basic_info.get('요리분류') or
            basic_info.get('RECIPE_TYPE') or
            str(basic_info.get('RECIPE_TY_NM', ''))
        ).strip()
        
        # 조리 방법
        cooking_method = (
            basic_info.get('COOKING_MTH_NM') or
            basic_info.get('조리방법') or
            basic_info.get('COOKING_METHOD') or
            str(basic_info.get('COOKING_MTH_NM', ''))
        ).strip()
        
        # 난이도
        difficulty = (
            basic_info.get('RECIPE_LV_NM') or
            basic_info.get('난이도') or
            basic_info.get('DIFFICULTY') or
            str(basic_info.get('RECIPE_LV_NM', ''))
        ).strip()
        
        # 조리 시간
        cooking_time = (
            basic_info.get('COOKING_TIME') or
            basic_info.get('조리시간') or
            str(basic_info.get('COOKING_TIME', ''))
        ).strip()
        
        # 분류 정규화
        normalized_category = self.normalize_category(recipe_type)
        normalized_difficulty = self.normalize_difficulty(difficulty)
        
        return {
            'name': recipe_name,
            'category': normalized_category,
            'cooking_method': cooking_method,
            'difficulty': normalized_difficulty,
            'cooking_time': cooking_time,
            'original_type': recipe_type
        }
    
    def normalize_category(self, category: str) -> str:
        """카테고리 정규화"""
        if not category:
            return "기타"
        
        category_lower = category.lower()
        for standard, variants in self.category_mapping.items():
            for variant in variants:
                if variant in category_lower:
                    return standard
        
        return category if category else "기타"
    
    def normalize_difficulty(self, difficulty: str) -> str:
        """난이도 정규화"""
        if not difficulty:
            return "보통"
        
        difficulty_lower = difficulty.lower()
        for standard, variants in self.difficulty_mapping.items():
            for variant in variants:
                if variant in difficulty_lower:
                    return standard
        
        return difficulty if difficulty else "보통"
    
    def generate_recipe_summary(self, recipe_info: Dict[str, Any]) -> str:
        """레시피 요약 생성"""
        name = recipe_info.get('name', '')
        category = recipe_info.get('category', '')
        main_ingredients = recipe_info.get('main_ingredients', [])
        cooking_method = recipe_info.get('cooking_method', '')
        difficulty = recipe_info.get('difficulty', '')
        
        summary_parts = []
        
        if name:
            summary_parts.append(f"{name}는")
        
        if category:
            summary_parts.append(f"{category} 종류의")
        
        if main_ingredients:
            ingredients_str = ', '.join(main_ingredients[:3])
            summary_parts.append(f"{ingredients_str}를 주재료로 하는")
        
        if cooking_method:
            summary_parts.append(f"{cooking_method}")
        
        if difficulty:
            summary_parts.append(f"({difficulty})")
        
        summary_parts.append("요리입니다.")
        
        return ' '.join(summary_parts)
    
    def process_single_recipe(self, recipe_id: str, recipe_data: Dict[str, Any]) -> Dict[str, Any]:
        """단일 레시피 처리"""
        try:
            basic_info = recipe_data['basic']
            ingredients_list = recipe_data['ingredients']
            processes_list = recipe_data['processes']
            
            # 기본정보 정규화
            recipe_info = self.normalize_recipe_info(basic_info)
            
            # 레시피명 유효성 검사
            if not recipe_info['name'] or len(recipe_info['name']) < 2:
                return {}
            
            # 주요 재료 추출
            main_ingredients = self.extract_main_ingredients(ingredients_list)
            
            # 조리 과정 추출
            cooking_steps = self.extract_cooking_steps(processes_list)
            
            # 재료 텍스트 생성
            ingredients_text = ', '.join([item.get('IRDNT_NM', '') for item in ingredients_list if item.get('IRDNT_NM')])
            
            # 처리된 레시피 구조
            processed = {
                'id': recipe_id,
                'name': recipe_info['name'],
                'category': recipe_info['category'],
                'cooking_method': recipe_info['cooking_method'],
                'difficulty': recipe_info['difficulty'],
                'cooking_time': recipe_info['cooking_time'],
                'ingredients': ingredients_text,
                'main_ingredients': main_ingredients,
                'steps': cooking_steps,
                'ingredient_count': len(ingredients_list),
                'process_count': len(processes_list),
            }
            
            # 레시피 요약 생성
            processed['summary'] = self.generate_recipe_summary(processed)
            
            # 빈 값들 정리
            cleaned_recipe = {}
            for key, value in processed.items():
                if value not in ['', None, [], {}, 'None']:
                    cleaned_recipe[key] = value
            
            # 최소 필수 조건 확인
            required_fields = ['id', 'name']
            if all(field in cleaned_recipe for field in required_fields):
                return cleaned_recipe
            
            return {}
            
        except Exception as e:
            print(f"❌ 레시피 {recipe_id} 처리 중 오류: {e}")
            return {}
    
    def process_all_recipes(self, grouped_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 레시피 처리"""
        processed_recipes = []
        
        print(f"🚀 {len(grouped_data)}개 레시피 처리 시작...")
        
        # 진행률 표시를 위한 체크포인트
        total_count = len(grouped_data)
        checkpoints = [int(total_count * i / 10) for i in range(1, 11)]
        
        for i, (recipe_id, recipe_data) in enumerate(grouped_data.items()):
            try:
                processed = self.process_single_recipe(recipe_id, recipe_data)
                
                if processed:
                    processed_recipes.append(processed)
                
                # 진행률 표시
                if (i + 1) in checkpoints:
                    progress = ((i + 1) / total_count) * 100
                    print(f"📈 진행률: {progress:.0f}% ({i + 1}/{total_count}) - 유효: {len(processed_recipes)}개")
                    
            except Exception as e:
                print(f"❌ 레시피 {recipe_id} 처리 실패: {e}")
                continue
        
        print(f"✅ 처리 완료: {len(processed_recipes)}개 유효 레시피")
        return processed_recipes
    
    def print_processing_statistics(self, recipes: List[Dict[str, Any]]):
        """처리 결과 통계 출력"""
        if not recipes:
            return
        
        categories = defaultdict(int)
        methods = defaultdict(int)
        difficulties = defaultdict(int)
        ingredient_counts = defaultdict(int)
        
        for recipe in recipes:
            categories[recipe.get('category', '기타')] += 1
            methods[recipe.get('cooking_method', '기타')] += 1
            difficulties[recipe.get('difficulty', '보통')] += 1
            
            for ingredient in recipe.get('main_ingredients', []):
                ingredient_counts[ingredient] += 1
        
        print(f"\n📊 처리 결과 통계:")
        print(f"   총 레시피: {len(recipes)}개")
        
        print(f"\n🗂️ 카테고리 분포:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count}개")
        
        print(f"\n🍳 조리방법 분포 (상위 5개):")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {method}: {count}개")
            
        print(f"\n⭐ 난이도 분포:")
        for difficulty, count in sorted(difficulties.items(), key=lambda x: x[1], reverse=True):
            print(f"   {difficulty}: {count}개")
        
        print(f"\n🥕 인기 재료 (상위 10개):")
        for ingredient, count in sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {ingredient}: {count}개 레시피")
    
    def save_processed_recipes(self, recipes: List[Dict[str, Any]], filepath: str):
        """처리된 레시피 저장"""
        # 상세 메타데이터 생성
        categories = defaultdict(int)
        methods = defaultdict(int)
        difficulties = defaultdict(int)
        ingredients = defaultdict(int)
        
        for recipe in recipes:
            categories[recipe.get('category', '기타')] += 1
            methods[recipe.get('cooking_method', '기타')] += 1
            difficulties[recipe.get('difficulty', '보통')] += 1
            for ingredient in recipe.get('main_ingredients', []):
                ingredients[ingredient] += 1
        
        metadata = {
            'processing_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'total_recipes': len(recipes),
            'processing_version': '4.0_mafra',
            'data_source': '농림축산식품 공공데이터포털',
            'features': [
                'mafra_api_integration',
                'multi_table_joining',
                'ingredient_normalization',
                'category_standardization',
                'cooking_step_extraction',
                'recipe_summarization'
            ]
        }
        
        statistics = {
            'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            'cooking_methods': dict(sorted(methods.items(), key=lambda x: x[1], reverse=True)),
            'difficulties': dict(sorted(difficulties.items(), key=lambda x: x[1], reverse=True)),
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
    print("🚀 농림축산식품 레시피 데이터 전처리 시작...")
    
    # 필요한 파일들 확인
    required_files = [RECIPE_BASIC_PATH, RECIPE_INGREDIENT_PATH, RECIPE_PROCESS_PATH]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"❌ 필요한 데이터 파일이 없습니다:")
        for f in missing_files:
            print(f"   {f}")
        print("먼저 enhanced_data_collector.py를 실행해주세요.")
        return
    
    processor = MafraRecipeDataProcessor()
    
    # 데이터 로드
    raw_data = processor.load_mafra_data()
    
    if not any(raw_data.values()):
        print("❌ 유효한 데이터를 찾을 수 없습니다.")
        return
    
    # 레시피 ID별 그룹화
    grouped_data = processor.group_data_by_recipe_id(raw_data)
    
    if not grouped_data:
        print("❌ 그룹화된 데이터가 없습니다.")
        return
    
    # 데이터 처리
    processed_recipes = processor.process_all_recipes(grouped_data)
    
    if processed_recipes:
        # 통계 출력
        processor.print_processing_statistics(processed_recipes)
        
        # 처리된 데이터 저장
        processor.save_processed_recipes(processed_recipes, PROCESSED_RECIPES_PATH)
        
        # 샘플 출력
        print(f"\n📋 샘플 레시피:")
        for i, recipe in enumerate(processed_recipes[:3]):
            print(f"\n{i+1}. {recipe.get('name', 'N/A')}")
            print(f"   ID: {recipe.get('id', 'N/A')}")
            print(f"   카테고리: {recipe.get('category', 'N/A')}")
            print(f"   난이도: {recipe.get('difficulty', 'N/A')}")
            print(f"   조리방법: {recipe.get('cooking_method', 'N/A')}")
            print(f"   주재료: {', '.join(recipe.get('main_ingredients', []))}")
            print(f"   요약: {recipe.get('summary', 'N/A')}")
            
    else:
        print("❌ 처리된 레시피가 없습니다.")

if __name__ == "__main__":
    main()