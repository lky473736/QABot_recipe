"""
수정된 레시피 데이터 전처리기 - 다양한 데이터 구조 지원
"""
import json
import re
from typing import List, Dict, Any, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class FixedRecipeDataProcessor:
    def __init__(self):
        pass
        
    def load_raw_data(self, filepath: str) -> List[Dict[str, Any]]:
        """원본 데이터 로드 - 다양한 구조 지원"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 파일 로드 성공: {filepath}")
            print(f"📊 최상위 데이터 타입: {type(data)}")
            
            # 데이터 구조 분석 및 추출
            recipes = []
            
            if isinstance(data, dict):
                if 'metadata' in data and 'recipes' in data:
                    # 메타데이터가 있는 구조 (개선된 수집기 결과)
                    print("✅ 메타데이터 구조 감지")
                    recipes = data['recipes']
                    print(f"📈 메타데이터: {data['metadata']}")
                elif 'COOKRCP01' in data:
                    # API 응답 구조
                    print("✅ API 응답 구조 감지")
                    if 'row' in data['COOKRCP01']:
                        row_data = data['COOKRCP01']['row']
                        if isinstance(row_data, list):
                            recipes = row_data
                        else:
                            recipes = [row_data]
                else:
                    # 기타 딕셔너리 구조
                    print("ℹ️ 기타 딕셔너리 구조")
                    # 값들 중에서 리스트를 찾기
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict):
                                recipes = value
                                print(f"✅ '{key}' 키에서 레시피 배열 발견")
                                break
                    
                    if not recipes:
                        recipes = [data]  # 단일 레시피인 경우
                        
            elif isinstance(data, list):
                print("✅ 리스트 구조 감지")
                recipes = data
            else:
                print(f"❌ 지원하지 않는 데이터 타입: {type(data)}")
                return []
            
            print(f"🍳 추출된 레시피 개수: {len(recipes)}")
            
            # 레시피 데이터 유효성 확인
            valid_recipes = []
            for i, recipe in enumerate(recipes):
                if isinstance(recipe, dict):
                    valid_recipes.append(recipe)
                elif isinstance(recipe, str):
                    print(f"⚠️ 레시피 {i}는 문자열입니다: {recipe[:50]}...")
                    # 문자열인 경우 JSON 파싱 시도
                    try:
                        parsed_recipe = json.loads(recipe)
                        if isinstance(parsed_recipe, dict):
                            valid_recipes.append(parsed_recipe)
                    except:
                        print(f"❌ 레시피 {i} 파싱 실패")
                else:
                    print(f"❌ 레시피 {i}는 예상치 못한 타입: {type(recipe)}")
            
            print(f"✅ 유효한 레시피: {len(valid_recipes)}개")
            return valid_recipes
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            return []
    
    def clean_text(self, text: Union[str, None]) -> str:
        """텍스트 정리 - None 값 처리 개선"""
        if text is None or text == 'None' or not text:
            return ""
        
        # 문자열로 변환
        text = str(text)
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?()\-/\d]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_recipe_steps(self, recipe: Dict[str, Any]) -> List[str]:
        """조리 순서 추출 - 개선된 버전"""
        steps = []
        
        # 조리 순서 필드들 (MANUAL01 ~ MANUAL20)
        for i in range(1, 21):
            manual_key = f"MANUAL{i:02d}"
            if manual_key in recipe:
                step_text = self.clean_text(recipe[manual_key])
                if step_text and step_text not in ["", "-", "None", "없음"]:
                    # 번호 제거 (예: "1. " 제거)
                    step_text = re.sub(r'^[\d]+\.\s*', '', step_text)
                    steps.append(step_text)
        
        return steps
    
    def extract_recipe_images(self, recipe: Dict[str, Any]) -> List[str]:
        """레시피 이미지 URL 추출 - 개선된 버전"""
        images = []
        
        # 메인 이미지
        main_img = recipe.get('ATT_FILE_NO_MAIN')
        if main_img and main_img not in ["", "-", "None"]:
            images.append(main_img)
        
        # 조리 과정 이미지들 (MANUAL_IMG01 ~ MANUAL_IMG20)
        for i in range(1, 21):
            img_key = f"MANUAL_IMG{i:02d}"
            if img_key in recipe:
                img_url = recipe[img_key]
                if img_url and img_url not in ["", "-", "None"]:
                    images.append(img_url)
        
        # 중복 제거
        return list(set(images))
    
    def normalize_nutrition_value(self, value: Union[str, int, float, None]) -> str:
        """영양 정보 값 정규화"""
        if value is None or value == 'None' or not value:
            return ""
        
        # 문자열로 변환 후 숫자만 추출
        value_str = str(value).strip()
        if value_str in ["", "-", "0", "0.0"]:
            return ""
        
        # 숫자 추출
        numbers = re.findall(r'\d+\.?\d*', value_str)
        if numbers:
            return numbers[0]
        
        return ""
    
    def process_single_recipe(self, raw_recipe: Dict[str, Any]) -> Dict[str, Any]:
        """단일 레시피 처리 - 개선된 버전"""
        try:
            # 필수 필드 확인
            recipe_id = raw_recipe.get('RCP_SEQ', '')
            recipe_name = self.clean_text(raw_recipe.get('RCP_NM', ''))
            
            if not recipe_name:
                print(f"⚠️ 레시피 이름이 없음: ID {recipe_id}")
                return {}
            
            processed = {
                'id': str(recipe_id) if recipe_id else f"recipe_{hash(recipe_name)}",
                'name': recipe_name,
                'cooking_method': self.clean_text(raw_recipe.get('RCP_WAY2', '')),
                'category': self.clean_text(raw_recipe.get('RCP_PAT2', '')),
                'ingredients': self.clean_text(raw_recipe.get('RCP_PARTS_DTLS', '')),
                'calories': self.normalize_nutrition_value(raw_recipe.get('INFO_ENG')),
                'carbs': self.normalize_nutrition_value(raw_recipe.get('INFO_CAR')),
                'protein': self.normalize_nutrition_value(raw_recipe.get('INFO_PRO')),
                'fat': self.normalize_nutrition_value(raw_recipe.get('INFO_FAT')),
                'sodium': self.normalize_nutrition_value(raw_recipe.get('INFO_NA')),
                'steps': self.extract_recipe_steps(raw_recipe),
                'images': self.extract_recipe_images(raw_recipe),
                'tip': self.clean_text(raw_recipe.get('RCP_NA_TIP', '')),
                'hashtag': self.clean_text(raw_recipe.get('HASH_TAG', '')),
            }
            
            # 빈 값들 제거
            processed = {k: v for k, v in processed.items() if v not in ['', None, [], 'None']}
            
            # 최소 필수 필드 확인
            if not processed.get('name') or not processed.get('id'):
                return {}
            
            return processed
            
        except Exception as e:
            print(f"❌ 레시피 처리 중 오류: {e}")
            print(f"   원본 데이터 키: {list(raw_recipe.keys()) if isinstance(raw_recipe, dict) else 'dict가 아님'}")
            return {}
    
    def process_all_recipes(self, raw_recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 레시피 처리 - 개선된 버전"""
        processed_recipes = []
        
        print(f"📊 총 {len(raw_recipes)}개의 레시피를 처리합니다...")
        
        for i, recipe in enumerate(raw_recipes):
            try:
                if not isinstance(recipe, dict):
                    print(f"❌ 레시피 {i}는 딕셔너리가 아닙니다: {type(recipe)}")
                    continue
                
                processed = self.process_single_recipe(recipe)
                
                if processed:  # 빈 딕셔너리가 아닌 경우만 추가
                    processed_recipes.append(processed)
                
                if (i + 1) % 50 == 0:
                    print(f"✅ 처리 완료: {i + 1}/{len(raw_recipes)} (유효: {len(processed_recipes)}개)")
                    
            except Exception as e:
                print(f"❌ 레시피 처리 실패 (인덱스 {i}): {e}")
                continue
        
        print(f"🎉 처리 완료: {len(processed_recipes)}개의 유효한 레시피")
        
        # 처리 결과 통계
        if processed_recipes:
            categories = {}
            methods = {}
            
            for recipe in processed_recipes:
                cat = recipe.get('category', '기타')
                method = recipe.get('cooking_method', '기타')
                categories[cat] = categories.get(cat, 0) + 1
                methods[method] = methods.get(method, 0) + 1
            
            print(f"\n📈 카테고리 분포:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {cat}: {count}개")
                
            print(f"\n🍳 조리방법 분포:")
            for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {method}: {count}개")
        
        return processed_recipes
    
    def save_processed_recipes(self, recipes: List[Dict[str, Any]], filepath: str):
        """처리된 레시피 저장 - 메타데이터 포함"""
        # 메타데이터 생성
        metadata = {
            'processing_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'total_processed': len(recipes),
            'processing_version': '2.0_fixed'
        }
        
        # 데이터 구조
        data_with_metadata = {
            'metadata': metadata,
            'recipes': recipes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ 처리된 레시피 저장 완료: {filepath}")

def main():
    """메인 실행 함수"""
    print("🔧 수정된 데이터 전처리를 시작합니다...")
    
    # 디버깅 먼저 실행
    print("\n=== 1단계: 데이터 구조 분석 ===")
    if not RAW_RECIPES_PATH.exists():
        print(f"❌ 원본 레시피 파일을 찾을 수 없습니다: {RAW_RECIPES_PATH}")
        print("먼저 data_collector.py 또는 improved_data_collector.py를 실행해주세요.")
        return
    
    # 데이터 로드 및 처리
    print("\n=== 2단계: 데이터 로드 ===")
    processor = FixedRecipeDataProcessor()
    raw_recipes = processor.load_raw_data(RAW_RECIPES_PATH)
    
    if not raw_recipes:
        print("❌ 유효한 레시피 데이터를 찾을 수 없습니다.")
        return
    
    # 데이터 처리
    print("\n=== 3단계: 데이터 처리 ===")
    processed_recipes = processor.process_all_recipes(raw_recipes)
    
    if processed_recipes:
        # 처리된 데이터 저장
        print("\n=== 4단계: 결과 저장 ===")
        processor.save_processed_recipes(processed_recipes, PROCESSED_RECIPES_PATH)
        
        # 샘플 출력
        print("\n=== 5단계: 샘플 결과 ===")
        sample = processed_recipes[0]
        print(f"📋 샘플 레시피:")
        for key, value in sample.items():
            if isinstance(value, list):
                display_value = f"[{len(value)}개 항목]" if value else "[]"
            elif isinstance(value, str) and len(value) > 50:
                display_value = f"{value[:50]}..."
            else:
                display_value = value
            print(f"   {key}: {display_value}")
    else:
        print("❌ 처리된 레시피가 없습니다.")

if __name__ == "__main__":
    main()