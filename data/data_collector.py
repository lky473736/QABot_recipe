"""
개선된 식품안전처 조리식품 레시피 API 데이터 수집기
문제점: 기존 136개 -> 목표: 1000개+ 수집
"""
import requests
import json
import time
import xmltodict
from typing import List, Dict, Any
import sys
import os
from urllib.parse import quote
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class ImprovedRecipeDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = FOOD_SAFETY_BASE_URL
        self.service_id = RECIPE_SERVICE_ID
        self.session = requests.Session()
        
        # 요청 헤더 설정
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml, text/xml, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
    def build_api_url(self, start_idx: int = 1, end_idx: int = 1000, **kwargs) -> str:
        """API URL 생성 - 개선된 버전"""
        url = f"{self.base_url}/{self.api_key}/{self.service_id}/xml/{start_idx}/{end_idx}"
        
        # 추가 파라미터가 있다면 URL 인코딩 후 추가
        if kwargs:
            params = []
            for key, value in kwargs.items():
                if value:
                    encoded_value = quote(str(value), safe='')
                    params.append(f"{key}={encoded_value}")
            if params:
                url += "/" + "&".join(params)
                
        return url
    
    def test_api_connection(self) -> bool:
        """API 연결 테스트"""
        test_url = self.build_api_url(1, 5)
        try:
            print(f"API 연결 테스트: {test_url}")
            response = self.session.get(test_url, timeout=10)
            print(f"응답 상태코드: {response.status_code}")
            print(f"응답 헤더: {dict(response.headers)}")
            
            if response.status_code == 200:
                # XML 파싱 테스트
                data = xmltodict.parse(response.content)
                print(f"XML 파싱 성공")
                
                if 'COOKRCP01' in data:
                    if 'row' in data['COOKRCP01']:
                        rows = data['COOKRCP01']['row']
                        if isinstance(rows, list):
                            print(f"테스트 결과: {len(rows)}개 레시피 확인")
                        else:
                            print(f"테스트 결과: 1개 레시피 확인")
                        return True
                    else:
                        print("ERROR: 'row' 키가 없습니다.")
                        print(f"응답 구조: {list(data['COOKRCP01'].keys())}")
                else:
                    print("ERROR: 'COOKRCP01' 키가 없습니다.")
                    print(f"응답 구조: {list(data.keys())}")
            else:
                print(f"HTTP 오류: {response.status_code}")
                print(f"응답 내용: {response.text[:500]}")
                
        except Exception as e:
            print(f"API 연결 테스트 실패: {e}")
            return False
        
        return False
    
    def fetch_recipes_batch(self, start_idx: int = 1, end_idx: int = 1000) -> List[Dict[str, Any]]:
        """레시피 배치 가져오기 - 개선된 버전"""
        url = self.build_api_url(start_idx, end_idx)
        
        try:
            print(f"요청: {start_idx}-{end_idx} (총 {end_idx-start_idx+1}개 요청)")
            print(f"URL: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # 응답 내용 확인
            if not response.content:
                print("빈 응답 수신")
                return []
            
            # XML을 딕셔너리로 변환
            try:
                data = xmltodict.parse(response.content)
            except Exception as e:
                print(f"XML 파싱 실패: {e}")
                print(f"응답 내용 (처음 500자): {response.content[:500]}")
                return []
            
            # 레시피 데이터 추출
            recipes = []
            if 'COOKRCP01' in data:
                cookrcp_data = data['COOKRCP01']
                
                # 결과 확인
                if 'RESULT' in cookrcp_data:
                    result = cookrcp_data['RESULT']
                    result_code = result.get('CODE', '')
                    result_msg = result.get('MSG', '')
                    
                    if result_code != 'INFO-000':
                        print(f"API 오류: {result_code} - {result_msg}")
                        return []
                
                # 레시피 행 추출
                if 'row' in cookrcp_data:
                    rows = cookrcp_data['row']
                    if isinstance(rows, list):
                        recipes = rows
                    else:
                        recipes = [rows]  # 단일 항목인 경우
                        
                    print(f"✅ 성공: {len(recipes)}개 레시피 수집")
                else:
                    print("⚠️ 경고: 'row' 데이터가 없습니다.")
                    
            else:
                print("❌ 오류: 'COOKRCP01' 키가 없습니다.")
                print(f"응답 키들: {list(data.keys()) if isinstance(data, dict) else 'dict가 아님'}")
            
            return recipes
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 네트워크 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            return []
    
    def collect_all_recipes_improved(self, total_target: int = 1200) -> List[Dict[str, Any]]:
        """모든 레시피 데이터 수집 - 개선된 전략"""
        all_recipes = []
        
        # 1단계: API 연결 테스트
        print("=== 1단계: API 연결 테스트 ===")
        if not self.test_api_connection():
            print("❌ API 연결 테스트 실패. 종료합니다.")
            return []
        
        print("\n=== 2단계: 전체 데이터 수집 시작 ===")
        
        # 다양한 배치 크기로 시도
        batch_strategies = [
            (1, 1000),      # 전체 한번에
            (1, 500),       # 절반씩
            (501, 1000),    # 나머지 절반
            (1, 100),       # 100개씩
            (101, 200),
            (201, 300),
            (301, 400),
            (401, 500),
            (501, 600),
            (601, 700),
            (701, 800),
            (801, 900),
            (901, 1000),
            (1001, 1100),   # 혹시 1000개 이상이 있는지 확인
            (1101, 1200),
        ]
        
        collected_ids = set()  # 중복 제거용
        
        for start_idx, end_idx in batch_strategies:
            print(f"\n--- 배치 {start_idx}-{end_idx} 수집 시도 ---")
            
            recipes = self.fetch_recipes_batch(start_idx, end_idx)
            
            if recipes:
                # 중복 제거
                new_recipes = []
                for recipe in recipes:
                    recipe_id = recipe.get('RCP_SEQ', f"no_id_{len(all_recipes)}")
                    if recipe_id not in collected_ids:
                        collected_ids.add(recipe_id)
                        new_recipes.append(recipe)
                
                all_recipes.extend(new_recipes)
                print(f"✅ 새로운 레시피 {len(new_recipes)}개 추가 (누적: {len(all_recipes)}개)")
                
                # 목표 달성 시 중단
                if len(all_recipes) >= total_target:
                    print(f"🎯 목표 {total_target}개 달성!")
                    break
            else:
                print(f"❌ 배치 {start_idx}-{end_idx}에서 데이터 없음")
            
            # API 과부하 방지
            time.sleep(1)
        
        # 3단계: 특정 조건으로 추가 수집 시도
        if len(all_recipes) < 500:  # 여전히 적다면
            print("\n=== 3단계: 조건부 검색으로 추가 수집 ===")
            
            # 인기 재료들로 검색
            popular_ingredients = [
                "쇠고기", "돼지고기", "닭고기", "생선", "두부", "계란", 
                "감자", "양파", "마늘", "배추", "무", "당근"
            ]
            
            for ingredient in popular_ingredients:
                print(f"재료 '{ingredient}'로 검색...")
                recipes = self.fetch_recipes_with_ingredient(ingredient)
                
                # 중복 제거 후 추가
                new_recipes = []
                for recipe in recipes:
                    recipe_id = recipe.get('RCP_SEQ', f"ingredient_{ingredient}_{len(all_recipes)}")
                    if recipe_id not in collected_ids:
                        collected_ids.add(recipe_id)
                        new_recipes.append(recipe)
                
                if new_recipes:
                    all_recipes.extend(new_recipes)
                    print(f"✅ '{ingredient}' 검색으로 {len(new_recipes)}개 추가 (누적: {len(all_recipes)}개)")
                
                time.sleep(0.5)  # 짧은 대기
                
                if len(all_recipes) >= total_target:
                    break
        
        print(f"\n🎉 최종 수집 완료: {len(all_recipes)}개 레시피")
        return all_recipes
    
    def fetch_recipes_with_ingredient(self, ingredient: str) -> List[Dict[str, Any]]:
        """특정 재료로 레시피 검색"""
        try:
            recipes = self.fetch_recipes_batch(1, 1000, RCP_PARTS_DTLS=ingredient)
            return recipes
        except Exception as e:
            print(f"재료 '{ingredient}' 검색 실패: {e}")
            return []
    
    def validate_recipe_data(self, recipe: Dict[str, Any]) -> bool:
        """레시피 데이터 유효성 검사"""
        required_fields = ['RCP_SEQ', 'RCP_NM']
        
        for field in required_fields:
            if not recipe.get(field):
                return False
        
        # 빈 값이나 의미없는 값 체크
        recipe_name = recipe.get('RCP_NM', '').strip()
        if not recipe_name or recipe_name in ['-', 'None', '']:
            return False
            
        return True
    
    def save_recipes_with_metadata(self, recipes: List[Dict[str, Any]], filepath: str):
        """메타데이터와 함께 레시피 저장"""
        # 유효성 검사
        valid_recipes = [recipe for recipe in recipes if self.validate_recipe_data(recipe)]
        
        # 메타데이터 생성
        metadata = {
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_collected': len(recipes),
            'valid_recipes': len(valid_recipes),
            'api_source': f"{self.base_url}/{self.service_id}",
            'collection_strategy': 'improved_batch_collection'
        }
        
        # 데이터 구조
        data_with_metadata = {
            'metadata': metadata,
            'recipes': valid_recipes
        }
        
        # 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_with_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 수집 통계:")
        print(f"- 총 수집된 레시피: {len(recipes)}개")
        print(f"- 유효한 레시피: {len(valid_recipes)}개")
        print(f"- 저장 위치: {filepath}")
        
        # 샘플 레시피 정보 출력
        if valid_recipes:
            print(f"\n📋 샘플 레시피 정보:")
            sample = valid_recipes[0]
            print(f"- ID: {sample.get('RCP_SEQ', 'N/A')}")
            print(f"- 이름: {sample.get('RCP_NM', 'N/A')}")
            print(f"- 조리방법: {sample.get('RCP_WAY2', 'N/A')}")
            print(f"- 카테고리: {sample.get('RCP_PAT2', 'N/A')}")

def main():
    """메인 실행 함수"""
    if FOOD_SAFETY_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ config.py에서 FOOD_SAFETY_API_KEY를 설정해주세요!")
        return
    
    print("🚀 개선된 레시피 데이터 수집을 시작합니다...")
    print(f"📍 API 키: {FOOD_SAFETY_API_KEY[:10]}...")
    
    collector = ImprovedRecipeDataCollector(FOOD_SAFETY_API_KEY)
    
    # 개선된 수집 방법으로 데이터 수집
    recipes = collector.collect_all_recipes_improved(total_target=1000)
    
    if recipes:
        # 메타데이터와 함께 저장
        collector.save_recipes_with_metadata(recipes, RAW_RECIPES_PATH)
        print(f"\n✅ 성공: 총 {len(recipes)}개의 레시피를 수집했습니다!")
        
        # 간단한 통계 출력
        categories = {}
        cooking_methods = {}
        
        for recipe in recipes:
            category = recipe.get('RCP_PAT2', '기타')
            method = recipe.get('RCP_WAY2', '기타')
            
            categories[category] = categories.get(category, 0) + 1
            cooking_methods[method] = cooking_methods.get(method, 0) + 1
        
        print(f"\n📈 카테고리별 분포:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {cat}: {count}개")
            
        print(f"\n🍳 조리방법별 분포:")
        for method, count in sorted(cooking_methods.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {method}: {count}개")
            
    else:
        print("❌ 레시피 데이터 수집에 실패했습니다.")
        print("\n🔍 문제 해결 방법:")
        print("1. API 키가 올바른지 확인")
        print("2. 인터넷 연결 상태 확인")
        print("3. 식품안전처 API 서버 상태 확인")

if __name__ == "__main__":
    main()