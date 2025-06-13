"""
농림축산식품 공공데이터포털 레시피 데이터 수집기
- 레시피 기본정보, 재료정보, 과정정보 수집
- 데이터 통합 및 검증
"""
import requests
import json
import time
from typing import List, Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class MafraRecipeDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = MAFRA_BASE_URL
        self.session = requests.Session()
        
        # 요청 헤더 설정
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/json, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # 수집된 데이터 저장
        self.recipe_basic = []
        self.recipe_ingredients = []
        self.recipe_processes = []
        
    def build_api_url(self, service_id: str, start_idx: int = 1, end_idx: int = 1000) -> str:
        """API URL 생성"""
        return f"{self.base_url}/{self.api_key}/json/{service_id}/{start_idx}/{end_idx}"
    
    def test_api_connection(self) -> bool:
        """API 연결 테스트"""
        test_url = self.build_api_url(RECIPE_BASIC_SERVICE_ID, 1, 5)
        try:
            print(f"API 연결 테스트: {test_url}")
            response = self.session.get(test_url, timeout=10)
            print(f"응답 상태코드: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"JSON 파싱 성공")
                
                # 응답 구조 확인
                if RECIPE_BASIC_SERVICE_ID in data:
                    if 'row' in data[RECIPE_BASIC_SERVICE_ID]:
                        rows = data[RECIPE_BASIC_SERVICE_ID]['row']
                        if isinstance(rows, list):
                            print(f"테스트 결과: {len(rows)}개 레시피 확인")
                        else:
                            print(f"테스트 결과: 1개 레시피 확인")
                        return True
                    else:
                        print("ERROR: 'row' 키가 없습니다.")
                        print(f"응답 구조: {list(data[RECIPE_BASIC_SERVICE_ID].keys())}")
                else:
                    print(f"ERROR: '{RECIPE_BASIC_SERVICE_ID}' 키가 없습니다.")
                    print(f"응답 구조: {list(data.keys())}")
            else:
                print(f"HTTP 오류: {response.status_code}")
                print(f"응답 내용: {response.text[:500]}")
                
        except Exception as e:
            print(f"API 연결 테스트 실패: {e}")
            return False
        
        return False
    
    def fetch_recipe_basic_data(self) -> List[Dict[str, Any]]:
        """레시피 기본정보 수집 (537개)"""
        print("🍳 레시피 기본정보 수집 중...")
        
        url = self.build_api_url(RECIPE_BASIC_SERVICE_ID, 1, 1000)
        
        try:
            print(f"요청 URL: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if RECIPE_BASIC_SERVICE_ID in data and 'row' in data[RECIPE_BASIC_SERVICE_ID]:
                rows = data[RECIPE_BASIC_SERVICE_ID]['row']
                recipes = rows if isinstance(rows, list) else [rows]
                
                print(f"✅ 레시피 기본정보 수집 완료: {len(recipes)}개")
                self.recipe_basic = recipes
                return recipes
            else:
                print("❌ 레시피 기본정보 데이터 없음")
                return []
                
        except Exception as e:
            print(f"❌ 레시피 기본정보 수집 실패: {e}")
            return []
    
    def fetch_recipe_ingredient_data(self) -> List[Dict[str, Any]]:
        """레시피 재료정보 수집 (6104개)"""
        print("🥕 레시피 재료정보 수집 중...")
        
        all_ingredients = []
        
        # 1000개씩 분할 수집
        for start_idx in range(1, 7001, 1000):
            end_idx = min(start_idx + 999, 7000)
            url = self.build_api_url(RECIPE_INGREDIENT_SERVICE_ID, start_idx, end_idx)
            
            try:
                print(f"  재료정보 배치 {start_idx}-{end_idx} 수집 중...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if RECIPE_INGREDIENT_SERVICE_ID in data and 'row' in data[RECIPE_INGREDIENT_SERVICE_ID]:
                    rows = data[RECIPE_INGREDIENT_SERVICE_ID]['row']
                    ingredients = rows if isinstance(rows, list) else [rows]
                    all_ingredients.extend(ingredients)
                    print(f"  ✅ 배치 {start_idx}-{end_idx}: {len(ingredients)}개 수집")
                
                time.sleep(0.5)  # API 과부하 방지
                
            except Exception as e:
                print(f"  ❌ 배치 {start_idx}-{end_idx} 수집 실패: {e}")
                continue
        
        print(f"✅ 레시피 재료정보 수집 완료: {len(all_ingredients)}개")
        self.recipe_ingredients = all_ingredients
        return all_ingredients
    
    def fetch_recipe_process_data(self) -> List[Dict[str, Any]]:
        """레시피 과정정보 수집 (3022개)"""
        print("👨‍🍳 레시피 과정정보 수집 중...")
        
        all_processes = []
        
        # 1000개씩 분할 수집
        for start_idx in range(1, 4001, 1000):
            end_idx = min(start_idx + 999, 4000)
            url = self.build_api_url(RECIPE_PROCESS_SERVICE_ID, start_idx, end_idx)
            
            try:
                print(f"  과정정보 배치 {start_idx}-{end_idx} 수집 중...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if RECIPE_PROCESS_SERVICE_ID in data and 'row' in data[RECIPE_PROCESS_SERVICE_ID]:
                    rows = data[RECIPE_PROCESS_SERVICE_ID]['row']
                    processes = rows if isinstance(rows, list) else [rows]
                    all_processes.extend(processes)
                    print(f"  ✅ 배치 {start_idx}-{end_idx}: {len(processes)}개 수집")
                
                time.sleep(0.5)  # API 과부하 방지
                
            except Exception as e:
                print(f"  ❌ 배치 {start_idx}-{end_idx} 수집 실패: {e}")
                continue
        
        print(f"✅ 레시피 과정정보 수집 완료: {len(all_processes)}개")
        self.recipe_processes = all_processes
        return all_processes
    
    def collect_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """모든 레시피 데이터 수집"""
        print("🚀 농림축산식품 공공데이터 수집 시작")
        
        # 1단계: API 연결 테스트
        print("\n=== 1단계: API 연결 테스트 ===")
        if not self.test_api_connection():
            print("❌ API 연결 테스트 실패. 종료합니다.")
            return {}
        
        # 2단계: 레시피 기본정보 수집
        print("\n=== 2단계: 레시피 기본정보 수집 ===")
        basic_data = self.fetch_recipe_basic_data()
        
        # 3단계: 레시피 재료정보 수집
        print("\n=== 3단계: 레시피 재료정보 수집 ===")
        ingredient_data = self.fetch_recipe_ingredient_data()
        
        # 4단계: 레시피 과정정보 수집
        print("\n=== 4단계: 레시피 과정정보 수집 ===")
        process_data = self.fetch_recipe_process_data()
        
        return {
            'basic': basic_data,
            'ingredients': ingredient_data,
            'processes': process_data
        }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """데이터 무결성 검증"""
        print("\n📊 데이터 무결성 검증 중...")
        
        # 레시피 코드 추출
        basic_codes = set()
        ingredient_codes = set()
        process_codes = set()
        
        for item in self.recipe_basic:
            code = item.get('RECIPE_ID') or item.get('레시피일련번호')
            if code:
                basic_codes.add(str(code))
        
        for item in self.recipe_ingredients:
            code = item.get('RECIPE_ID') or item.get('레시피일련번호')
            if code:
                ingredient_codes.add(str(code))
        
        for item in self.recipe_processes:
            code = item.get('RECIPE_ID') or item.get('레시피일련번호')
            if code:
                process_codes.add(str(code))
        
        # 무결성 분석
        integrity_report = {
            'basic_count': len(basic_codes),
            'ingredient_count': len(ingredient_codes),
            'process_count': len(process_codes),
            'basic_ingredient_match': len(basic_codes & ingredient_codes),
            'basic_process_match': len(basic_codes & process_codes),
            'all_match': len(basic_codes & ingredient_codes & process_codes),
            'orphan_ingredients': len(ingredient_codes - basic_codes),
            'orphan_processes': len(process_codes - basic_codes)
        }
        
        print(f"✅ 데이터 무결성 분석 완료")
        print(f"   기본정보 레시피 수: {integrity_report['basic_count']}")
        print(f"   재료정보 레시피 수: {integrity_report['ingredient_count']}")
        print(f"   과정정보 레시피 수: {integrity_report['process_count']}")
        print(f"   기본-재료 매칭: {integrity_report['basic_ingredient_match']}")
        print(f"   기본-과정 매칭: {integrity_report['basic_process_match']}")
        print(f"   완전 매칭: {integrity_report['all_match']}")
        
        return integrity_report
    
    def save_collected_data(self, data: Dict[str, List[Dict[str, Any]]]):
        """수집된 데이터 저장"""
        print("\n💾 수집된 데이터 저장 중...")
        
        # 메타데이터 생성
        metadata = {
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_source': self.base_url,
            'api_key_prefix': self.api_key[:10] + '...',
            'basic_count': len(data.get('basic', [])),
            'ingredient_count': len(data.get('ingredients', [])),
            'process_count': len(data.get('processes', [])),
            'integrity_report': self.validate_data_integrity()
        }
        
        # 개별 파일 저장
        try:
            # 기본정보 저장
            basic_data = {
                'metadata': metadata,
                'basic_info': data.get('basic', [])
            }
            with open(RECIPE_BASIC_PATH, 'w', encoding='utf-8') as f:
                json.dump(basic_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 기본정보 저장: {RECIPE_BASIC_PATH}")
            
            # 재료정보 저장
            ingredient_data = {
                'metadata': metadata,
                'ingredient_info': data.get('ingredients', [])
            }
            with open(RECIPE_INGREDIENT_PATH, 'w', encoding='utf-8') as f:
                json.dump(ingredient_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 재료정보 저장: {RECIPE_INGREDIENT_PATH}")
            
            # 과정정보 저장
            process_data = {
                'metadata': metadata,
                'process_info': data.get('processes', [])
            }
            with open(RECIPE_PROCESS_PATH, 'w', encoding='utf-8') as f:
                json.dump(process_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 과정정보 저장: {RECIPE_PROCESS_PATH}")
            
            # 통합 데이터 저장 (기존 시스템 호환성)
            integrated_data = {
                'metadata': metadata,
                'recipes': data.get('basic', [])  # 기본정보를 메인으로 사용
            }
            with open(RAW_RECIPES_PATH, 'w', encoding='utf-8') as f:
                json.dump(integrated_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 통합 데이터 저장: {RAW_RECIPES_PATH}")
            
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
            raise
    
    def print_sample_data(self):
        """샘플 데이터 출력"""
        print(f"\n📋 샘플 데이터:")
        
        if self.recipe_basic:
            print(f"\n--- 기본정보 샘플 ---")
            sample = self.recipe_basic[0]
            for key, value in list(sample.items())[:5]:
                print(f"  {key}: {value}")
        
        if self.recipe_ingredients:
            print(f"\n--- 재료정보 샘플 ---")
            sample = self.recipe_ingredients[0]
            for key, value in list(sample.items())[:5]:
                print(f"  {key}: {value}")
        
        if self.recipe_processes:
            print(f"\n--- 과정정보 샘플 ---")
            sample = self.recipe_processes[0]
            for key, value in list(sample.items())[:5]:
                print(f"  {key}: {value}")

def main():
    """메인 실행 함수"""
    print("🚀 농림축산식품 공공데이터 레시피 수집을 시작합니다...")
    print(f"📍 API 키: {MAFRA_API_KEY[:10]}...")
    
    collector = MafraRecipeDataCollector(MAFRA_API_KEY)
    
    # 데이터 수집
    collected_data = collector.collect_all_data()
    
    if collected_data:
        # 무결성 검증
        collector.validate_data_integrity()
        
        # 데이터 저장
        collector.save_collected_data(collected_data)
        
        # 샘플 데이터 출력
        collector.print_sample_data()
        
        print(f"\n🎉 데이터 수집 완료!")
        print(f"   기본정보: {len(collected_data.get('basic', []))}개")
        print(f"   재료정보: {len(collected_data.get('ingredients', []))}개")
        print(f"   과정정보: {len(collected_data.get('processes', []))}개")
        
    else:
        print("❌ 데이터 수집에 실패했습니다.")
        print("\n🔍 문제 해결 방법:")
        print("1. API 키가 올바른지 확인")
        print("2. 인터넷 연결 상태 확인")
        print("3. 농림축산식품 공공데이터포털 API 서버 상태 확인")

if __name__ == "__main__":
    main()