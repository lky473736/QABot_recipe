import requests
import json
import pandas as pd
import time
import os
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from urllib.parse import quote
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KoreanRecipeDataCollector:
    def __init__(self):
        # 공공데이터포털 API 키들 (실제 사용시 발급받아야 함)
        self.api_keys = {
            'food_safety': '0662ae02bb6549ed8e0b',  # 식약처 API 키
            'rural_dev': 'YOUR_RURAL_DEV_API_KEY',      # 농촌진흥청 API 키
            'agri_food': 'YOUR_AGRI_FOOD_API_KEY'       # 농림축산식품부 API 키
        }
        
        # API URL들
        self.api_urls = {
            'food_safety': 'http://openapi.foodsafetykorea.go.kr/api',
            'rural_dev': 'http://openapi.naas.go.kr/service',
            'agri_food': 'http://211.237.50.150:7080/openapi'
        }
        
        # 데이터 저장 디렉토리
        self.data_dir = 'recipe_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_food_safety_recipes(self, start_idx=1, end_idx=1000):
        """식품의약품안전처 레시피 데이터 수집"""
        logger.info("식약처 레시피 데이터 수집 시작")
        
        all_recipes = []
        batch_size = 100
        
        for start in range(start_idx, end_idx + 1, batch_size):
            end = min(start + batch_size - 1, end_idx)
            
            # API URL 구성
            url = f"{self.api_urls['food_safety']}/{self.api_keys['food_safety']}/COOKRCP01/json/{start}/{end}"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'COOKRCP01' in data and 'row' in data['COOKRCP01']:
                    recipes = data['COOKRCP01']['row']
                    all_recipes.extend(recipes)
                    logger.info(f"수집된 레시피: {start}-{end} ({len(recipes)}개)")
                else:
                    logger.warning(f"데이터 없음: {start}-{end}")
                    
                time.sleep(0.1)  # API 호출 제한 고려
                
            except requests.RequestException as e:
                logger.error(f"API 호출 실패 {start}-{end}: {e}")
                continue
        
        # JSON 파일로 저장
        output_file = os.path.join(self.data_dir, 'food_safety_recipes.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_recipes, f, ensure_ascii=False, indent=2)
        
        logger.info(f"식약처 레시피 데이터 저장 완료: {len(all_recipes)}개 ({output_file})")
        return all_recipes
    
    def collect_rural_dev_recipes(self):
        """농촌진흥청 농식품 데이터 수집"""
        logger.info("농촌진흥청 농식품 데이터 수집 시작")
        
        # 농촌진흥청 API는 XML 형태로 제공
        url = f"{self.api_urls['rural_dev']}/FoodNutritionInfo/getFoodNutritionInfo"
        
        params = {
            'serviceKey': self.api_keys['rural_dev'],
            'numOfRows': 1000,
            'pageNo': 1
        }
        
        all_foods = []
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # XML 파싱
            root = ET.fromstring(response.content)
            
            for item in root.findall('.//item'):
                food_data = {}
                for child in item:
                    food_data[child.tag] = child.text
                all_foods.append(food_data)
            
            # JSON 파일로 저장
            output_file = os.path.join(self.data_dir, 'rural_dev_foods.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_foods, f, ensure_ascii=False, indent=2)
            
            logger.info(f"농촌진흥청 데이터 저장 완료: {len(all_foods)}개 ({output_file})")
            
        except Exception as e:
            logger.error(f"농촌진흥청 API 호출 실패: {e}")
            
        return all_foods
    
    def collect_mock_data(self):
        """실제 API 키가 없을 때 사용할 모의 데이터 생성"""
        logger.info("모의 데이터 생성 시작")
        
        mock_recipes = [
            {
                "RCP_SEQ": "1",
                "RCP_NM": "김치찌개",
                "RCP_WAY2": "끓이기",
                "RCP_PAT2": "찌개",
                "INFO_WGT": "300g",
                "INFO_ENG": "180kcal",
                "INFO_CAR": "12g",
                "INFO_PRO": "15g",
                "INFO_FAT": "8g",
                "INFO_NA": "800mg",
                "RCP_PARTS_DTLS": "김치 200g, 돼지고기 100g, 두부 100g, 대파 20g, 마늘 5쪽, 고춧가루 1큰술",
                "MANUAL01": "김치를 한입 크기로 자른다",
                "MANUAL02": "돼지고기를 작은 크기로 자르고 볶아 기름을 낸다",
                "MANUAL03": "볶은 고기에 김치를 넣고 함께 볶는다",
                "MANUAL04": "물을 넣고 끓인 후 두부와 대파를 넣는다",
                "MANUAL05": "마지막에 마늘과 고춧가루로 간을 맞춘다",
                "ATT_FILE_NO_MAIN": "kimchi_jjigae.jpg",
                "HASH_TAG": "#김치찌개 #찌개 #한식 #집밥"
            },
            {
                "RCP_SEQ": "2", 
                "RCP_NM": "된장찌개",
                "RCP_WAY2": "끓이기",
                "RCP_PAT2": "찌개",
                "INFO_WGT": "250g",
                "INFO_ENG": "120kcal",
                "INFO_CAR": "8g",
                "INFO_PRO": "10g",
                "INFO_FAT": "5g",
                "INFO_NA": "600mg",
                "RCP_PARTS_DTLS": "된장 2큰술, 두부 100g, 호박 50g, 양파 30g, 버섯 20g, 멸치육수 300ml",
                "MANUAL01": "멸치로 육수를 우린다",
                "MANUAL02": "된장을 체에 걸러 육수에 푼다",
                "MANUAL03": "야채(호박, 양파, 버섯)를 넣고 끓인다",
                "MANUAL04": "두부를 넣고 한소끔 더 끓인다",
                "MANUAL05": "기호에 따라 파를 넣고 마무리한다",
                "ATT_FILE_NO_MAIN": "doenjang_jjigae.jpg",
                "HASH_TAG": "#된장찌개 #찌개 #한식 #건강식"
            },
            {
                "RCP_SEQ": "3",
                "RCP_NM": "고구마죽",
                "RCP_WAY2": "끓이기",
                "RCP_PAT2": "죽/스프",
                "INFO_WGT": "200g",
                "INFO_ENG": "150kcal",
                "INFO_CAR": "35g",
                "INFO_PRO": "3g",
                "INFO_FAT": "1g",
                "INFO_NA": "50mg",
                "RCP_PARTS_DTLS": "고구마 200g, 찹쌀 30g, 물 500ml, 소금 약간",
                "MANUAL01": "고구마는 껍질을 벗기고 적당한 크기로 자른다",
                "MANUAL02": "찹쌀은 미리 불려놓는다",
                "MANUAL03": "고구마를 삶아서 으깬다",
                "MANUAL04": "으깬 고구마와 찹쌀을 물과 함께 끓인다",
                "MANUAL05": "걸쭉해질 때까지 저어가며 끓이고 소금으로 간한다",
                "ATT_FILE_NO_MAIN": "sweet_potato_porridge.jpg",
                "HASH_TAG": "#고구마죽 #죽 #이유식 #건강식"
            },
            {
                "RCP_SEQ": "4",
                "RCP_NM": "불고기",
                "RCP_WAY2": "볶기",
                "RCP_PAT2": "구이",
                "INFO_WGT": "200g",
                "INFO_ENG": "280kcal",
                "INFO_CAR": "10g",
                "INFO_PRO": "25g",
                "INFO_FAT": "15g",
                "INFO_NA": "500mg",
                "RCP_PARTS_DTLS": "소고기 200g, 양파 100g, 당근 50g, 간장 3큰술, 설탕 1큰술, 마늘 3쪽, 배 50g",
                "MANUAL01": "소고기를 얇게 썰어 준비한다",
                "MANUAL02": "양파와 당근을 채썰고 마늘을 다진다",
                "MANUAL03": "배를 갈아서 고기와 함께 재운다",
                "MANUAL04": "간장, 설탕, 마늘로 양념장을 만든다",
                "MANUAL05": "팬에 고기와 야채를 넣고 양념장과 함께 볶는다",
                "ATT_FILE_NO_MAIN": "bulgogi.jpg",
                "HASH_TAG": "#불고기 #구이 #한식 #고기요리"
            },
            {
                "RCP_SEQ": "5",
                "RCP_NM": "잡채",
                "RCP_WAY2": "볶기",
                "RCP_PAT2": "나물",
                "INFO_WGT": "150g",
                "INFO_ENG": "200kcal",
                "INFO_CAR": "30g",
                "INFO_PRO": "8g",
                "INFO_FAT": "6g",
                "INFO_NA": "400mg",
                "RCP_PARTS_DTLS": "당면 100g, 시금치 50g, 당근 30g, 버섯 30g, 소고기 50g, 간장 2큰술, 참기름 1큰술",
                "MANUAL01": "당면을 끓는 물에 삶아 준비한다",
                "MANUAL02": "각종 채소를 채썰어 각각 볶는다",
                "MANUAL03": "소고기를 볶아 준비한다",
                "MANUAL04": "모든 재료를 한데 모아 간장과 참기름으로 무친다",
                "MANUAL05": "마지막에 깨를 뿌려 완성한다",
                "ATT_FILE_NO_MAIN": "japchae.jpg",
                "HASH_TAG": "#잡채 #나물 #한식 #명절음식"
            }
        ]
        
        # 확장 데이터 생성
        extended_recipes = []
        for i, base_recipe in enumerate(mock_recipes):
            # 기본 레시피 추가
            extended_recipes.append(base_recipe)
            
            # 변형 레시피들 생성
            variations = self._generate_recipe_variations(base_recipe, i)
            extended_recipes.extend(variations)
        
        # JSON 파일로 저장
        output_file = os.path.join(self.data_dir, 'mock_recipes.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extended_recipes, f, ensure_ascii=False, indent=2)
        
        logger.info(f"모의 데이터 생성 완료: {len(extended_recipes)}개 ({output_file})")
        return extended_recipes
    
    def _generate_recipe_variations(self, base_recipe, idx):
        """기본 레시피의 변형 버전들 생성"""
        variations = []
        
        # 김치찌개 변형들
        if base_recipe["RCP_NM"] == "김치찌개":
            variations.extend([
                {
                    **base_recipe,
                    "RCP_SEQ": f"{idx+1}_1",
                    "RCP_NM": "참치김치찌개",
                    "RCP_PARTS_DTLS": "김치 200g, 참치캔 1개, 두부 100g, 대파 20g, 마늘 5쪽",
                    "MANUAL02": "참치캔을 기름째 볶아 맛을 낸다",
                    "HASH_TAG": "#참치김치찌개 #찌개 #한식 #간편식"
                },
                {
                    **base_recipe,
                    "RCP_SEQ": f"{idx+1}_2", 
                    "RCP_NM": "순두부김치찌개",
                    "RCP_PARTS_DTLS": "김치 200g, 순두부 1모, 달걀 1개, 대파 20g",
                    "MANUAL04": "순두부를 넣고 끓인 후 달걀을 풀어넣는다",
                    "HASH_TAG": "#순두부김치찌개 #찌개 #한식 #부드러운맛"
                }
            ])
        
        # 된장찌개 변형들
        elif base_recipe["RCP_NM"] == "된장찌개":
            variations.extend([
                {
                    **base_recipe,
                    "RCP_SEQ": f"{idx+1}_1",
                    "RCP_NM": "시래기된장찌개", 
                    "RCP_PARTS_DTLS": "된장 2큰술, 시래기 100g, 두부 50g, 멸치육수 300ml",
                    "MANUAL03": "불린 시래기를 넣고 충분히 끓인다",
                    "HASH_TAG": "#시래기된장찌개 #찌개 #한식 #시골밥상"
                }
            ])
        
        return variations
    
    def collect_all_data(self):
        """모든 데이터 소스에서 데이터 수집"""
        logger.info("전체 데이터 수집 시작")
        
        all_data = {
            'food_safety_recipes': [],
            'rural_dev_foods': [],
            'mock_recipes': []
        }
        
        # API 키가 설정되어 있으면 실제 API 호출
        if self.api_keys['food_safety'] != 'YOUR_FOOD_SAFETY_API_KEY':
            all_data['food_safety_recipes'] = self.collect_food_safety_recipes()
        
        if self.api_keys['rural_dev'] != 'YOUR_RURAL_DEV_API_KEY':
            all_data['rural_dev_foods'] = self.collect_rural_dev_recipes()
        
        # 항상 모의 데이터는 생성
        all_data['mock_recipes'] = self.collect_mock_data()
        
        # 통합 데이터 파일 생성
        output_file = os.path.join(self.data_dir, 'all_recipe_data.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전체 데이터 수집 완료: {output_file}")
        return all_data

if __name__ == "__main__":
    collector = KoreanRecipeDataCollector()
    
    # API 키 설정 (실제 사용시 발급받은 키로 변경)
    # collector.api_keys['food_safety'] = '실제_식약처_API_키'
    # collector.api_keys['rural_dev'] = '실제_농촌진흥청_API_키'
    
    # 데이터 수집 실행
    all_data = collector.collect_all_data()
    
    print("데이터 수집 완료!")
    for key, data in all_data.items():
        print(f"{key}: {len(data)}개")