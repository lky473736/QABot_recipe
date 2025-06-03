import requests
import xmltodict
import json
import time
import re

API_KEY = '0662ae02bb6549ed8e0b'
SERVICE_ID = 'COOKRCP01'
DATA_TYPE = 'xml'
START_INDEX = 1
END_INDEX = 1000

# 유사한 레시피 이름인지 확인
def is_similar_recipe(name1, name2):
    if name1 == name2:
        return True
    if name1 in name2 or name2 in name1:
        return True
    base1 = re.sub(r'(찌개|찌게|국|탕|볶음|구이|찜|조림|무침|전)$', '', name1)
    base2 = re.sub(r'(찌개|찌게|국|탕|볶음|구이|찜|조림|무침|전)$', '', name2)
    return base1 == base2 and base1

# 원형 레시피인지 확인
def is_base_recipe(name):
    if not name:
        return False
    if re.search(r'\d|[a-zA-Z!@#$%^&*()_+=]', name):
        return False
    if len(name) > 15:
        return False
    return True

def fetch_recipes(start, end):
    url = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/{SERVICE_ID}/{DATA_TYPE}/{start}/{end}"
    try:
        response = requests.get(url, timeout=10)
        data_dict = xmltodict.parse(response.content)
        recipes = data_dict.get(SERVICE_ID, {}).get('row', [])
        print(f"✅ Fetched {len(recipes)} recipes from {start} to {end}")
        return recipes
    except Exception as e:
        print(f"❌ Error fetching recipes {start}-{end}: {e}")
        return []

def collect_all_recipes():
    all_recipes = []
    for start in range(1, 20000, 1000):  # 최대 20,000개까지 수집
        end = start + 999
        batch = fetch_recipes(start, end)
        if not batch:
            continue
        all_recipes.extend(batch)
        time.sleep(0.3)
    print(f"📦 총 수집된 레시피 수: {len(all_recipes)}")
    return all_recipes

def filter_unique_recipes(recipes):
    unique = []
    seen_names = []
    for r in recipes:
        name = r.get('RCP_NM', '').strip()
        if not is_base_recipe(name):
            continue
        if any(is_similar_recipe(name, s) for s in seen_names):
            continue
        unique.append(r)
        seen_names.append(name)
    print(f"✨ 필터링 후 유니크 레시피 수: {len(unique)}")
    return unique

def save_recipes_to_json(recipes, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    print(f"💾 저장 완료: {filename}")

def main():
    all_recipes = collect_all_recipes()
    if not all_recipes:
        print("❌ 레시피 수집 실패")
        return
    unique_recipes = filter_unique_recipes(all_recipes)
    save_recipes_to_json(unique_recipes, 'unique_recipes.json')

if __name__ == '__main__':
    main()
