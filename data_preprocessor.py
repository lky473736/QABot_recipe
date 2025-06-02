import json
import pandas as pd
import re
import os
from typing import List, Dict, Any, Tuple
import logging
from konlpy.tag import Okt
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeDataPreprocessor:
    def __init__(self, data_dir='recipe_data'):
        self.data_dir = data_dir
        self.okt = Okt()
        
        # QA 생성을 위한 템플릿들
        self.qa_templates = {
            'recipe_method': [
                "{name}은 어떻게 만들어요?",
                "{name} 만드는 법을 알려주세요",
                "{name} 레시피를 알려주세요",
                "{name}를 만들려면 어떻게 해야 하나요?"
            ],
            'ingredients': [
                "{name}의 재료는 무엇인가요?",
                "{name}에 들어가는 재료를 알려주세요",
                "{name}를 만들 때 필요한 재료는?",
                "{name} 재료 목록을 알려주세요"
            ],
            'calories': [
                "{name}의 칼로리는 얼마나 되나요?",
                "{name} 영양 정보를 알려주세요",
                "{name}는 몇 칼로리인가요?"
            ],
            'cooking_method': [
                "끓이는 요리에는 어떤 것들이 있나요?",
                "볶기로 만드는 한국 요리는?",
                "{method}로 만드는 요리를 추천해주세요"
            ],
            'category': [
                "찌개 종류에는 어떤 것들이 있나요?",
                "{category} 요리를 알려주세요",
                "한국의 전통 {category}는?"
            ],
            'nutrition': [
                "칼로리가 낮은 한국 요리는?",
                "단백질이 많은 한국 요리를 추천해주세요",
                "다이어트에 좋은 한국 음식은?"
            ]
        }
    
    def load_recipe_data(self) -> List[Dict]:
        """수집된 레시피 데이터 로드"""
        all_recipes = []
        
        # 모의 데이터 로드
        mock_file = os.path.join(self.data_dir, 'mock_recipes.json')
        if os.path.exists(mock_file):
            with open(mock_file, 'r', encoding='utf-8') as f:
                mock_data = json.load(f)
                all_recipes.extend(mock_data)
        
        # 실제 API 데이터 로드 (있는 경우)
        api_file = os.path.join(self.data_dir, 'food_safety_recipes.json')
        if os.path.exists(api_file):
            with open(api_file, 'r', encoding='utf-8') as f:
                api_data = json.load(f)
                all_recipes.extend(api_data)
        
        logger.info(f"총 {len(all_recipes)}개의 레시피 데이터 로드됨")
        return all_recipes
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정제
        text = re.sub(r'[^\w\s가-힣.,!?()]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_cooking_steps(self, recipe: Dict) -> List[str]:
        """요리 과정 추출"""
        steps = []
        
        for i in range(1, 21):  # MANUAL01 ~ MANUAL20
            manual_key = f"MANUAL{i:02d}"
            if manual_key in recipe and recipe[manual_key]:
                step = self.clean_text(recipe[manual_key])
                if step:
                    steps.append(step)
        
        return steps
    
    def create_qa_pairs(self, recipes: List[Dict]) -> List[Dict]:
        """레시피 데이터로부터 QA 쌍 생성"""
        qa_pairs = []
        
        for recipe in recipes:
            recipe_name = recipe.get('RCP_NM', '')
            if not recipe_name:
                continue
            
            # 1. 레시피 방법 질문
            qa_pairs.extend(self._create_recipe_method_qa(recipe))
            
            # 2. 재료 질문
            qa_pairs.extend(self._create_ingredients_qa(recipe))
            
            # 3. 칼로리/영양 정보 질문
            qa_pairs.extend(self._create_nutrition_qa(recipe))
            
            # 4. 조리법별 질문
            qa_pairs.extend(self._create_cooking_method_qa(recipe))
            
            # 5. 카테고리별 질문
            qa_pairs.extend(self._create_category_qa(recipe))
        
        # 일반적인 요리 질문들 추가
        qa_pairs.extend(self._create_general_qa(recipes))
        
        logger.info(f"총 {len(qa_pairs)}개의 QA 쌍 생성됨")
        return qa_pairs
    
    def _create_recipe_method_qa(self, recipe: Dict) -> List[Dict]:
        """레시피 만드는 법 QA 생성"""
        qa_pairs = []
        recipe_name = recipe.get('RCP_NM', '')
        
        # 조리 과정 추출
        steps = self.extract_cooking_steps(recipe)
        if not steps:
            return qa_pairs
        
        # 답변 생성
        answer = f"{recipe_name} 만드는 법:\n"
        for i, step in enumerate(steps, 1):
            answer += f"{i}. {step}\n"
        
        # 재료 정보 추가
        ingredients = recipe.get('RCP_PARTS_DTLS', '')
        if ingredients:
            answer = f"재료: {ingredients}\n\n" + answer
        
        # 질문 생성
        for template in self.qa_templates['recipe_method']:
            question = template.format(name=recipe_name)
            qa_pairs.append({
                'question': question,
                'answer': answer.strip(),
                'context': self._create_context_from_recipe(recipe),
                'recipe_id': recipe.get('RCP_SEQ', ''),
                'category': 'recipe_method'
            })
        
        return qa_pairs
    
    def _create_ingredients_qa(self, recipe: Dict) -> List[Dict]:
        """재료 관련 QA 생성"""
        qa_pairs = []
        recipe_name = recipe.get('RCP_NM', '')
        ingredients = recipe.get('RCP_PARTS_DTLS', '')
        
        if not ingredients:
            return qa_pairs
        
        for template in self.qa_templates['ingredients']:
            question = template.format(name=recipe_name)
            answer = f"{recipe_name}의 재료는 {ingredients}입니다."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': self._create_context_from_recipe(recipe),
                'recipe_id': recipe.get('RCP_SEQ', ''),
                'category': 'ingredients'
            })
        
        return qa_pairs
    
    def _create_nutrition_qa(self, recipe: Dict) -> List[Dict]:
        """영양 정보 QA 생성"""
        qa_pairs = []
        recipe_name = recipe.get('RCP_NM', '')
        
        # 영양 정보 수집
        nutrition_info = []
        if recipe.get('INFO_ENG'):
            nutrition_info.append(f"칼로리: {recipe['INFO_ENG']}")
        if recipe.get('INFO_CAR'):
            nutrition_info.append(f"탄수화물: {recipe['INFO_CAR']}")
        if recipe.get('INFO_PRO'):
            nutrition_info.append(f"단백질: {recipe['INFO_PRO']}")
        if recipe.get('INFO_FAT'):
            nutrition_info.append(f"지방: {recipe['INFO_FAT']}")
        if recipe.get('INFO_NA'):
            nutrition_info.append(f"나트륨: {recipe['INFO_NA']}")
        
        if not nutrition_info:
            return qa_pairs
        
        nutrition_text = ", ".join(nutrition_info)
        
        for template in self.qa_templates['calories']:
            question = template.format(name=recipe_name)
            answer = f"{recipe_name}의 영양 정보는 {nutrition_text}입니다."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': self._create_context_from_recipe(recipe),
                'recipe_id': recipe.get('RCP_SEQ', ''),
                'category': 'nutrition'
            })
        
        return qa_pairs
    
    def _create_cooking_method_qa(self, recipe: Dict) -> List[Dict]:
        """조리법별 QA 생성"""
        qa_pairs = []
        cooking_method = recipe.get('RCP_WAY2', '')
        recipe_name = recipe.get('RCP_NM', '')
        
        if not cooking_method:
            return qa_pairs
        
        # 조리법별 질문 생성 (확률적으로)
        if random.random() < 0.3:  # 30% 확률로 생성
            question = f"{cooking_method}로 만드는 요리를 추천해주세요"
            answer = f"{cooking_method}로 만드는 요리로는 {recipe_name}이 있습니다. {recipe_name}은 {recipe.get('RCP_PARTS_DTLS', '')}로 만들 수 있습니다."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': self._create_context_from_recipe(recipe),
                'recipe_id': recipe.get('RCP_SEQ', ''),
                'category': 'cooking_method'
            })
        
        return qa_pairs
    
    def _create_category_qa(self, recipe: Dict) -> List[Dict]:
        """카테고리별 QA 생성"""
        qa_pairs = []
        category = recipe.get('RCP_PAT2', '')
        recipe_name = recipe.get('RCP_NM', '')
        
        if not category:
            return qa_pairs
        
        # 카테고리별 질문 생성 (확률적으로)
        if random.random() < 0.2:  # 20% 확률로 생성
            question = f"{category} 요리를 알려주세요"
            answer = f"{category} 요리로는 {recipe_name}이 있습니다."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': self._create_context_from_recipe(recipe),
                'recipe_id': recipe.get('RCP_SEQ', ''),
                'category': 'category'
            })
        
        return qa_pairs
    
    def _create_general_qa(self, recipes: List[Dict]) -> List[Dict]:
        """일반적인 요리 질문들 생성"""
        qa_pairs = []
        
        # 조리법별 그룹핑
        cooking_methods = {}
        categories = {}
        
        for recipe in recipes:
            method = recipe.get('RCP_WAY2', '')
            category = recipe.get('RCP_PAT2', '')
            name = recipe.get('RCP_NM', '')
            
            if method and name:
                if method not in cooking_methods:
                    cooking_methods[method] = []
                cooking_methods[method].append(name)
            
            if category and name:
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
        
        # 조리법별 질문 생성
        for method, recipe_names in cooking_methods.items():
            if len(recipe_names) >= 2:
                question = f"{method}로 만드는 한국 요리에는 어떤 것들이 있나요?"
                answer = f"{method}로 만드는 한국 요리로는 {', '.join(recipe_names[:5])} 등이 있습니다."
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'context': self._create_context_from_recipes(recipes, method=method),
                    'recipe_id': 'general',
                    'category': 'general_cooking_method'
                })
        
        # 카테고리별 질문 생성
        for category, recipe_names in categories.items():
            if len(recipe_names) >= 2:
                question = f"한국의 전통 {category}에는 어떤 것들이 있나요?"
                answer = f"한국의 전통 {category}로는 {', '.join(recipe_names[:5])} 등이 있습니다."
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'context': self._create_context_from_recipes(recipes, category=category),
                    'recipe_id': 'general',
                    'category': 'general_category'
                })
        
        return qa_pairs
    
    def _create_context_from_recipe(self, recipe: Dict) -> str:
        """단일 레시피로부터 컨텍스트 생성"""
        context_parts = []
        
        # 기본 정보
        name = recipe.get('RCP_NM', '')
        method = recipe.get('RCP_WAY2', '')
        category = recipe.get('RCP_PAT2', '')
        
        if name:
            context_parts.append(f"요리명: {name}")
        if method:
            context_parts.append(f"조리방법: {method}")
        if category:
            context_parts.append(f"종류: {category}")
        
        # 영양 정보
        nutrition = []
        if recipe.get('INFO_ENG'):
            nutrition.append(f"칼로리: {recipe['INFO_ENG']}")
        if recipe.get('INFO_CAR'):
            nutrition.append(f"탄수화물: {recipe['INFO_CAR']}")
        if recipe.get('INFO_PRO'):
            nutrition.append(f"단백질: {recipe['INFO_PRO']}")
        if recipe.get('INFO_FAT'):
            nutrition.append(f"지방: {recipe['INFO_FAT']}")
        
        if nutrition:
            context_parts.append("영양정보: " + ", ".join(nutrition))
        
        # 재료
        ingredients = recipe.get('RCP_PARTS_DTLS', '')
        if ingredients:
            context_parts.append(f"재료: {ingredients}")
        
        # 조리 과정
        steps = self.extract_cooking_steps(recipe)
        if steps:
            context_parts.append("조리과정:")
            for i, step in enumerate(steps, 1):
                context_parts.append(f"{i}. {step}")
        
        return "\n".join(context_parts)
    
    def _create_context_from_recipes(self, recipes: List[Dict], method: str = None, category: str = None) -> str:
        """여러 레시피로부터 컨텍스트 생성"""
        filtered_recipes = recipes
        
        if method:
            filtered_recipes = [r for r in recipes if r.get('RCP_WAY2') == method]
        elif category:
            filtered_recipes = [r for r in recipes if r.get('RCP_PAT2') == category]
        
        context_parts = []
        for recipe in filtered_recipes[:3]:  # 최대 3개만
            recipe_context = self._create_context_from_recipe(recipe)
            context_parts.append(recipe_context)
        
        return "\n\n".join(context_parts)
    
    def save_korquad_format(self, qa_pairs: List[Dict], output_file: str):
        """KorQuAD 형식으로 데이터 저장"""
        korquad_data = {
            "version": "1.0",
            "data": []
        }
        
        # 컨텍스트별로 그룹핑
        context_groups = {}
        for qa in qa_pairs:
            context = qa['context']
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(qa)
        
        # KorQuAD 형식으로 변환
        for i, (context, qas) in enumerate(context_groups.items()):
            paragraphs = [{
                "context": context,
                "qas": []
            }]
            
            for qa in qas:
                # 답변의 시작 위치 찾기
                answer_text = qa['answer']
                start_idx = context.find(answer_text)
                if start_idx == -1:
                    # 정확한 매치가 없으면 부분 매치 시도
                    words = answer_text.split()[:3]  # 첫 3단어로 찾기
                    for word in words:
                        start_idx = context.find(word)
                        if start_idx != -1:
                            answer_text = word
                            break
                    
                    if start_idx == -1:
                        start_idx = 0
                        answer_text = answer_text[:50]  # 답변 길이 제한
                
                qas_item = {
                    "id": f"recipe_qa_{i}_{len(paragraphs[0]['qas'])}",
                    "question": qa['question'],
                    "answers": [{
                        "text": answer_text,
                        "answer_start": start_idx
                    }]
                }
                paragraphs[0]["qas"].append(qas_item)
            
            korquad_data["data"].append({
                "title": f"한국요리레시피_{i}",
                "paragraphs": paragraphs
            })
        
        # 파일 저장
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(korquad_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"KorQuAD 형식 데이터 저장 완료: {output_path}")
        return output_path
    
    def save_training_data(self, qa_pairs: List[Dict]):
        """학습용 데이터 여러 형식으로 저장"""
        
        # 1. 원본 QA 쌍 저장
        qa_file = os.path.join(self.data_dir, 'recipe_qa_pairs.json')
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 2. KorQuAD 형식 저장
        korquad_file = self.save_korquad_format(qa_pairs, 'recipe_korquad.json')
        
        # 3. CSV 형식 저장
        df = pd.DataFrame(qa_pairs)
        csv_file = os.path.join(self.data_dir, 'recipe_qa_pairs.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 4. 학습/검증 분할
        train_size = int(len(qa_pairs) * 0.8)
        train_data = qa_pairs[:train_size]
        val_data = qa_pairs[train_size:]
        
        train_file = os.path.join(self.data_dir, 'train_qa_pairs.json')
        val_file = os.path.join(self.data_dir, 'val_qa_pairs.json')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"학습 데이터 저장 완료:")
        logger.info(f"  - 전체 QA: {len(qa_pairs)}개 ({qa_file})")
        logger.info(f"  - KorQuAD 형식: {korquad_file}")
        logger.info(f"  - CSV 형식: {csv_file}")
        logger.info(f"  - 학습 데이터: {len(train_data)}개 ({train_file})")
        logger.info(f"  - 검증 데이터: {len(val_data)}개 ({val_file})")
        
        return {
            'qa_pairs': qa_file,
            'korquad': korquad_file,
            'csv': csv_file,
            'train': train_file,
            'validation': val_file
        }
    
    def generate_statistics(self, qa_pairs: List[Dict]) -> Dict:
        """데이터 통계 생성"""
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'categories': {},
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'avg_context_length': 0
        }
        
        question_lengths = []
        answer_lengths = []
        context_lengths = []
        
        for qa in qa_pairs:
            category = qa.get('category', 'unknown')
            if category not in stats['categories']:
                stats['categories'][category] = 0
            stats['categories'][category] += 1
            
            question_lengths.append(len(qa['question']))
            answer_lengths.append(len(qa['answer']))
            context_lengths.append(len(qa['context']))
        
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
            stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths)
            stats['avg_context_length'] = sum(context_lengths) / len(context_lengths)
        
        return stats
    
    def process_all(self):
        """전체 전처리 프로세스 실행"""
        logger.info("데이터 전처리 시작")
        
        # 1. 데이터 로드
        recipes = self.load_recipe_data()
        
        # 2. QA 쌍 생성
        qa_pairs = self.create_qa_pairs(recipes)
        
        # 3. 데이터 저장
        file_paths = self.save_training_data(qa_pairs)
        
        # 4. 통계 생성
        stats = self.generate_statistics(qa_pairs)
        
        # 5. 통계 저장
        stats_file = os.path.join(self.data_dir, 'data_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("데이터 전처리 완료")
        logger.info(f"통계: {stats}")
        
        return {
            'file_paths': file_paths,
            'statistics': stats,
            'total_recipes': len(recipes),
            'total_qa_pairs': len(qa_pairs)
        }

if __name__ == "__main__":
    preprocessor = RecipeDataPreprocessor()
    result = preprocessor.process_all()
    
    print("=== 데이터 전처리 완료 ===")
    print(f"처리된 레시피 수: {result['total_recipes']}")
    print(f"생성된 QA 쌍 수: {result['total_qa_pairs']}")
    print(f"카테고리별 분포: {result['statistics']['categories']}")
    print(f"생성된 파일들: {list(result['file_paths'].values())}")