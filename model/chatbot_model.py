"""
수정된 레시피 챗봇 모델 클래스 - 데이터 로딩 문제 해결
"""
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.text_preprocessor import TextPreprocessor

class RecipeChatbot:
    """레시피 챗봇 클래스 - 데이터 로딩 문제 해결"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.text_processor = TextPreprocessor()
        
        print(f"🔧 챗봇 초기화 중...")
        
        # 데이터 로드 (올바른 파일들 로드)
        self.recipes = self.load_recipes()
        self.qa_dataset = self.load_qa_dataset()
        
        print(f"📊 로드된 레시피 수: {len(self.recipes)}")
        print(f"📊 로드된 QA 수: {len(self.qa_dataset)}")
        
        # 데이터가 적으면 경고
        if len(self.recipes) < 100:
            print(f"⚠️ 레시피가 {len(self.recipes)}개밖에 없습니다. 더 많은 데이터가 필요할 수 있습니다.")
        
        if len(self.qa_dataset) < 100:
            print(f"⚠️ QA가 {len(self.qa_dataset)}개밖에 없습니다. 더 많은 QA 데이터가 필요할 수 있습니다.")
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 사전 훈련된 KcBERT 사용
            self.load_pretrained_model()
    
    def load_recipes(self) -> List[Dict[str, Any]]:
        """레시피 데이터 로드 - 올바른 구조 처리"""
        try:
            if PROCESSED_RECIPES_PATH.exists():
                print(f"📂 레시피 파일 로딩: {PROCESSED_RECIPES_PATH}")
                with open(PROCESSED_RECIPES_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 구조 분석
                recipes = []
                
                if isinstance(data, dict):
                    if 'metadata' in data and 'recipes' in data:
                        # 메타데이터가 있는 구조
                        recipes = data['recipes']
                        print(f"✅ 메타데이터 구조에서 레시피 로드")
                        print(f"📈 메타데이터: {data['metadata']}")
                    elif 'recipes' in data:
                        recipes = data['recipes']
                    else:
                        # 다른 키에서 리스트 찾기
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], dict) and 'name' in value[0]:
                                    recipes = value
                                    print(f"✅ '{key}' 키에서 레시피 로드")
                                    break
                elif isinstance(data, list):
                    recipes = data
                
                # 유효한 레시피만 필터링
                valid_recipes = []
                for recipe in recipes:
                    if isinstance(recipe, dict) and recipe.get('name'):
                        valid_recipes.append(recipe)
                
                print(f"📊 유효한 레시피: {len(valid_recipes)}개")
                return valid_recipes
            else:
                print(f"❌ 레시피 파일이 없습니다: {PROCESSED_RECIPES_PATH}")
                return []
                
        except Exception as e:
            print(f"❌ 레시피 로드 실패: {e}")
            return []
    
    def load_qa_dataset(self) -> List[Dict[str, Any]]:
        """QA 데이터셋 로드 - 올바른 구조 처리"""
        try:
            if QA_DATASET_PATH.exists():
                print(f"📂 QA 파일 로딩: {QA_DATASET_PATH}")
                with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 구조 분석
                qa_pairs = []
                
                if isinstance(data, dict):
                    if 'metadata' in data and 'qa_pairs' in data:
                        # 메타데이터가 있는 구조
                        qa_pairs = data['qa_pairs']
                        print(f"✅ 메타데이터 구조에서 QA 로드")
                        print(f"📈 메타데이터: {data['metadata']}")
                    elif 'qa_pairs' in data:
                        qa_pairs = data['qa_pairs']
                    else:
                        # 다른 키에서 리스트 찾기
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], dict) and 'question' in value[0]:
                                    qa_pairs = value
                                    print(f"✅ '{key}' 키에서 QA 로드")
                                    break
                elif isinstance(data, list):
                    qa_pairs = data
                
                # 유효한 QA만 필터링
                valid_qa = []
                for qa in qa_pairs:
                    if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                        valid_qa.append(qa)
                
                print(f"📊 유효한 QA: {len(valid_qa)}개")
                return valid_qa
            else:
                print(f"❌ QA 파일이 없습니다: {QA_DATASET_PATH}")
                return []
                
        except Exception as e:
            print(f"❌ QA 로드 실패: {e}")
            return []
    
    def load_pretrained_model(self):
        """사전 훈련된 KcBERT 모델 로드"""
        print("사전 훈련된 KcBERT 모델을 로드합니다...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            self.text_processor.load_tokenizer(MODEL_NAME)
            print("✅ KcBERT 로드 완료")
        except Exception as e:
            print(f"❌ KcBERT 로드 실패: {e}")
    
    def load_model(self, model_path: str):
        """훈련된 모델 로드"""
        print(f"훈련된 모델을 로드합니다: {model_path}")
        
        try:
            # 설정 로드
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"📋 모델 설정: {config}")
            else:
                print("⚠️ config.json이 없습니다. 기본 설정 사용.")
                config = {'model_name': MODEL_NAME}
            
            # 토크나이저 로드
            tokenizer_path = os.path.join(model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print("✅ 커스텀 토크나이저 로드")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(config.get('model_name', MODEL_NAME))
                print("✅ 기본 토크나이저 로드")
            
            # 모델 로드
            self.model = AutoModel.from_pretrained(config.get('model_name', MODEL_NAME))
            
            # 훈련된 가중치 로드
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                print("📦 훈련된 가중치 로드 중...")
                state_dict = torch.load(model_file, map_location=self.device)
                
                # 모델 구조가 다를 수 있으므로 strict=False 사용
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ 누락된 키: {len(missing_keys)}개")
                if unexpected_keys:
                    print(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
                
                print("✅ 훈련된 가중치 로드 완료")
            else:
                print("⚠️ pytorch_model.bin이 없습니다. 사전 훈련된 가중치 사용.")
            
            self.model.to(self.device)
            self.model.eval()
            self.text_processor.load_tokenizer(tokenizer_path if os.path.exists(tokenizer_path) else config.get('model_name', MODEL_NAME))
            
            print("✅ 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🔄 기본 KcBERT로 폴백")
            self.load_pretrained_model()
    
    def encode_text(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
        if not self.tokenizer or not self.model:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        try:
            # 텍스트 전처리
            clean_text = self.text_processor.clean_text(text)
            
            # 토크나이징 (길이 제한)
            inputs = self.tokenizer(
                clean_text,
                add_special_tokens=True,
                max_length=300,  # KcBERT 차원에 맞춤
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 인코딩
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 안전한 임베딩 추출
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    # [CLS] 토큰 사용
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️ 텍스트 인코딩 실패: {e}")
            # 더미 임베딩 반환
            return np.zeros((1, 300))
    
    def find_similar_qa(self, question: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """유사한 QA 찾기"""
        if not self.qa_dataset:
            print("⚠️ QA 데이터셋이 없습니다.")
            return []
        
        try:
            # 질문 인코딩
            question_embedding = self.encode_text(question)
            
            # 모든 QA의 질문 인코딩
            qa_embeddings = []
            valid_qa = []
            
            for qa in self.qa_dataset[:100]:  # 성능을 위해 처음 100개만 사용
                try:
                    qa_embedding = self.encode_text(qa['question'])
                    qa_embeddings.append(qa_embedding)
                    valid_qa.append(qa)
                except:
                    continue
            
            if not qa_embeddings:
                print("⚠️ 유효한 QA 임베딩이 없습니다.")
                return []
            
            # 유사도 계산
            qa_embeddings = np.vstack(qa_embeddings)
            similarities = cosine_similarity(question_embedding, qa_embeddings)[0]
            
            # 상위 k개 선택
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.3:  # 임계값
                    results.append((valid_qa[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"⚠️ 유사 QA 찾기 실패: {e}")
            return []
    
    def search_recipes_by_ingredient(self, ingredient: str) -> List[Dict[str, Any]]:
        """재료로 레시피 검색"""
        matching_recipes = []
        
        for recipe in self.recipes:
            ingredients_text = recipe.get('ingredients', '')
            if ingredient in ingredients_text:
                matching_recipes.append(recipe)
        
        return matching_recipes[:5]
    
    def search_recipes_by_name(self, name: str) -> List[Dict[str, Any]]:
        """이름으로 레시피 검색"""
        matching_recipes = []
        
        for recipe in self.recipes:
            recipe_name = recipe.get('name', '')
            if name in recipe_name:
                matching_recipes.append(recipe)
        
        return matching_recipes[:3]
    
    def get_recipe_by_id(self, recipe_id: str) -> Dict[str, Any]:
        """ID로 레시피 찾기"""
        for recipe in self.recipes:
            if recipe.get('id') == recipe_id:
                return recipe
        return {}
    
    def format_recipe_response(self, recipe: Dict[str, Any], response_type: str = 'full') -> str:
        """레시피 응답 포맷팅"""
        if not recipe:
            return "해당 레시피를 찾을 수 없습니다."
        
        name = recipe.get('name', '알 수 없는 요리')
        
        if response_type == 'ingredients':
            ingredients = recipe.get('ingredients', '재료 정보가 없습니다.')
            return f"{name}의 재료:\n{ingredients}"
        
        elif response_type == 'steps':
            steps = recipe.get('steps', [])
            if steps:
                steps_text = []
                for i, step in enumerate(steps[:10], 1):
                    steps_text.append(f"{i}. {step}")
                return f"{name} 만드는 방법:\n\n" + "\n".join(steps_text)
            else:
                return f"{name}의 조리법 정보가 없습니다."
        
        elif response_type == 'nutrition':
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
                return f"{name}의 영양정보:\n" + "\n".join(nutrition_info)
            else:
                return f"{name}의 영양정보가 없습니다."
        
        else:  # full
            response = f"🍳 {name}\n\n"
            
            if recipe.get('ingredients'):
                response += f"📋 재료:\n{recipe['ingredients']}\n\n"
            
            steps = recipe.get('steps', [])
            if steps:
                response += "👨‍🍳 조리법:\n"
                for i, step in enumerate(steps[:5], 1):
                    response += f"{i}. {step}\n"
                if len(steps) > 5:
                    response += f"... (총 {len(steps)}단계)\n"
                response += "\n"
            
            nutrition_info = []
            if recipe.get('calories'):
                nutrition_info.append(f"칼로리: {recipe['calories']}kcal")
            if nutrition_info:
                response += f"📊 영양정보: {', '.join(nutrition_info)}\n"
            
            return response.strip()
    
    def classify_question_intent(self, question: str) -> str:
        """질문 의도 분류"""
        question_lower = question.lower()
        
        if any(word in question for word in ['재료', '뭐가 들어가', '필요한 재료']):
            return 'ingredients'
        elif any(word in question for word in ['만들', '조리', '어떻게', '방법', '레시피']):
            return 'steps'
        elif any(word in question for word in ['칼로리', '영양', '열량']):
            return 'nutrition'
        elif any(word in question for word in ['팁', '비법', '주의사항']):
            return 'tips'
        elif any(word in question for word in ['추천', '뭐', '무엇', '요리']):
            return 'recommendation'
        else:
            return 'general'
    
    def generate_response(self, user_input: str) -> str:
        """사용자 입력에 대한 응답 생성"""
        if not user_input.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 궁금한 것이 있으시면 언제든 물어보세요!"
        
        # 텍스트 전처리
        clean_input = self.text_processor.normalize_question(user_input)
        
        # 인사말 처리
        if any(greeting in clean_input for greeting in ['안녕', '헬로', '하이']):
            return "안녕하세요! 레시피 챗봇입니다. 요리 레시피나 재료에 대해 궁금한 것이 있으시면 언제든 물어보세요! 🍳"
        
        # 도움말 처리
        if any(help_word in clean_input for help_word in ['도움', '도와줘', '뭐 해줄 수 있어']):
            return f"""레시피 챗봇이 도와드릴 수 있는 것들:

🔍 재료로 요리 검색: "감자로 뭐 만들 수 있어?"
📝 레시피 조리법: "김치찌개 만드는 법"
📋 요리 재료 확인: "불고기 재료가 뭐야?"
📊 영양정보 확인: "계란말이 칼로리"
💡 조리 팁: "파스타 만들 때 팁"

현재 {len(self.recipes)}개의 레시피와 {len(self.qa_dataset)}개의 QA 데이터를 보유하고 있습니다.
편하게 물어보세요!"""
        
        # 질문 의도 분류
        intent = self.classify_question_intent(clean_input)
        
        # 유사한 QA 찾기
        similar_qa = self.find_similar_qa(clean_input)
        
        if similar_qa and similar_qa[0][1] > 0.7:  # 높은 유사도
            return similar_qa[0][0]['answer']
        
        # 재료 추출 및 검색
        ingredients = self.text_processor.extract_ingredients(clean_input)
        if ingredients:
            recipes = self.search_recipes_by_ingredient(ingredients[0])
            if recipes:
                if intent == 'ingredients':
                    return self.format_recipe_response(recipes[0], 'ingredients')
                elif intent == 'steps':
                    return self.format_recipe_response(recipes[0], 'steps')
                elif intent == 'nutrition':
                    return self.format_recipe_response(recipes[0], 'nutrition')
                else:
                    recipe_names = [recipe['name'] for recipe in recipes]
                    return f"{ingredients[0]}로 만들 수 있는 요리들을 추천해드릴게요:\n\n" + "\n".join([f"• {name}" for name in recipe_names])
        
        # 레시피 이름 추출 및 검색
        recipe_name = self.text_processor.extract_recipe_name(clean_input)
        if recipe_name:
            recipes = self.search_recipes_by_name(recipe_name)
            if recipes:
                if intent == 'ingredients':
                    return self.format_recipe_response(recipes[0], 'ingredients')
                elif intent == 'steps':
                    return self.format_recipe_response(recipes[0], 'steps')
                elif intent == 'nutrition':
                    return self.format_recipe_response(recipes[0], 'nutrition')
                elif intent == 'tips':
                    return self.format_recipe_response(recipes[0], 'tips')
                else:
                    return self.format_recipe_response(recipes[0], 'full')
        
        # 유사한 QA가 있다면 사용
        if similar_qa:
            return similar_qa[0][0]['answer']
        
        # 기본 응답
        return f"""죄송해요, 해당 질문에 대한 답변을 찾을 수 없습니다. 😅

현재 시스템 상태:
• 레시피 수: {len(self.recipes)}개
• QA 데이터: {len(self.qa_dataset)}개

다음과 같이 질문해보세요:
• "감자로 뭐 만들 수 있어?"
• "김치찌개 만드는 법"
• "불고기 재료가 뭐야?"
• "계란말이 칼로리"

더 구체적으로 질문해주시면 더 정확한 답변을 드릴 수 있어요!"""