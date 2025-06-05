"""
개선된 레시피 챗봇 모델 클래스
- 훈련된 모델 로드 및 추론
- 향상된 질문 이해 및 답변 생성
- 의미적 유사도 기반 검색
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import re
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.text_preprocessor import TextPreprocessor

class EnhancedRecipeChatbotModel(nn.Module):
    """향상된 레시피 챗봇 모델 (추론용)"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base", hidden_dropout_prob: float = 0.1):
        super(EnhancedRecipeChatbotModel, self).__init__()
        
        # KcBERT 설정 로드
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = hidden_dropout_prob
        
        # KcBERT 모델 로드
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # 모델 차원
        self.hidden_size = self.bert.config.hidden_size
        
        # QA 매칭을 위한 헤드
        self.qa_classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # 임베딩 생성을 위한 프로젝션 헤드
        self.embedding_projection = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """순전파"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        qa_logits = self.qa_classifier(pooled_output)
        embeddings = self.embedding_projection(pooled_output)
        
        return {
            'qa_logits': qa_logits,
            'embeddings': embeddings,
            'pooled_output': pooled_output
        }
    
    def encode_question(self, input_ids, attention_mask):
        """질문만 인코딩 (검색용)"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            pooled_output = outputs.pooler_output
            embeddings = self.embedding_projection(pooled_output)
            
            return embeddings

class EnhancedRecipeChatbot:
    """개선된 레시피 챗봇 클래스"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.text_processor = TextPreprocessor()
        
        print(f"🤖 개선된 레시피 챗봇 초기화 중...")
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 데이터 로드
        self.recipes = self.load_enhanced_recipes()
        self.qa_dataset = self.load_enhanced_qa_dataset()
        
        print(f"📊 로드된 레시피 수: {len(self.recipes)}")
        print(f"📊 로드된 QA 수: {len(self.qa_dataset)}")
        
        # 검색 인덱스 구축
        self.recipe_index = self.build_recipe_index()
        self.qa_embeddings = None
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            self.load_trained_model(model_path)
        else:
            self.load_pretrained_model()
        
        # QA 임베딩 사전 계산
        if self.model and self.qa_dataset:
            self.precompute_qa_embeddings()
        
        print("✅ 챗봇 초기화 완료!")
    
    def load_enhanced_recipes(self) -> List[Dict[str, Any]]:
        """개선된 레시피 데이터 로드"""
        try:
            if PROCESSED_RECIPES_PATH.exists():
                print(f"📂 레시피 파일 로딩: {PROCESSED_RECIPES_PATH}")
                with open(PROCESSED_RECIPES_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                recipes = []
                if isinstance(data, dict) and 'recipes' in data:
                    recipes = data['recipes']
                    if 'metadata' in data:
                        print(f"📈 레시피 메타데이터: {data['metadata'].get('total_recipes', 0)}개")
                elif isinstance(data, list):
                    recipes = data
                
                # 유효한 레시피만 필터링
                valid_recipes = []
                for recipe in recipes:
                    if (isinstance(recipe, dict) and 
                        recipe.get('name') and 
                        recipe.get('main_ingredients')):
                        valid_recipes.append(recipe)
                
                print(f"✅ 유효한 레시피: {len(valid_recipes)}개")
                return valid_recipes
            else:
                print(f"❌ 레시피 파일 없음: {PROCESSED_RECIPES_PATH}")
                return []
                
        except Exception as e:
            print(f"❌ 레시피 로드 실패: {e}")
            return []
    
    def load_enhanced_qa_dataset(self) -> List[Dict[str, Any]]:
        """개선된 QA 데이터셋 로드"""
        try:
            if QA_DATASET_PATH.exists():
                print(f"📂 QA 파일 로딩: {QA_DATASET_PATH}")
                with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                qa_pairs = []
                if isinstance(data, dict) and 'qa_pairs' in data:
                    qa_pairs = data['qa_pairs']
                    if 'metadata' in data:
                        print(f"📈 QA 메타데이터: {data['metadata'].get('total_qa_pairs', 0)}개")
                elif isinstance(data, list):
                    qa_pairs = data
                
                # 유효한 QA만 필터링
                valid_qa = []
                for qa in qa_pairs:
                    if (isinstance(qa, dict) and 
                        qa.get('question') and 
                        qa.get('answer')):
                        valid_qa.append(qa)
                
                print(f"✅ 유효한 QA: {len(valid_qa)}개")
                return valid_qa
            else:
                print(f"❌ QA 파일 없음: {QA_DATASET_PATH}")
                return []
                
        except Exception as e:
            print(f"❌ QA 로드 실패: {e}")
            return []
    
    def build_recipe_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """레시피 검색 인덱스 구축"""
        print("🔍 레시피 검색 인덱스 구축 중...")
        
        index = {
            'by_ingredient': defaultdict(list),
            'by_name': defaultdict(list),
            'by_category': defaultdict(list),
            'by_cooking_method': defaultdict(list)
        }
        
        for recipe in self.recipes:
            recipe_name = recipe.get('name', '').lower()
            main_ingredients = recipe.get('main_ingredients', [])
            category = recipe.get('category', '').lower()
            cooking_method = recipe.get('cooking_method', '').lower()
            
            # 재료별 인덱스
            for ingredient in main_ingredients:
                if ingredient:
                    index['by_ingredient'][ingredient.lower()].append(recipe)
            
            # 이름별 인덱스 (부분 매칭을 위해 단어별로)
            for word in recipe_name.split():
                if len(word) >= 2:
                    index['by_name'][word].append(recipe)
            
            # 카테고리별 인덱스
            if category:
                index['by_category'][category].append(recipe)
            
            # 조리방법별 인덱스
            if cooking_method:
                index['by_cooking_method'][cooking_method].append(recipe)
        
        print(f"✅ 검색 인덱스 구축 완료")
        print(f"   재료: {len(index['by_ingredient'])}개")
        print(f"   이름: {len(index['by_name'])}개") 
        print(f"   카테고리: {len(index['by_category'])}개")
        print(f"   조리방법: {len(index['by_cooking_method'])}개")
        
        return index
    
    def load_pretrained_model(self):
        """사전 훈련된 모델 로드 (KcBERT)"""
        print("📥 사전 훈련된 KcBERT 모델 로드 중...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            
            self.model = EnhancedRecipeChatbotModel(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            
            self.text_processor.load_tokenizer(MODEL_NAME)
            print("✅ 사전 훈련된 모델 로드 완료")
        except Exception as e:
            print(f"❌ 사전 훈련된 모델 로드 실패: {e}")
    
    def load_trained_model(self, model_path: str):
        """훈련된 모델 로드"""
        print(f"📥 훈련된 모델 로드 중: {model_path}")
        
        try:
            # 설정 로드
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"📋 모델 설정: {config}")
                model_name = config.get('model_name', MODEL_NAME)
            else:
                print("⚠️ config.json 없음, 기본 설정 사용")
                model_name = MODEL_NAME
            
            # 토크나이저 로드
            tokenizer_path = os.path.join(model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print("✅ 훈련된 토크나이저 로드")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✅ 기본 토크나이저 로드")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            
            # 모델 초기화
            self.model = EnhancedRecipeChatbotModel(model_name)
            
            # 훈련된 가중치 로드
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                print("📦 훈련된 가중치 로드 중...")
                state_dict = torch.load(model_file, map_location=self.device)
                
                # 키 이름 매핑 (필요한 경우)
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    # 모델 구조 변경으로 인한 키 이름 수정
                    if key.startswith('module.'):
                        key = key[7:]  # 'module.' 제거
                    fixed_state_dict[key] = value
                
                # 모델에 가중치 로드
                missing_keys, unexpected_keys = self.model.load_state_dict(fixed_state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ 누락된 키: {len(missing_keys)}개")
                if unexpected_keys:
                    print(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
                
                print("✅ 훈련된 가중치 로드 완료")
            else:
                print("⚠️ pytorch_model.bin 없음, 사전 훈련된 가중치 사용")
            
            self.model.to(self.device)
            self.model.eval()
            self.text_processor.load_tokenizer(tokenizer_path if os.path.exists(tokenizer_path) else model_name)
            
            print("✅ 훈련된 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 훈련된 모델 로드 실패: {e}")
            print("🔄 사전 훈련된 모델로 폴백")
            self.load_pretrained_model()
    
    def precompute_qa_embeddings(self):
        """QA 임베딩 사전 계산"""
        print("💾 QA 임베딩 사전 계산 중...")
        
        try:
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(self.qa_dataset), batch_size):
                batch = self.qa_dataset[i:i+batch_size]
                batch_embeddings = []
                
                for qa in batch:
                    question = str(qa.get('question', '')).strip()
                    embedding = self.encode_text(question)
                    batch_embeddings.append(embedding.flatten())
                
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
            
            if embeddings:
                self.qa_embeddings = np.array(embeddings)
                print(f"✅ QA 임베딩 계산 완료: {self.qa_embeddings.shape}")
            else:
                print("❌ QA 임베딩 계산 실패")
                
        except Exception as e:
            print(f"❌ QA 임베딩 계산 오류: {e}")
            self.qa_embeddings = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
        if not self.tokenizer or not self.model:
            return np.zeros((1, 768))  # 기본 차원
        
        try:
            # 텍스트 전처리
            clean_text = self.text_processor.clean_text(text)
            
            # 토크나이징
            inputs = self.tokenizer(
                clean_text,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 인코딩
            with torch.no_grad():
                embeddings = self.model.encode_question(
                    inputs['input_ids'],
                    inputs['attention_mask']
                )
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"⚠️ 텍스트 인코딩 실패: {e}")
            return np.zeros((1, 768))
    
    def find_similar_qa(self, question: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """유사한 QA 찾기 (임베딩 기반)"""
        if not self.qa_dataset or self.qa_embeddings is None:
            return []
        
        try:
            # 질문 인코딩
            question_embedding = self.encode_text(question)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(question_embedding, self.qa_embeddings)[0]
            
            # 상위 k개 선택
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.3:  # 임계값
                    results.append((self.qa_dataset[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"⚠️ 유사 QA 찾기 실패: {e}")
            return []
    
    def search_recipes_by_ingredient(self, ingredient: str) -> List[Dict[str, Any]]:
        """재료로 레시피 검색 (인덱스 기반)"""
        ingredient_lower = ingredient.lower()
        matching_recipes = []
        
        # 정확한 매칭
        if ingredient_lower in self.recipe_index['by_ingredient']:
            matching_recipes.extend(self.recipe_index['by_ingredient'][ingredient_lower])
        
        # 부분 매칭
        for indexed_ingredient, recipes in self.recipe_index['by_ingredient'].items():
            if (ingredient_lower in indexed_ingredient or 
                indexed_ingredient in ingredient_lower):
                matching_recipes.extend(recipes)
        
        # 중복 제거
        seen_ids = set()
        unique_recipes = []
        for recipe in matching_recipes:
            recipe_id = recipe.get('id', '')
            if recipe_id not in seen_ids:
                seen_ids.add(recipe_id)
                unique_recipes.append(recipe)
        
        return unique_recipes[:10]  # 최대 10개
    
    def search_recipes_by_name(self, name: str) -> List[Dict[str, Any]]:
        """이름으로 레시피 검색 (인덱스 기반)"""
        name_lower = name.lower()
        matching_recipes = []
        
        # 단어별 검색
        for word in name_lower.split():
            if len(word) >= 2 and word in self.recipe_index['by_name']:
                matching_recipes.extend(self.recipe_index['by_name'][word])
        
        # 전체 이름으로 검색
        for recipe in self.recipes:
            recipe_name = recipe.get('name', '').lower()
            if name_lower in recipe_name:
                matching_recipes.append(recipe)
        
        # 중복 제거 및 관련성 순 정렬
        seen_ids = set()
        unique_recipes = []
        for recipe in matching_recipes:
            recipe_id = recipe.get('id', '')
            if recipe_id not in seen_ids:
                seen_ids.add(recipe_id)
                unique_recipes.append(recipe)
        
        return unique_recipes[:5]  # 최대 5개
    
    def search_recipes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """카테고리로 레시피 검색"""
        category_lower = category.lower()
        
        if category_lower in self.recipe_index['by_category']:
            return self.recipe_index['by_category'][category_lower][:8]
        
        return []
    
    def search_recipes_by_cooking_method(self, method: str) -> List[Dict[str, Any]]:
        """조리방법으로 레시피 검색"""
        method_lower = method.lower()
        
        if method_lower in self.recipe_index['by_cooking_method']:
            return self.recipe_index['by_cooking_method'][method_lower][:8]
        
        return []
    
    def classify_question_intent(self, question: str) -> str:
        """질문 의도 분류 (개선된 버전)"""
        question_lower = question.lower()
        
        # 패턴 기반 의도 분류
        if any(word in question_lower for word in ['재료', '뭐가 들어가', '필요한 재료', '들어가는']):
            return 'ingredients'
        elif any(word in question_lower for word in ['만들', '조리', '어떻게', '방법', '레시피', '요리법', '조리법']):
            return 'cooking_method'
        elif any(word in question_lower for word in ['칼로리', '영양', '열량', '영양정보', '영양성분']):
            return 'nutrition'
        elif any(word in question_lower for word in ['팁', '비법', '주의사항', '노하우', '꿀팁']):
            return 'tips'
        elif any(word in question_lower for word in ['추천', '뭐', '무엇', '종류', '메뉴']):
            return 'recommendation'
        elif any(word in question_lower for word in ['볶음', '구이', '찜', '탕', '국', '찌개']):
            return 'cooking_method_search'
        else:
            return 'general'
    
    def extract_entities(self, question: str) -> Dict[str, List[str]]:
        """질문에서 엔티티 추출"""
        entities = {
            'ingredients': [],
            'recipe_names': [],
            'categories': [],
            'cooking_methods': []
        }
        
        # 재료 추출
        entities['ingredients'] = self.text_processor.extract_ingredients(question)
        
        # 레시피 이름 추출
        recipe_name = self.text_processor.extract_recipe_name(question)
        if recipe_name:
            entities['recipe_names'].append(recipe_name)
        
        # 카테고리 키워드 매칭
        categories = ['밑반찬', '메인반찬', '국', '탕', '찌개', '밥', '죽', '면', '후식', '간식']
        for category in categories:
            if category in question:
                entities['categories'].append(category)
        
        # 조리방법 키워드 매칭  
        methods = ['볶음', '구이', '찜', '조림', '튀김', '끓임', '무침']
        for method in methods:
            if method in question:
                entities['cooking_methods'].append(method)
        
        return entities
    
    def format_recipe_response(self, recipe: Dict[str, Any], response_type: str = 'full') -> str:
        """레시피 응답 포맷팅 (개선된 버전)"""
        if not recipe:
            return "해당 레시피를 찾을 수 없습니다."
        
        name = recipe.get('name', '알 수 없는 요리')
        
        if response_type == 'ingredients':
            ingredients = recipe.get('ingredients', '')
            main_ingredients = recipe.get('main_ingredients', [])
            
            response = f"🍳 {name}의 재료:\n\n"
            if main_ingredients:
                response += "주요 재료:\n"
                for ingredient in main_ingredients:
                    response += f"• {ingredient}\n"
                response += "\n"
            
            if ingredients and ingredients != ' '.join(main_ingredients):
                response += f"상세 재료:\n{ingredients}"
            
            return response
        
        elif response_type == 'cooking_method':
            steps = recipe.get('steps', [])
            if steps:
                response = f"👨‍🍳 {name} 만드는 방법:\n\n"
                for i, step in enumerate(steps[:8], 1):
                    response += f"{i}. {step}\n"
                
                # 추가 정보
                category = recipe.get('category', '')
                cooking_method = recipe.get('cooking_method', '')
                if category:
                    response += f"\n📂 카테고리: {category}"
                if cooking_method:
                    response += f"\n🔥 조리방법: {cooking_method}"
                
                return response
            else:
                return f"{name}의 조리법 정보가 없습니다."
        
        elif response_type == 'nutrition':
            nutrition = recipe.get('nutrition', {})
            if nutrition:
                response = f"📊 {name}의 영양정보:\n\n"
                
                nutrition_labels = {
                    'calories': '칼로리', 'carbs': '탄수화물',
                    'protein': '단백질', 'fat': '지방', 'sodium': '나트륨'
                }
                units = {
                    'calories': 'kcal', 'carbs': 'g',
                    'protein': 'g', 'fat': 'g', 'sodium': 'mg'
                }
                
                for key, label in nutrition_labels.items():
                    if key in nutrition:
                        unit = units.get(key, '')
                        response += f"• {label}: {nutrition[key]}{unit}\n"
                
                return response
            else:
                return f"{name}의 영양정보가 없습니다."
        
        else:  # full
            response = f"🍳 {name}\n\n"
            
            # 카테고리 및 조리방법
            category = recipe.get('category', '')
            cooking_method = recipe.get('cooking_method', '')
            if category:
                response += f"📂 카테고리: {category}\n"
            if cooking_method:
                response += f"🔥 조리방법: {cooking_method}\n"
            response += "\n"
            
            # 주요 재료
            main_ingredients = recipe.get('main_ingredients', [])
            if main_ingredients:
                response += "🥕 주요 재료:\n"
                for ingredient in main_ingredients[:5]:
                    response += f"• {ingredient}\n"
                response += "\n"
            
            # 조리 순서 (간략)
            steps = recipe.get('steps', [])
            if steps:
                response += "👨‍🍳 조리법 (요약):\n"
                for i, step in enumerate(steps[:3], 1):
                    response += f"{i}. {step}\n"
                if len(steps) > 3:
                    response += f"... (총 {len(steps)}단계)\n"
                response += "\n"
            
            # 영양정보 (간략)
            nutrition = recipe.get('nutrition', {})
            if nutrition.get('calories'):
                response += f"📊 칼로리: {nutrition['calories']}kcal\n"
            
            return response.strip()
    
    def generate_response(self, user_input: str) -> str:
        """사용자 입력에 대한 응답 생성 (개선된 버전)"""
        if not user_input.strip():
            return "무엇을 도와드릴까요? 레시피나 요리에 대해 궁금한 것이 있으시면 언제든 물어보세요!"
        
        # 텍스트 전처리
        clean_input = self.text_processor.normalize_question(user_input)
        
        # 인사말 처리
        if any(greeting in clean_input for greeting in ['안녕', '헬로', '하이', '처음']):
            return "안녕하세요! 레시피 챗봇입니다. 요리 레시피나 재료에 대해 궁금한 것이 있으시면 언제든 물어보세요! 🍳"
        
        # 도움말 처리
        if any(help_word in clean_input for help_word in ['도움', '도와줘', '뭐 해줄 수 있어', '기능']):
            return f"""레시피 챗봇이 도와드릴 수 있는 것들:

🔍 재료로 요리 검색: "감자로 뭐 만들 수 있어?"
📝 레시피 조리법: "김치찌개 만드는 법"
📋 요리 재료 확인: "불고기 재료가 뭐야?"
📊 영양정보 확인: "계란말이 칼로리"
💡 조리 팁: "파스타 맛있게 만드는 팁"
🗂️ 카테고리별 추천: "국물 요리 추천해줘"

현재 {len(self.recipes)}개의 레시피와 {len(self.qa_dataset)}개의 QA 데이터를 보유하고 있습니다.
편하게 물어보세요!"""
        
        # 질문 의도 분류
        intent = self.classify_question_intent(clean_input)
        
        # 엔티티 추출
        entities = self.extract_entities(clean_input)
        
        # 유사한 QA 찾기 (우선 시도)
        similar_qa = self.find_similar_qa(clean_input, top_k=3)
        
        if similar_qa and similar_qa[0][1] > 0.7:  # 높은 유사도
            return similar_qa[0][0]['answer']
        
        # 엔티티 기반 검색
        response = self.handle_entity_based_search(entities, intent)
        if response:
            return response
        
        # 중간 유사도 QA 사용
        if similar_qa and similar_qa[0][1] > 0.4:
            return similar_qa[0][0]['answer']
        
        # 기본 응답
        return f"""죄송해요, 해당 질문에 대한 정확한 답변을 찾을 수 없습니다. 😅

💡 다음과 같이 질문해보세요:
• "감자로 뭐 만들 수 있어?"
• "김치찌개 만드는 법"
• "불고기 재료가 뭐야?"
• "계란말이 칼로리"

현재 시스템 상태:
• 레시피 수: {len(self.recipes)}개
• QA 데이터: {len(self.qa_dataset)}개

더 구체적으로 질문해주시면 정확한 답변을 드릴 수 있어요!"""
    
    def handle_entity_based_search(self, entities: Dict[str, List[str]], intent: str) -> str:
        """엔티티 기반 검색 처리"""
        # 재료 검색
        if entities['ingredients']:
            ingredient = entities['ingredients'][0]
            recipes = self.search_recipes_by_ingredient(ingredient)
            if recipes:
                if intent == 'ingredients':
                    return self.format_recipe_response(recipes[0], 'ingredients')
                elif intent == 'cooking_method':
                    return self.format_recipe_response(recipes[0], 'cooking_method')
                elif intent == 'nutrition':
                    return self.format_recipe_response(recipes[0], 'nutrition')
                else:
                    # 레시피 목록 반환
                    recipe_names = [recipe['name'] for recipe in recipes[:5]]
                    return f"{ingredient}로 만들 수 있는 요리들을 추천해드릴게요:\n\n" + "\n".join([f"• {name}" for name in recipe_names])
        
        # 레시피 이름 검색
        if entities['recipe_names']:
            recipe_name = entities['recipe_names'][0]
            recipes = self.search_recipes_by_name(recipe_name)
            if recipes:
                if intent == 'ingredients':
                    return self.format_recipe_response(recipes[0], 'ingredients')
                elif intent == 'cooking_method':
                    return self.format_recipe_response(recipes[0], 'cooking_method')
                elif intent == 'nutrition':
                    return self.format_recipe_response(recipes[0], 'nutrition')
                else:
                    return self.format_recipe_response(recipes[0], 'full')
        
        # 카테고리 검색
        if entities['categories']:
            category = entities['categories'][0]
            recipes = self.search_recipes_by_category(category)
            if recipes:
                recipe_names = [recipe['name'] for recipe in recipes[:6]]
                return f"{category} 요리를 추천해드릴게요:\n\n" + "\n".join([f"• {name}" for name in recipe_names])
        
        # 조리방법 검색
        if entities['cooking_methods']:
            method = entities['cooking_methods'][0]
            recipes = self.search_recipes_by_cooking_method(method)
            if recipes:
                recipe_names = [recipe['name'] for recipe in recipes[:6]]
                return f"{method} 요리들을 추천해드릴게요:\n\n" + "\n".join([f"• {name}" for name in recipe_names])
        
        return None
