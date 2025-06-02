"""
텍스트 전처리 유틸리티
"""
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = None
        
    def load_tokenizer(self, model_name: str = "beomi/kcbert-base"):
        """토크나이저 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer
    
    def clean_text(self, text: str) -> str:
        """기본 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?()\-/\d]', '', text)
        
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        # 양쪽 공백 제거
        return text.strip()
    
    def normalize_question(self, question: str) -> str:
        """질문 정규화"""
        question = self.clean_text(question)
        
        # 존댓말 통일
        question = re.sub(r'요$|요\?$', '?', question)
        question = re.sub(r'습니다$|습니다\?$', '?', question)
        question = re.sub(r'해주세요$|해줘요$', '해줘', question)
        question = re.sub(r'알려주세요$|알려줘요$', '알려줘', question)
        
        # 물음표 정리
        if not question.endswith('?') and any(word in question for word in ['뭐', '어떻게', '언제', '왜', '어디']):
            question += '?'
        
        return question
    
    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        if not text:
            return []
        
        # 요리 관련 키워드
        cooking_keywords = [
            '요리', '레시피', '만들기', '조리법', '재료', '칼로리', '영양',
            '끓이기', '볶기', '굽기', '찌기', '튀기기', '무치기',
            '국', '찌개', '볶음', '구이', '찜', '탕', '죽', '밥', '면',
            '반찬', '간식', '후식', '메인'
        ]
        
        # 재료 키워드
        ingredient_keywords = [
            '쇠고기', '돼지고기', '닭고기', '생선', '새우', '오징어', '두부', '계란',
            '쌀', '면', '밀가루', '감자', '고구마', '양파', '마늘', '대파',
            '배추', '무', '당근', '호박', '브로콜리', '시금치', '버섯', '김치'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        # 요리 키워드 찾기
        for keyword in cooking_keywords + ingredient_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def tokenize_for_model(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """모델용 토크나이징"""
        if not self.tokenizer:
            raise ValueError("토크나이저가 로드되지 않았습니다. load_tokenizer()를 먼저 호출하세요.")
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def preprocess_qa_pair(self, question: str, answer: str, max_length: int = 512) -> Dict[str, Any]:
        """QA 쌍 전처리"""
        # 텍스트 정리
        clean_question = self.normalize_question(question)
        clean_answer = self.clean_text(answer)
        
        # 결합된 텍스트 생성 (BERT 스타일)
        combined_text = f"[CLS] {clean_question} [SEP] {clean_answer} [SEP]"
        
        # 토크나이징
        if self.tokenizer:
            encoded = self.tokenize_for_model(combined_text, max_length)
            return {
                'question': clean_question,
                'answer': clean_answer,
                'combined_text': combined_text,
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'token_type_ids': encoded.get('token_type_ids')
            }
        else:
            return {
                'question': clean_question,
                'answer': clean_answer,
                'combined_text': combined_text
            }
    
    def is_recipe_related(self, text: str) -> bool:
        """요리/레시피 관련 텍스트인지 판단"""
        cooking_indicators = [
            '요리', '레시피', '만들', '조리', '재료', '음식', '먹',
            '칼로리', '영양', '맛', '끓', '볶', '구', '찌', '튀',
            '국', '찌개', '밥', '면', '반찬', '간식'
        ]
        
        text_lower = text.lower()
        return any(indicator in text for indicator in cooking_indicators)
    
    def extract_recipe_name(self, text: str) -> str:
        """텍스트에서 레시피 이름 추출"""
        # 일반적인 레시피 이름 패턴
        patterns = [
            r'(\w+찌개)',
            r'(\w+볶음)',
            r'(\w+구이)',
            r'(\w+찜)',
            r'(\w+탕)',
            r'(\w+죽)',
            r'(\w+밥)',
            r'(\w+국)',
            r'(\w+면)',
            r'(\w+튀김)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 패턴에 맞지 않으면 키워드 기반 추출
        words = text.split()
        for word in words:
            if len(word) >= 2 and any(end in word for end in ['찌개', '볶음', '구이', '찜', '탕', '죽', '밥', '국', '면']):
                return word
        
        return ""
    
    def extract_ingredients(self, text: str) -> List[str]:
        """텍스트에서 재료 추출"""
        # 일반적인 재료들
        common_ingredients = [
            '쇠고기', '돼지고기', '닭고기', '생선', '새우', '오징어', '문어',
            '두부', '계란', '달걀', '쌀', '면', '국수', '밀가루',
            '감자', '고구마', '양파', '마늘', '대파', '생강',
            '배추', '무', '당근', '호박', '오이', '토마토',
            '브로콜리', '시금치', '상추', '버섯', '김치', '콩나물'
        ]
        
        found_ingredients = []
        for ingredient in common_ingredients:
            if ingredient in text:
                found_ingredients.append(ingredient)
        
        return found_ingredients