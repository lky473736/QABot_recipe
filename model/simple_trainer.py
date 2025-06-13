"""
간단한 테스트용 모델 훈련기
- 메모리 최적화
- 디버깅 출력 강화
- 단계별 진행 확인
"""
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import sys
import os
import time
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class SimpleQADataset(Dataset):
    """간단한 QA 데이터셋"""
    
    def __init__(self, qa_data: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        print(f"📊 데이터셋 초기화 중... (최대 길이: {max_length})")
        
        self.qa_data = qa_data[:100]  # 테스트용으로 100개만 사용
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"✅ 데이터셋 생성 완료: {len(self.qa_data)}개 QA 쌍")
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        question = str(qa.get('question', '')).strip()
        answer = str(qa.get('answer', '')).strip()
        
        # 간단한 인코딩
        encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'question': question,
            'answer': answer
        }

class SimpleRecipeModel(nn.Module):
    """간단한 레시피 모델"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base"):
        super(SimpleRecipeModel, self).__init__()
        
        print(f"🔧 모델 초기화: {model_name}")
        
        try:
            # 모델 크기 줄이기
            self.bert = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=False,
                output_attentions=False
            )
            print("✅ BERT 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ BERT 모델 로드 실패: {e}")
            print("🔄 더 작은 모델로 대체...")
            self.bert = AutoModel.from_pretrained("klue/bert-base")
        
        # 간단한 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        print("✅ 모델 초기화 완료")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return logits

class SimpleTrainer:
    """간단한 훈련기"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base"):
        print("🚀 간단한 훈련기 초기화 중...")
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 사용 디바이스: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # 토크나이저 로드
        try:
            print(f"📥 토크나이저 로드 중: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            print("✅ 토크나이저 로드 완료")
        except Exception as e:
            print(f"❌ 토크나이저 로드 실패: {e}")
            print("🔄 기본 토크나이저 사용")
            self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        
        self.model_name = model_name
        self.model = None
        self.train_dataset = None
        
    def load_qa_data(self, qa_dataset_path: str) -> List[Dict[str, Any]]:
        """QA 데이터 로드"""
        print(f"📂 QA 데이터 로드 중: {qa_dataset_path}")
        
        try:
            with open(qa_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            qa_data = []
            if isinstance(data, dict) and 'qa_pairs' in data:
                qa_data = data['qa_pairs']
                print(f"📈 메타데이터 확인됨")
            elif isinstance(data, list):
                qa_data = data
            
            # 유효성 검사 및 필터링
            valid_qa = []
            for qa in qa_data:
                if (isinstance(qa, dict) and 
                    qa.get('question') and 
                    qa.get('answer') and
                    len(str(qa['question']).strip()) >= 3 and
                    len(str(qa['answer']).strip()) >= 5):
                    valid_qa.append(qa)
            
            print(f"✅ 유효한 QA: {len(valid_qa)}개")
            
            # 테스트용으로 처음 200개만 사용
            test_qa = valid_qa[:200]
            print(f"🧪 테스트용 QA: {len(test_qa)}개 사용")
            
            return test_qa
            
        except Exception as e:
            print(f"❌ QA 데이터 로드 실패: {e}")
            return []
    
    def prepare_data(self, qa_dataset_path: str):
        """데이터 준비"""
        print("🚀 데이터 준비 시작...")
        
        qa_data = self.load_qa_data(qa_dataset_path)
        
        if not qa_data:
            raise ValueError("유효한 QA 데이터를 찾을 수 없습니다.")
        
        # 간단한 분할 (80:20)
        split_idx = int(len(qa_data) * 0.8)
        train_data = qa_data[:split_idx]
        val_data = qa_data[split_idx:]
        
        print(f"✅ 훈련 데이터: {len(train_data)}개")
        print(f"✅ 검증 데이터: {len(val_data)}개")
        
        # 데이터셋 생성 (더 작은 max_length 사용)
        self.train_dataset = SimpleQADataset(train_data, self.tokenizer, max_length=128)
        self.val_dataset = SimpleQADataset(val_data, self.tokenizer, max_length=128)
        
        return self.train_dataset, self.val_dataset
    
    def initialize_model(self):
        """모델 초기화"""
        print("🔧 모델 초기화 중...")
        
        try:
            self.model = SimpleRecipeModel(self.model_name)
            self.model.to(self.device)
            
            # 모델 파라미터 수 계산
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"✅ 모델 초기화 완료")
            print(f"   총 파라미터: {total_params:,}")
            print(f"   훈련 가능 파라미터: {trainable_params:,}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            raise
    
    def train(self, num_epochs: int = 1, batch_size: int = 4, learning_rate: float = 5e-5):
        """간단한 훈련"""
        print("🚀 간단한 훈련 시작...")
        
        if self.model is None:
            self.initialize_model()
        
        if self.train_dataset is None:
            raise ValueError("훈련 데이터가 준비되지 않았습니다.")
        
        # 작은 배치 크기 사용
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # 윈도우 호환성
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # 옵티마이저
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 손실 함수
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"📊 훈련 설정:")
        print(f"   에포크: {num_epochs}")
        print(f"   배치 크기: {batch_size}")
        print(f"   학습률: {learning_rate}")
        print(f"   훈련 배치 수: {len(train_loader)}")
        print(f"   검증 배치 수: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"에포크 {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # 훈련
            self.model.train()
            total_loss = 0
            
            print("🏃‍♂️ 훈련 시작...")
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="훈련")):
                try:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # 순전파
                    logits = self.model(input_ids, attention_mask)
                    
                    # 타겟 (모든 QA를 positive로)
                    targets = torch.ones(input_ids.size(0), 1).to(self.device)
                    
                    # 손실 계산
                    loss = criterion(logits, targets)
                    
                    # 역전파
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # 메모리 정리
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n❌ 배치 {batch_idx} 훈련 오류: {e}")
                    continue
            
            avg_loss = total_loss / len(train_loader)
            print(f"🎯 평균 훈련 손실: {avg_loss:.4f}")
            
            # 검증 (간단히)
            self.model.eval()
            val_loss = 0
            
            print("📊 검증 시작...")
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="검증"):
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        
                        logits = self.model(input_ids, attention_mask)
                        targets = torch.ones(input_ids.size(0), 1).to(self.device)
                        
                        loss = criterion(logits, targets)
                        val_loss += loss.item()
                        
                    except Exception as e:
                        print(f"❌ 검증 오류: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            print(f"📊 평균 검증 손실: {avg_val_loss:.4f}")
        
        print(f"\n🎉 훈련 완료!")
        return True
    
    def save_model(self, save_path):
        """모델 저장 (간단 버전)"""
        print(f"💾 모델 저장 중: {save_path}")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 모델 가중치만 저장
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
            # 간단한 설정 저장
            config = {
                'model_name': self.model_name,
                'model_type': 'SimpleRecipeModel',
                'training_completed': True,
                'note': 'simple_test_model'
            }
            
            with open(save_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print("✅ 간단한 모델 저장 완료!")
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")

def main():
    """메인 훈련 함수"""
    print("🚀 간단한 레시피 챗봇 모델 훈련 시작!")
    print("⚡ 이 버전은 테스트용으로 빠르게 실행됩니다.")
    
    # 진행 상황 출력
    def print_progress(message):
        print(f"\n{'='*60}")
        print(f"📍 {message}")
        print(f"{'='*60}")
        time.sleep(1)  # 확인용 대기
    
    print_progress("1단계: QA 데이터 확인")
    
    if not QA_DATASET_PATH.exists():
        print(f"❌ QA 데이터셋을 찾을 수 없습니다: {QA_DATASET_PATH}")
        print("먼저 enhanced_qa_generator.py를 실행해주세요.")
        return
    
    try:
        print_progress("2단계: 훈련기 초기화")
        trainer = SimpleTrainer(MODEL_NAME)
        
        print_progress("3단계: 데이터 준비")
        trainer.prepare_data(QA_DATASET_PATH)
        
        print_progress("4단계: 모델 훈련")
        success = trainer.train(
            num_epochs=1,     # 1 에포크만
            batch_size=4,     # 작은 배치
            learning_rate=5e-5
        )
        
        if success:
            print_progress("5단계: 모델 저장")
            trainer.save_model(TRAINED_MODEL_DIR)
            print(f"\n🎉 간단한 훈련 완료!")
        
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()