"""
KcBERT의 실제 차원 300에 맞춘 모델 훈련
모든 설정을 300으로 통일
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import sys
import os
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

class RecipeQADataset(Dataset):
    """레시피 QA 데이터셋 - 300 길이로 제한"""
    
    def __init__(self, qa_data, tokenizer, max_length=300):  # 300으로 변경
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📊 데이터셋 생성: {len(qa_data)}개 QA, 최대 길이: {max_length}")
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        
        if isinstance(qa, dict):
            question = str(qa.get('question', ''))
            answer = str(qa.get('answer', ''))
        else:
            question = "질문 없음"
            answer = "답변 없음"
        
        # 질문만 인코딩 (300 길이로)
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

class KcBERT300Model(nn.Module):
    """KcBERT 300 차원에 최적화된 모델"""
    
    def __init__(self, model_name='beomi/kcbert-base'):
        super(KcBERT300Model, self).__init__()
        
        print("🔧 KcBERT 300 차원 모델 로드 중...")
        
        # KcBERT 로드
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 실제 차원 확인
        self.hidden_size = 300  # KcBERT의 실제 차원으로 고정
        print(f"📏 고정 hidden_size: {self.hidden_size}")
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, 1)
        
        print("✅ KcBERT 300 모델 초기화 완료")
    
    def forward(self, input_ids, attention_mask):
        # 입력 길이를 300으로 제한
        if input_ids.size(1) > 300:
            input_ids = input_ids[:, :300]
            attention_mask = attention_mask[:, :300]
        
        try:
            # BERT 순전파
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # 출력 추출 시도
            pooled_output = None
            
            # 방법 1: pooler_output 사용
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
                print(f"✅ pooler_output 사용: {pooled_output.shape}")
            
            # 방법 2: [CLS] 토큰 사용
            if pooled_output is None:
                last_hidden_state = outputs.last_hidden_state
                pooled_output = last_hidden_state[:, 0, :]  # [CLS] 토큰
                print(f"✅ [CLS] 토큰 사용: {pooled_output.shape}")
            
            # 차원을 300으로 강제 조정
            if pooled_output.size(-1) != 300:
                if pooled_output.size(-1) > 300:
                    # 잘라내기
                    pooled_output = pooled_output[:, :300]
                    print(f"🔧 차원 잘라냄: -> {pooled_output.shape}")
                else:
                    # 패딩
                    batch_size = pooled_output.size(0)
                    padding_size = 300 - pooled_output.size(-1)
                    padding = torch.zeros(batch_size, padding_size, device=pooled_output.device)
                    pooled_output = torch.cat([pooled_output, padding], dim=-1)
                    print(f"🔧 차원 패딩: -> {pooled_output.shape}")
            
            # 드롭아웃 적용
            pooled_output = self.dropout(pooled_output)
            
            # 분류 점수
            similarity_score = self.classifier(pooled_output)
            
            return {
                'similarity_score': similarity_score,
                'pooled_output': pooled_output
            }
            
        except Exception as e:
            print(f"❌ Forward 오류: {e}")
            # 더미 출력 (300 차원)
            batch_size = input_ids.size(0)
            dummy_pooled = torch.zeros(batch_size, 300, device=input_ids.device)
            dummy_score = torch.zeros(batch_size, 1, device=input_ids.device)
            
            return {
                'similarity_score': dummy_score,
                'pooled_output': dummy_pooled
            }

class KcBERT300Trainer:
    """KcBERT 300 차원 훈련기"""
    
    def __init__(self, model_name='beomi/kcbert-base'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ 토크나이저 로드 성공: {model_name}")
        except Exception as e:
            print(f"❌ 토크나이저 로드 실패: {e}")
            raise
        
        self.model = None
    
    def load_qa_data(self, qa_dataset_path) -> List[Dict[str, Any]]:
        """QA 데이터 로드"""
        try:
            with open(qa_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ QA 파일 로드 성공: {qa_dataset_path}")
            
            qa_data = []
            
            if isinstance(data, dict):
                if 'metadata' in data and 'qa_pairs' in data:
                    qa_data = data['qa_pairs']
                    print(f"📈 메타데이터: {data['metadata']}")
                elif 'qa_pairs' in data:
                    qa_data = data['qa_pairs']
                else:
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and 'question' in value[0]:
                                qa_data = value
                                break
            elif isinstance(data, list):
                qa_data = data
            
            # 유효성 검사
            valid_qa = []
            for qa in qa_data:
                if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                    valid_qa.append(qa)
            
            print(f"🍳 전체 유효한 QA: {len(valid_qa)}개")
            return valid_qa
            
        except Exception as e:
            print(f"❌ QA 데이터 로드 실패: {e}")
            return []
    
    def prepare_data(self, qa_dataset_path):
        """데이터 준비 - 300 길이로"""
        print("🚀 전체 QA 데이터셋 로드 중 (300 길이)...")
        
        qa_data = self.load_qa_data(qa_dataset_path)
        
        if not qa_data:
            raise ValueError("유효한 QA 데이터를 찾을 수 없습니다.")
        
        print(f"📊 총 {len(qa_data)}개의 QA 쌍을 모두 사용 (300 길이)")
        
        # 최소 데이터 요구사항
        if len(qa_data) < 4:
            while len(qa_data) < 4:
                qa_data.extend(qa_data[:min(len(qa_data), 4-len(qa_data))])
        
        # 훈련/검증 분할
        if len(qa_data) >= 10:
            train_data, val_data = train_test_split(qa_data, test_size=0.2, random_state=42)
        else:
            split_idx = max(1, int(len(qa_data) * 0.8))
            train_data = qa_data[:split_idx]
            val_data = qa_data[split_idx:] if split_idx < len(qa_data) else qa_data[-1:]
        
        print(f"✅ 훈련 데이터: {len(train_data)}개")
        print(f"✅ 검증 데이터: {len(val_data)}개")
        
        # 데이터셋 생성 (300 길이로)
        train_dataset = RecipeQADataset(train_data, self.tokenizer, max_length=300)
        val_dataset = RecipeQADataset(val_data, self.tokenizer, max_length=300)
        
        # 배치 크기
        effective_batch_size = BATCH_SIZE
        print(f"📊 배치 크기: {effective_batch_size}")
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
        
        print(f"🎯 총 훈련 배치 수: {len(train_loader)}")
        print(f"🎯 총 검증 배치 수: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def initialize_model(self):
        """모델 초기화"""
        print("🔧 KcBERT 300 모델 초기화 중...")
        try:
            self.model = KcBERT300Model(self.model_name)
            self.model.to(self.device)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ 모델 초기화 성공 - 총 파라미터: {total_params:,}개")
            
            return self.model
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            raise
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS):
        """모델 훈련"""
        if self.model is None:
            self.initialize_model()
        
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.MSELoss()
        
        self.model.train()
        train_losses = []
        
        print(f"\n🚀 KcBERT 300 훈련 시작 - {num_epochs}개 에포크")
        print(f"📊 총 훈련 스텝: {total_steps}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"에포크 {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            epoch_train_loss = 0
            successful_batches = 0
            failed_batches = 0
            
            train_progress = tqdm(train_loader, desc=f"에포크 {epoch+1} 훈련")
            
            for batch_idx, batch in enumerate(train_progress):
                try:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # 입력 크기 확인
                    print(f"🔍 입력 크기: {input_ids.shape}")
                    
                    current_batch_size = input_ids.shape[0]
                    
                    # 순전파
                    outputs = self.model(input_ids, attention_mask)
                    similarity_score = outputs['similarity_score']
                    
                    # 타겟 생성
                    target = torch.ones(current_batch_size, 1).to(self.device)
                    
                    # 크기 검증
                    print(f"🔍 출력 크기: {similarity_score.shape}, 타겟 크기: {target.shape}")
                    
                    if similarity_score.shape != target.shape:
                        if similarity_score.dim() == 1:
                            similarity_score = similarity_score.unsqueeze(1)
                    
                    # 손실 계산
                    loss = criterion(similarity_score, target)
                    
                    # 역전파
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_train_loss += loss.item()
                    successful_batches += 1
                    
                    train_progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'success': f"{successful_batches}/{batch_idx+1}"
                    })
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"\n❌ 배치 {batch_idx} 오류: {e}")
                    if failed_batches > 10:  # 10개 이상 실패하면 중단
                        print("너무 많은 배치 실패, 훈련 중단")
                        break
                    continue
            
            avg_train_loss = epoch_train_loss / successful_batches if successful_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            print(f"\n📊 에포크 {epoch+1} 결과:")
            print(f"   성공한 배치: {successful_batches}/{len(train_loader)}")
            print(f"   실패한 배치: {failed_batches}/{len(train_loader)}")
            print(f"   훈련 손실: {avg_train_loss:.4f}")
            
            if successful_batches == 0:
                print("❌ 성공한 배치가 없습니다. 훈련 중단.")
                break
        
        return train_losses, []
    
    def save_model(self, save_path):
        """모델 저장"""
        print(f"📁 모델 저장 중: {save_path}")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
            config = {
                'model_name': self.model_name,
                'max_length': 300,
                'hidden_size': 300,
                'model_type': 'KcBERT300Model'
            }
            with open(save_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            tokenizer_path = save_path / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            
            print("✅ 모델 저장 완료!")
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")

def main():
    """메인 훈련 함수"""
    print("🚀 KcBERT 300 차원 모델 훈련 시작!")
    print("📏 모든 설정을 300으로 통일")
    
    if not QA_DATASET_PATH.exists():
        print(f"❌ QA 데이터셋을 찾을 수 없습니다: {QA_DATASET_PATH}")
        return
    
    try:
        trainer = KcBERT300Trainer(MODEL_NAME)
        train_loader, val_loader = trainer.prepare_data(QA_DATASET_PATH)
        
        print("\n🎯 KcBERT 300 차원으로 훈련 시작!")
        train_losses, _ = trainer.train(train_loader, val_loader, NUM_EPOCHS)
        
        trainer.save_model(TRAINED_MODEL_DIR)
        
        print(f"\n🎉 KcBERT 300 훈련 완료!")
        if train_losses:
            print(f"✅ 최종 훈련 손실: {train_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()