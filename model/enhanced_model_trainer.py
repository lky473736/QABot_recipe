"""
개선된 레시피 챗봇 모델 훈련기
- KcBERT 기반 QA 모델 올바른 훈련
- 질문-답변 매칭 학습
- 임베딩 기반 유사도 학습
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    AdamW, AutoConfig
)
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import sys
import os
from typing import List, Dict, Any, Tuple
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
class QADataset(Dataset):
    """질문-답변 데이터셋"""
    
    def __init__(self, qa_data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"📊 QA 데이터셋 생성: {len(qa_data)}개 QA 쌍")
        
        # 데이터 유효성 검사
        valid_qa = []
        for qa in qa_data:
            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                valid_qa.append(qa)
        
        self.qa_data = valid_qa
        print(f"✅ 유효한 QA: {len(self.qa_data)}개")
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        question = str(qa.get('question', '')).strip()
        answer = str(qa.get('answer', '')).strip()
        
        # 질문과 답변을 함께 인코딩 (BERT 방식)
        encoding = self.tokenizer(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 질문만 인코딩 (검색용)
        question_encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'question_input_ids': question_encoding['input_ids'].flatten(),
            'question_attention_mask': question_encoding['attention_mask'].flatten(),
            'question': question,
            'answer': answer
        }

class EnhancedRecipeChatbotModel(nn.Module):
    """향상된 레시피 챗봇 모델"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base", hidden_dropout_prob: float = 0.1):
        super(EnhancedRecipeChatbotModel, self).__init__()
        
        print(f"🔧 모델 초기화: {model_name}")
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.max_position_embeddings = 512  # ➕ 추가 이 부분!

        self.bert = AutoModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)
        
        # 모델 차원
        self.hidden_size = self.bert.config.hidden_size
        print(f"📏 BERT hidden size: {self.hidden_size}")
        
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
        
        print("✅ 모델 초기화 완료")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """순전파"""
        # BERT 인코딩
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # [CLS] 토큰 임베딩 추출
        pooled_output = outputs.pooler_output
        
        # QA 매칭 점수
        qa_logits = self.qa_classifier(pooled_output)
        
        # 임베딩 생성
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

class EnhancedModelTrainer:
    """향상된 모델 훈련기"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 사용 디바이스: {self.device}")
        
        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            print(f"✅ 토크나이저 로드 성공")
        except Exception as e:
            print(f"❌ 토크나이저 로드 실패: {e}")
            raise
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
    
    def load_qa_data(self, qa_dataset_path: str) -> List[Dict[str, Any]]:
        """QA 데이터 로드"""
        try:
            with open(qa_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ QA 파일 로드 성공: {qa_dataset_path}")
            
            qa_data = []
            if isinstance(data, dict):
                if 'qa_pairs' in data:
                    qa_data = data['qa_pairs']
                    if 'metadata' in data:
                        print(f"📈 메타데이터: {data['metadata']}")
                    if 'statistics' in data:
                        print(f"📊 통계 정보 포함")
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
                    # 너무 긴 답변 필터링 (토큰 한계 고려)
                    question = str(qa['question']).strip()
                    answer = str(qa['answer']).strip()
                    
                    if len(question) >= 3 and len(answer) >= 5 and len(answer) <= 1000:
                        valid_qa.append(qa)
            
            print(f"🍳 전체 유효한 QA: {len(valid_qa)}개")
            return valid_qa
            
        except Exception as e:
            print(f"❌ QA 데이터 로드 실패: {e}")
            return []
    
    def prepare_data(self, qa_dataset_path: str, test_size: float = 0.2):
        """데이터 준비"""
        print("🚀 QA 데이터 준비 중...")
        
        qa_data = self.load_qa_data(qa_dataset_path)
        
        if not qa_data:
            raise ValueError("유효한 QA 데이터를 찾을 수 없습니다.")
        
        # 데이터 셔플
        random.shuffle(qa_data)
        
        # 훈련/검증 분할
        if len(qa_data) >= 10:
            train_data, val_data = train_test_split(
                qa_data, 
                test_size=test_size, 
                random_state=42            )
        else:
            # 작은 데이터셋의 경우
            split_idx = max(1, int(len(qa_data) * 0.8))
            train_data = qa_data[:split_idx]
            val_data = qa_data[split_idx:] if split_idx < len(qa_data) else qa_data[-1:]
        
        print(f"✅ 훈련 데이터: {len(train_data)}개")
        print(f"✅ 검증 데이터: {len(val_data)}개")
        
        # 데이터셋 생성
        self.train_dataset = QADataset(train_data, self.tokenizer, max_length=512)
        self.val_dataset = QADataset(val_data, self.tokenizer, max_length=512)
        
        return self.train_dataset, self.val_dataset
    
    def initialize_model(self):
        """모델 초기화"""
        print("🔧 모델 초기화 중...")
        
        try:
            self.model = EnhancedRecipeChatbotModel(self.model_name)
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
    
    def train(self, num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """모델 훈련"""
        if self.model is None:
            self.initialize_model()
        
        if self.train_dataset is None:
            raise ValueError("훈련 데이터가 준비되지 않았습니다.")
        
        # 데이터로더 생성
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows 호환성
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 옵티마이저와 스케줄러
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # 손실 함수
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"\n🚀 훈련 시작")
        print(f"   에포크: {num_epochs}")
        print(f"   배치 크기: {batch_size}")
        print(f"   학습률: {learning_rate}")
        print(f"   총 스텝: {total_steps}")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"에포크 {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 훈련 단계
            self.model.train()
            total_train_loss = 0
            
            train_progress = tqdm(train_loader, desc=f"훈련 에포크 {epoch+1}")
            
            for batch_idx, batch in enumerate(train_progress):
                try:
                    optimizer.zero_grad()
                    
                    # 데이터 GPU로 이동
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    
                    # 순전파
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                    # 타겟 생성 (QA 매칭은 모두 positive로 학습)
                    batch_size = input_ids.size(0)
                    targets = torch.ones(batch_size, 1).to(self.device)
                    
                    # 손실 계산
                    qa_logits = outputs['qa_logits']
                    loss = criterion(qa_logits, targets)
                    
                    # 역전파
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    total_train_loss += loss.item()
                    
                    train_progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                except Exception as e:
                    print(f"\n❌ 배치 {batch_idx} 훈련 오류: {e}")
                    continue
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 검증 단계
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"검증 에포크 {epoch+1}")
                
                for batch in val_progress:
                    try:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        token_type_ids = batch['token_type_ids'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                        
                        batch_size = input_ids.size(0)
                        targets = torch.ones(batch_size, 1).to(self.device)
                        
                        qa_logits = outputs['qa_logits']
                        loss = criterion(qa_logits, targets)
                        
                        total_val_loss += loss.item()
                        
                        val_progress.set_postfix({'val_loss': f"{loss.item():.4f}"})
                        
                    except Exception as e:
                        print(f"\n❌ 검증 오류: {e}")
                        continue
            
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            val_losses.append(avg_val_loss)
            
            # 에포크 결과 출력
            print(f"\n📊 에포크 {epoch + 1} 결과:")
            print(f"   훈련 손실: {avg_train_loss:.4f}")
            print(f"   검증 손실: {avg_val_loss:.4f}")
            
            # 최고 성능 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"   🎯 최고 성능 갱신! 모델 저장...")
                self.save_checkpoint(epoch, avg_train_loss, avg_val_loss)
        
        print(f"\n🎉 훈련 완료!")
        print(f"   최종 훈련 손실: {train_losses[-1]:.4f}")
        print(f"   최종 검증 손실: {val_losses[-1]:.4f}")
        print(f"   최고 검증 손실: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """체크포인트 저장"""
        checkpoint_path = TRAINED_MODEL_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def save_model(self, save_path):
        """최종 모델 저장"""
        print(f"💾 모델 저장 중: {save_path}")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 모델 가중치 저장
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
            # 설정 저장
            config = {
                'model_name': self.model_name,
                'hidden_size': self.model.hidden_size,
                'model_type': 'EnhancedRecipeChatbotModel',
                'training_completed': True
            }
            
            with open(save_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 토크나이저 저장
            tokenizer_path = save_path / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            
            print("✅ 모델 저장 완료!")
            print(f"   모델 가중치: pytorch_model.bin")
            print(f"   설정 파일: config.json")
            print(f"   토크나이저: tokenizer/")
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            raise

def main():
    """메인 훈련 함수"""
    print("🚀 개선된 레시피 챗봇 모델 훈련 시작!")
    
    if not QA_DATASET_PATH.exists():
        print(f"❌ QA 데이터셋을 찾을 수 없습니다: {QA_DATASET_PATH}")
        print("먼저 enhanced_qa_generator.py를 실행해주세요.")
        return
    
    try:
        # 훈련기 초기화
        trainer = EnhancedModelTrainer(MODEL_NAME)
        
        # 데이터 준비
        trainer.prepare_data(QA_DATASET_PATH)
        
        # 모델 훈련
        train_losses, val_losses = trainer.train(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        # 모델 저장
        trainer.save_model(TRAINED_MODEL_DIR)
        
        print(f"\n🎉 훈련 완료!")
        print(f"✅ 최종 훈련 손실: {train_losses[-1]:.4f}")
        print(f"✅ 최종 검증 손실: {val_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("beomi/kcbert-base")
    print(config.max_position_embeddings)  # 만약 300이면 512로 변경


    main()
