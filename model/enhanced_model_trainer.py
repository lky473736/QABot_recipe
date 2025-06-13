"""
디버깅 강화된 레시피 챗봇 모델 훈련기
- 상세한 진행 상황 출력
- 메모리 사용량 모니터링
- 에러 상세 분석
"""
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import sys
import os
from typing import List, Dict, Any, Tuple
import random
import time
import psutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

def print_memory_usage():
    """메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"🔧 GPU 메모리: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    
    ram = psutil.virtual_memory()
    print(f"🔧 RAM 사용: {ram.percent:.1f}% ({ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB)")

def print_step(step_num, message, detail=""):
    """단계별 진행 상황 출력"""
    print(f"\n{'='*70}")
    print(f"📍 {step_num}: {message}")
    if detail:
        print(f"   {detail}")
    print(f"{'='*70}")
    print_memory_usage()
    time.sleep(1)

class DebugQADataset(Dataset):
    """디버깅이 강화된 QA 데이터셋"""
    
    def __init__(self, qa_data: List[Dict[str, Any]], tokenizer, max_length: int = 200):
        print_step("데이터셋 초기화", f"총 {len(qa_data)}개 QA 처리 중...")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("🔍 데이터 유효성 검사 중...")
        valid_qa = []
        for i, qa in enumerate(qa_data):
            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                question = str(qa['question']).strip()
                answer = str(qa['answer']).strip()
                
                if len(question) >= 3 and len(answer) >= 5 and len(answer) <= 500:
                    valid_qa.append(qa)
            
            if i % 1000 == 0:
                print(f"   처리 중: {i}/{len(qa_data)} ({i/len(qa_data)*100:.1f}%)")
        
        self.qa_data = valid_qa
        print(f"✅ 유효한 QA: {len(self.qa_data)}개 (원본의 {len(self.qa_data)/len(qa_data)*100:.1f}%)")
        
        # 데이터 통계
        question_lengths = [len(qa['question']) for qa in self.qa_data[:100]]
        answer_lengths = [len(qa['answer']) for qa in self.qa_data[:100]]
        
        print(f"📊 데이터 통계 (샘플 100개):")
        print(f"   평균 질문 길이: {np.mean(question_lengths):.1f}자")
        print(f"   평균 답변 길이: {np.mean(answer_lengths):.1f}자")
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        question = str(qa.get('question', '')).strip()
        answer = str(qa.get('answer', '')).strip()
        
        # 질문과 답변을 함께 인코딩
        try:
            encoding = self.tokenizer(
                question,
                answer,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
                'question': question,
                'answer': answer
            }
        except Exception as e:
            print(f"⚠️ 토크나이징 오류 (idx {idx}): {e}")
            # 기본값 반환
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
                'question': question,
                'answer': answer
            }

class DebugRecipeChatbotModel(nn.Module):
    """디버깅이 강화된 레시피 챗봇 모델"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base", hidden_dropout_prob: float = 0.1):
        super(DebugRecipeChatbotModel, self).__init__()
        
        print_step("모델 초기화", f"모델: {model_name}")
        
        try:
            print("📥 설정 로드 중...")
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.hidden_dropout_prob = hidden_dropout_prob
            print(f"✅ 설정 로드 완료 - Hidden size: {self.config.hidden_size}")
            
            print("📥 BERT 모델 로드 중...")
            self.bert = AutoModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True)
            print("✅ BERT 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🔄 더 작은 모델로 폴백...")
            try:
                self.config = AutoConfig.from_pretrained("klue/bert-base")
                self.bert = AutoModel.from_pretrained("klue/bert-base")
                print("✅ 폴백 모델 로드 완료")
            except Exception as e2:
                print(f"❌ 폴백 모델도 실패: {e2}")
                raise
        
        # 모델 차원
        self.hidden_size = self.bert.config.hidden_size
        print(f"📏 Hidden size: {self.hidden_size}")
        
        # QA 매칭을 위한 헤드
        print("🔧 분류 헤드 초기화...")
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
        try:
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
        except Exception as e:
            print(f"❌ 순전파 오류: {e}")
            raise

class DebugModelTrainer:
    """디버깅이 강화된 모델 훈련기"""
    
    def __init__(self, model_name: str = "beomi/kcbert-base"):
        print_step("훈련기 초기화", f"모델: {model_name}")
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 사용 디바이스: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   총 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # 토크나이저 로드
        try:
            print("📥 토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            print(f"✅ 토크나이저 로드 성공 - Vocab size: {len(self.tokenizer)}")
        except Exception as e:
            print(f"❌ 토크나이저 로드 실패: {e}")
            print("🔄 기본 토크나이저 사용...")
            self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
    
    def load_qa_data(self, qa_dataset_path: str) -> List[Dict[str, Any]]:
        """QA 데이터 로드"""
        print_step("QA 데이터 로드", f"파일: {qa_dataset_path}")
        
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(qa_dataset_path) / (1024 * 1024)
            print(f"📊 파일 크기: {file_size:.1f}MB")
            
            print("📂 JSON 파일 로드 중...")
            with open(qa_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("✅ JSON 파일 로드 완료")
            
            qa_data = []
            if isinstance(data, dict):
                if 'qa_pairs' in data:
                    qa_data = data['qa_pairs']
                    if 'metadata' in data:
                        metadata = data['metadata']
                        print(f"📈 메타데이터:")
                        for key, value in metadata.items():
                            print(f"   {key}: {value}")
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
            
            print(f"📊 로드된 QA 수: {len(qa_data)}개")
            
            # 데이터 샘플 확인
            if qa_data:
                sample = qa_data[0]
                print(f"📋 샘플 QA:")
                print(f"   질문: {sample.get('question', '')[:50]}...")
                print(f"   답변: {sample.get('answer', '')[:50]}...")
                print(f"   타입: {sample.get('type', 'N/A')}")
            
            return qa_data
            
        except Exception as e:
            print(f"❌ QA 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def prepare_data(self, qa_dataset_path: str, test_size: float = 0.2):
        """데이터 준비"""
        print_step("데이터 준비", "훈련/검증 데이터 분할")
        
        qa_data = self.load_qa_data(qa_dataset_path)
        
        if not qa_data:
            raise ValueError("유효한 QA 데이터를 찾을 수 없습니다.")
        
        print(f"🔀 데이터 셔플 중...")
        random.shuffle(qa_data)
        
        # 훈련/검증 분할
        if len(qa_data) >= 10:
            train_data, val_data = train_test_split(
                qa_data, 
                test_size=test_size, 
                random_state=42
            )
        else:
            split_idx = max(1, int(len(qa_data) * 0.8))
            train_data = qa_data[:split_idx]
            val_data = qa_data[split_idx:] if split_idx < len(qa_data) else qa_data[-1:]
        
        print(f"✅ 데이터 분할 완료:")
        print(f"   훈련 데이터: {len(train_data)}개")
        print(f"   검증 데이터: {len(val_data)}개")
        
        # 데이터셋 생성
        print("🔧 PyTorch 데이터셋 생성 중...")
        self.train_dataset = DebugQADataset(train_data, self.tokenizer, max_length=200)
        self.val_dataset = DebugQADataset(val_data, self.tokenizer, max_length=200)
        
        return self.train_dataset, self.val_dataset
    
    def initialize_model(self):
        """모델 초기화"""
        print_step("모델 초기화", "GPU 메모리로 이동")
        
        try:
            self.model = DebugRecipeChatbotModel(self.model_name)
            
            # 모델을 GPU로 이동
            print("🚀 모델을 GPU로 이동 중...")
            self.model.to(self.device)
            
            # 모델 파라미터 수 계산
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"✅ 모델 초기화 완료")
            print(f"   총 파라미터: {total_params:,}")
            print(f"   훈련 가능 파라미터: {trainable_params:,}")
            print(f"   모델 크기: {total_params * 4 / 1e9:.2f}GB")
            
            print_memory_usage()
            
            return self.model
            
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def train(self, num_epochs: int = 10, batch_size: int = 8, learning_rate: float = 5e-5):
        """모델 훈련"""
        print_step("훈련 시작", f"에포크: {num_epochs}, 배치: {batch_size}, 학습률: {learning_rate}")
        
        if self.model is None:
            self.initialize_model()
        
        if self.train_dataset is None:
            raise ValueError("훈련 데이터가 준비되지 않았습니다.")
        
        # 데이터로더 생성
        print("🔧 데이터로더 생성 중...")
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ 데이터로더 생성 완료:")
        print(f"   훈련 배치 수: {len(train_loader)}")
        print(f"   검증 배치 수: {len(val_loader)}")
        
        # 옵티마이저와 스케줄러
        print("🔧 옵티마이저 설정 중...")
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
        
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"✅ 훈련 설정 완료:")
        print(f"   총 스텝: {total_steps}")
        print(f"   웜업 스텝: {int(0.1 * total_steps)}")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print_step(f"에포크 {epoch + 1}/{num_epochs}", "훈련 및 검증")
            
            # 훈련 단계
            self.model.train()
            total_train_loss = 0
            successful_batches = 0
            
            print("🏃‍♂️ 훈련 중...")
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
                    
                    # 타겟 생성
                    batch_size_current = input_ids.size(0)
                    targets = torch.ones(batch_size_current, 1).to(self.device)
                    
                    # 손실 계산
                    qa_logits = outputs['qa_logits']
                    loss = criterion(qa_logits, targets)
                    
                    # 역전파
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_train_loss += loss.item()
                    successful_batches += 1
                    
                    train_progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                    })
                    
                    # 메모리 정리
                    if self.device.type == 'cuda' and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n❌ 배치 {batch_idx} 훈련 오류: {e}")
                    continue
            
            avg_train_loss = total_train_loss / successful_batches if successful_batches > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            print(f"✅ 훈련 완료 - 평균 손실: {avg_train_loss:.4f} (성공한 배치: {successful_batches}/{len(train_loader)})")
            
            # 검증 단계
            self.model.eval()
            total_val_loss = 0
            successful_val_batches = 0
            
            print("📊 검증 중...")
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
                        
                        batch_size_current = input_ids.size(0)
                        targets = torch.ones(batch_size_current, 1).to(self.device)
                        
                        qa_logits = outputs['qa_logits']
                        loss = criterion(qa_logits, targets)
                        
                        total_val_loss += loss.item()
                        successful_val_batches += 1
                        
                        val_progress.set_postfix({'val_loss': f"{loss.item():.4f}"})
                        
                    except Exception as e:
                        print(f"\n❌ 검증 오류: {e}")
                        continue
            
            avg_val_loss = total_val_loss / successful_val_batches if successful_val_batches > 0 else float('inf')
            val_losses.append(avg_val_loss)
            
            print(f"✅ 검증 완료 - 평균 손실: {avg_val_loss:.4f}")
            print_memory_usage()
            
            # 최고 성능 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"🎯 최고 성능 갱신! 체크포인트 저장...")
                self.save_checkpoint(epoch, avg_train_loss, avg_val_loss)
        
        print_step("훈련 완료", f"최종 훈련 손실: {train_losses[-1]:.4f}, 최종 검증 손실: {val_losses[-1]:.4f}")
        
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
        print(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def save_model(self, save_path):
        """최종 모델 저장"""
        print_step("모델 저장", f"경로: {save_path}")
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 모델 가중치 저장
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
            # 설정 저장
            config = {
                'model_name': self.model_name,
                'hidden_size': self.model.hidden_size,
                'model_type': 'DebugRecipeChatbotModel',
                'training_completed': True,
                'debug_version': True
            }
            
            with open(save_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 토크나이저 저장
            tokenizer_path = save_path / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_path)
            
            print("✅ 모델 저장 완료!")
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            raise

def main():
    """메인 훈련 함수 (디버깅 강화)"""
    print("🚀 디버깅 강화된 레시피 챗봇 모델 훈련 시작!")
    print("⚡ 진행 상황을 상세히 출력합니다...")
    
    print_step("환경 확인", "시스템 환경 및 라이브러리 확인")
    
    # 라이브러리 버전 확인
    print(f"🔧 PyTorch 버전: {torch.__version__}")
    try:
        import transformers
        print(f"🔧 Transformers 버전: {transformers.__version__}")
    except:
        print("❌ Transformers 라이브러리 없음")
    
    print_step("데이터 확인", "QA 데이터셋 존재 확인")
    
    if not QA_DATASET_PATH.exists():
        print(f"❌ QA 데이터셋을 찾을 수 없습니다: {QA_DATASET_PATH}")
        print("먼저 enhanced_qa_generator.py를 실행해주세요.")
        return
    
    # 파일 크기 확인
    file_size = QA_DATASET_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ QA 파일 크기: {file_size:.1f}MB")
    
    try:
        print_step("훈련기 초기화", "디버깅 강화 훈련기 생성")
        trainer = DebugModelTrainer(MODEL_NAME)
        
        print_step("데이터 준비", "QA 데이터 로드 및 전처리")
        trainer.prepare_data(QA_DATASET_PATH)
        
        print_step("모델 훈련", "실제 훈련 시작 (시간이 오래 걸릴 수 있음)")
        
        # 작은 설정으로 테스트 훈련
        train_losses, val_losses = trainer.train(
            num_epochs=1,    # 테스트용 1 에포크
            batch_size=2,    # 매우 작은 배치
            learning_rate=5e-5
        )
        
        print_step("모델 저장", "훈련된 모델 저장")
        trainer.save_model(TRAINED_MODEL_DIR)
        
        print_step("훈련 완료", "모든 단계 성공적으로 완료")
        print(f"✅ 최종 훈련 손실: {train_losses[-1]:.4f}")
        print(f"✅ 최종 검증 손실: {val_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 치명적 오류 발생:")
        print(f"   오류 메시지: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 문제 해결 방법:")
        print(f"1. 메모리 부족 시: batch_size를 1로 줄여보세요")
        print(f"2. 모델 로딩 실패 시: 인터넷 연결을 확인하세요")
        print(f"3. CUDA 오류 시: CPU 모드로 실행해보세요")
        print(f"4. 간단한 버전 시도: python model/simple_trainer.py")

if __name__ == "__main__":
    main()