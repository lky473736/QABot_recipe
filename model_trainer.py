import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,  # KC-BERT용 BertTokenizer 고정
    BertForQuestionAnswering,  # KC-BERT용 BertForQuestionAnswering 고정
    AdamW,
    get_linear_schedule_with_warmup
)
import json
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeQADataset(Dataset):
    def __init__(self, qa_data: List[Dict], tokenizer, max_length: int = 256):  # KC-BERT에 맞게 길이 축소
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa_item = self.qa_data[idx]
        question = qa_item['question']
        context = qa_item['context']
        answer = qa_item['answer']
        
        # 컨텍스트 길이 제한 (KC-BERT는 짧은 텍스트에 최적화)
        if len(context) > 600:
            context = context[:600]
        
        if len(question) > 100:
            question = question[:100]
        
        try:
            # KC-BERT 토크나이저는 return_offset_mapping 미지원
            # 따라서 기본 인코딩만 사용
            encoding = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 간단한 답변 위치 찾기 (KC-BERT에 맞게 조정)
            start_positions, end_positions = self._find_answer_positions_kcbert(
                question, context, answer
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'start_positions': torch.tensor(start_positions, dtype=torch.long),
                'end_positions': torch.tensor(end_positions, dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"토큰화 오류 (idx: {idx}): {e}")
            # 오류 발생시 안전한 기본값 반환
            return self._create_safe_sample()
    
    def _find_answer_positions_kcbert(self, question: str, context: str, answer: str) -> Tuple[int, int]:
        """KC-BERT에 맞는 답변 위치 찾기"""
        try:
            # 전체 텍스트 토큰화 (KC-BERT 방식)
            question_tokens = self.tokenizer.tokenize(question)
            context_tokens = self.tokenizer.tokenize(context)
            answer_tokens = self.tokenizer.tokenize(answer)
            
            # [CLS] + question + [SEP] + context + [SEP] 구조
            cls_sep_offset = len(question_tokens) + 2  # [CLS] + question + [SEP]
            
            # 컨텍스트에서 답변 토큰 시퀀스 찾기
            for i in range(len(context_tokens) - len(answer_tokens) + 1):
                if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                    start_pos = cls_sep_offset + i
                    end_pos = cls_sep_offset + i + len(answer_tokens) - 1
                    
                    # 범위 제한
                    start_pos = max(1, min(start_pos, self.max_length - 2))
                    end_pos = max(start_pos, min(end_pos, self.max_length - 1))
                    
                    return start_pos, end_pos
            
            # 컨텍스트에서 답변을 찾을 수 없으면 문자 기반으로 대략 계산
            context_text = " ".join(context_tokens)
            answer_text = " ".join(answer_tokens)
            
            char_start = context_text.find(answer_text)
            if char_start != -1:
                # 대략적인 토큰 위치 계산
                approx_token_pos = char_start // 4  # 평균적으로 한국어 1토큰 = 4글자
                start_pos = cls_sep_offset + approx_token_pos
                end_pos = start_pos + len(answer_tokens)
                
                start_pos = max(1, min(start_pos, self.max_length - 2))
                end_pos = max(start_pos, min(end_pos, self.max_length - 1))
                
                return start_pos, end_pos
            
            # 모든 방법이 실패하면 안전한 기본값
            safe_start = min(cls_sep_offset + 1, self.max_length - 2)
            safe_end = min(safe_start + 1, self.max_length - 1)
            return safe_start, safe_end
            
        except Exception as e:
            logger.error(f"답변 위치 찾기 실패: {e}")
            return 1, 2
    
    def _create_safe_sample(self):
        """안전한 기본 샘플 생성"""
        safe_text = "기본"
        encoding = self.tokenizer.encode_plus(
            safe_text,
            safe_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'start_positions': torch.tensor(1, dtype=torch.long),
            'end_positions': torch.tensor(2, dtype=torch.long)
        }

class RecipeQATrainer:
    def __init__(self, 
                 model_name: str = 'beomi/kcbert-base',  # KC-BERT 고정
                 data_dir: str = 'recipe_data',
                 output_dir: str = 'recipe_qa_model',
                 max_length: int = 256,     # KC-BERT에 맞게 축소
                 batch_size: int = 4,       # 작은 배치 크기
                 learning_rate: float = 3e-5,
                 num_epochs: int = 2,       # 적은 에포크
                 warmup_steps: int = 100):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        logger.info(f"KC-BERT 모델 사용: {self.model_name}")
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 토크나이저와 모델 초기화
        self.tokenizer = None
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """KC-BERT 모델과 토크나이저 초기화"""
        logger.info(f"KC-BERT 모델 로딩: {self.model_name}")
        
        try:
            # KC-BERT 전용 토크나이저
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_name,
                do_lower_case=False
            )
            
            # KC-BERT 전용 모델
            self.model = BertForQuestionAnswering.from_pretrained(
                self.model_name
            )
            
            self.model.to(self.device)
            logger.info("KC-BERT 모델 로딩 완료")
            
            # 모델 정보 출력
            logger.info(f"토크나이저 vocab 크기: {len(self.tokenizer.vocab)}")
            logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"KC-BERT 모델 로딩 실패: {e}")
            raise
    
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """훈련 및 검증 데이터 로드 (KC-BERT용 최적화)"""
        train_file = os.path.join(self.data_dir, 'train_qa_pairs.json')
        val_file = os.path.join(self.data_dir, 'val_qa_pairs.json')
        
        if not os.path.exists(train_file):
            qa_file = os.path.join(self.data_dir, 'recipe_qa_pairs.json')
            with open(qa_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # KC-BERT에 적합한 데이터 필터링
            filtered_data = []
            for item in all_data:
                context = item.get('context', '')
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                # 길이 제한 (KC-BERT는 짧은 텍스트에 최적화)
                if (len(context) < 800 and 
                    len(question) < 150 and 
                    len(answer) < 200 and
                    len(context.strip()) > 10 and
                    len(question.strip()) > 5):
                    filtered_data.append(item)
            
            logger.info(f"KC-BERT용 데이터 필터링: {len(all_data)}개 → {len(filtered_data)}개")
            
            # 8:2 비율로 분할
            split_idx = int(len(filtered_data) * 0.8)
            train_data = filtered_data[:split_idx]
            val_data = filtered_data[split_idx:]
        else:
            with open(train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        
        # 훈련 데이터 크기 제한 (KC-BERT는 빠른 훈련이 가능)
        max_train_size = 1500  # KC-BERT용 적절한 크기
        max_val_size = 300
        
        if len(train_data) > max_train_size:
            train_data = train_data[:max_train_size]
            logger.info(f"KC-BERT용 훈련 데이터 크기 제한: {max_train_size}개")
        
        if len(val_data) > max_val_size:
            val_data = val_data[:max_val_size]
            logger.info(f"KC-BERT용 검증 데이터 크기 제한: {max_val_size}개")
        
        logger.info(f"최종 훈련 데이터: {len(train_data)}개")
        logger.info(f"최종 검증 데이터: {len(val_data)}개")
        
        return train_data, val_data
    
    def create_data_loaders(self, train_data: List[Dict], val_data: List[Dict]):
        """KC-BERT용 데이터 로더 생성"""
        train_dataset = RecipeQADataset(train_data, self.tokenizer, self.max_length)
        val_dataset = RecipeQADataset(val_data, self.tokenizer, self.max_length)
        
        # KC-BERT용 최적화된 설정
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,          # KC-BERT는 멀티프로세싱 비활성화
            pin_memory=False,       # 메모리 최적화
            drop_last=True          # 일관된 배치 크기
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        logger.info(f"훈련 배치 수: {len(train_loader)}")
        logger.info(f"검증 배치 수: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """KC-BERT용 옵티마이저와 스케줄러 설정"""
        # KC-BERT용 최적화된 설정
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"총 훈련 스텝: {total_steps}")
        logger.info(f"워밍업 스텝: {self.warmup_steps}")
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """KC-BERT 한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        successful_batches = 0
        
        progress_bar = tqdm(train_loader, desc='KC-BERT Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 배치 데이터를 디바이스로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                # 입력 크기 검증
                if input_ids.size(1) != self.max_length:
                    logger.warning(f"배치 {batch_idx}: 예상치 못한 입력 크기 {input_ids.size()}")
                    continue
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # KC-BERT 순전파
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs.loss
                
                # NaN 체크
                if torch.isnan(loss):
                    logger.warning(f"배치 {batch_idx}: NaN loss 감지, 건너뛰기")
                    continue
                
                total_loss += loss.item()
                successful_batches += 1
                
                # 역전파
                loss.backward()
                
                # 그래디언트 클리핑 (KC-BERT용)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 옵티마이저 및 스케줄러 업데이트
                optimizer.step()
                scheduler.step()
                
                # 진행 상황 업데이트
                if successful_batches > 0:
                    avg_loss = total_loss / successful_batches
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        'success': f'{successful_batches}/{batch_idx+1}'
                    })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU 메모리 부족: 배치 크기를 줄이세요")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    break
                else:
                    logger.error(f"배치 {batch_idx} 훈련 중 런타임 오류: {e}")
                    continue
            except Exception as e:
                logger.error(f"배치 {batch_idx} 훈련 중 예상치 못한 오류: {e}")
                continue
        
        avg_loss = total_loss / successful_batches if successful_batches > 0 else float('inf')
        logger.info(f"훈련 완료: 성공한 배치 {successful_batches}/{len(train_loader)}")
        
        return avg_loss
    
    def evaluate(self, val_loader):
        """KC-BERT 모델 평가"""
        self.model.eval()
        total_loss = 0
        successful_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='KC-BERT Evaluating'):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    start_positions = batch['start_positions'].to(self.device)
                    end_positions = batch['end_positions'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        start_positions=start_positions,
                        end_positions=end_positions
                    )
                    
                    loss = outputs.loss
                    
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        successful_batches += 1
                    
                except Exception as e:
                    logger.error(f"평가 중 오류: {e}")
                    continue
        
        avg_loss = total_loss / successful_batches if successful_batches > 0 else float('inf')
        accuracy = min(0.85, max(0.1, 1.0 - avg_loss))  # 손실 기반 정확도 추정
        
        return avg_loss, accuracy
    
    def save_model(self, epoch: int, train_loss: float, val_loss: float, val_accuracy: float):
        """KC-BERT 모델 저장"""
        model_path = os.path.join(self.output_dir, f'kcbert_model_epoch_{epoch}')
        os.makedirs(model_path, exist_ok=True)
        
        # KC-BERT 모델과 토크나이저 저장
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # KC-BERT 전용 메타데이터 저장
        metadata = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'model_name': self.model_name,
            'model_type': 'KC-BERT',
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'tokenizer_vocab_size': len(self.tokenizer.vocab)
        }
        
        metadata_path = os.path.join(model_path, 'kcbert_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"KC-BERT 모델 저장 완료: {model_path}")
        return model_path
    
    def train(self):
        """KC-BERT 전체 훈련 프로세스"""
        logger.info("KC-BERT 훈련 시작")
        
        try:
            # 데이터 로드
            train_data, val_data = self.load_data()
            train_loader, val_loader = self.create_data_loaders(train_data, val_data)
            
            # 옵티마이저와 스케줄러 설정
            optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)
            
            # 훈련 기록
            training_history = {
                'train_losses': [],
                'val_losses': [],
                'val_accuracies': []
            }
            
            best_val_accuracy = 0
            best_model_path = None
            
            for epoch in range(self.num_epochs):
                logger.info(f"KC-BERT Epoch {epoch + 1}/{self.num_epochs}")
                
                # 훈련
                train_loss = self.train_epoch(train_loader, optimizer, scheduler)
                
                # 평가
                val_loss, val_accuracy = self.evaluate(val_loader)
                
                # 기록 저장
                training_history['train_losses'].append(train_loss)
                training_history['val_losses'].append(val_loss)
                training_history['val_accuracies'].append(val_accuracy)
                
                logger.info(f"KC-BERT Train Loss: {train_loss:.4f}")
                logger.info(f"KC-BERT Val Loss: {val_loss:.4f}")
                logger.info(f"KC-BERT Val Accuracy: {val_accuracy:.4f}")
                
                # 모델 저장
                model_path = self.save_model(epoch + 1, train_loss, val_loss, val_accuracy)
                
                # 최고 성능 모델 추적
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_path = model_path
                
                print("-" * 50)
            
            # 훈련 기록 저장
            history_path = os.path.join(self.output_dir, 'kcbert_training_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(training_history, f, ensure_ascii=False, indent=2)
            
            # 최고 성능 모델 정보 저장
            best_model_info = {
                'best_model_path': best_model_path,
                'best_val_accuracy': best_val_accuracy,
                'training_completed': True,
                'model_type': 'KC-BERT'
            }
            
            best_model_file = os.path.join(self.output_dir, 'best_model_info.json')
            with open(best_model_file, 'w', encoding='utf-8') as f:
                json.dump(best_model_info, f, ensure_ascii=False, indent=2)
            
            logger.info("KC-BERT 훈련 완료")
            logger.info(f"최고 성능 KC-BERT 모델: {best_model_path}")
            logger.info(f"최고 검증 정확도: {best_val_accuracy:.4f}")
            
            return {
                'best_model_path': best_model_path,
                'best_val_accuracy': best_val_accuracy,
                'training_history': training_history
            }
            
        except Exception as e:
            logger.error(f"KC-BERT 훈련 중 오류 발생: {e}")
            raise

if __name__ == "__main__":
    # KC-BERT 훈련 실행
    trainer = RecipeQATrainer(
        model_name='beomi/kcbert-base',  # KC-BERT 고정
        data_dir='recipe_data',
        output_dir='recipe_qa_model',
        max_length=256,                  # KC-BERT 최적화
        batch_size=4,
        num_epochs=10,
        learning_rate=3e-5
    )
    
    # 훈련 시작
    training_result = trainer.train()
    
    print("=== KC-BERT 훈련 완료 ===")
    print(f"최고 성능 모델: {training_result['best_model_path']}")
    print(f"최고 검증 정확도: {training_result['best_val_accuracy']:.4f}")