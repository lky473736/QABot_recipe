#!/usr/bin/env python3
"""
한국 전통요리 레시피 마스터 - 전체 파이프라인 실행 스크립트
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# 로컬 모듈들
from data_collector import KoreanRecipeDataCollector
from data_preprocessor import RecipeDataPreprocessor
from model_trainer import RecipeQATrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RecipeMasterPipeline:
    def __init__(self, 
                 data_dir: str = 'recipe_data',
                 model_dir: str = 'recipe_qa_model',
                 force_retrain: bool = False):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.force_retrain = force_retrain
        
        # 컴포넌트 초기화
        self.collector = KoreanRecipeDataCollector()
        self.preprocessor = RecipeDataPreprocessor(data_dir)
        self.trainer = None
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    
    def run_data_collection(self) -> Dict[str, Any]:
        """데이터 수집 단계"""
        logger.info("=" * 50)
        logger.info("1단계: 데이터 수집 시작")
        logger.info("=" * 50)
        
        # 이미 데이터가 있는지 확인
        mock_file = os.path.join(self.data_dir, 'mock_recipes.json')
        
        if os.path.exists(mock_file) and not self.force_retrain:
            logger.info("기존 데이터 파일 발견, 수집 단계 건너뛰기")
            with open(mock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"recipes_count": len(data), "status": "skipped"}
        
        # 데이터 수집 실행
        try:
            collected_data = self.collector.collect_all_data()
            
            total_recipes = sum(len(data) for data in collected_data.values())
            logger.info(f"데이터 수집 완료: 총 {total_recipes}개 레시피")
            
            return {
                "recipes_count": total_recipes,
                "data_sources": list(collected_data.keys()),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            raise
    
    def run_data_preprocessing(self) -> Dict[str, Any]:
        """데이터 전처리 단계"""
        logger.info("=" * 50)
        logger.info("2단계: 데이터 전처리 시작")
        logger.info("=" * 50)
        
        # 이미 전처리된 데이터가 있는지 확인
        qa_file = os.path.join(self.data_dir, 'recipe_qa_pairs.json')
        
        if os.path.exists(qa_file) and not self.force_retrain:
            logger.info("기존 전처리 데이터 발견, 전처리 단계 건너뛰기")
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            return {"qa_pairs_count": len(qa_data), "status": "skipped"}
        
        # 전처리 실행
        try:
            result = self.preprocessor.process_all()
            
            logger.info(f"데이터 전처리 완료:")
            logger.info(f"  - 처리된 레시피: {result['total_recipes']}개")
            logger.info(f"  - 생성된 QA 쌍: {result['total_qa_pairs']}개")
            logger.info(f"  - 카테고리별 분포: {result['statistics']['categories']}")
            
            return {
                "recipes_count": result['total_recipes'],
                "qa_pairs_count": result['total_qa_pairs'],
                "categories": result['statistics']['categories'],
                "file_paths": result['file_paths'],
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}")
            raise
    
    def run_model_training(self) -> Dict[str, Any]:
        """모델 훈련 단계"""
        logger.info("=" * 50)
        logger.info("3단계: 모델 훈련 시작")
        logger.info("=" * 50)
        
        # 이미 훈련된 모델이 있는지 확인
        best_model_file = os.path.join(self.model_dir, 'best_model_info.json')
        
        if os.path.exists(best_model_file) and not self.force_retrain:
            logger.info("기존 훈련 모델 발견, 훈련 단계 건너뛰기")
            with open(best_model_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return {
                "model_path": model_info['best_model_path'],
                "accuracy": model_info['best_val_accuracy'],
                "status": "skipped"
            }
        
        # 훈련 데이터 확인
        train_file = os.path.join(self.data_dir, 'train_qa_pairs.json')
        val_file = os.path.join(self.data_dir, 'val_qa_pairs.json')
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            raise FileNotFoundError("훈련 데이터가 없습니다. 전처리를 먼저 실행하세요.")
        
        # 트레이너 초기화
        self.trainer = RecipeQATrainer(
            model_name='beomi/kcbert-base',
            data_dir=self.data_dir,
            output_dir=self.model_dir,
            batch_size=4,  # 메모리에 맞게 조정
            num_epochs=3,
            learning_rate=3e-5
        )
        
        try:
            # 훈련 실행
            training_result = self.trainer.train()
            
            logger.info(f"모델 훈련 완료:")
            logger.info(f"  - 최고 성능 모델: {training_result['best_model_path']}")
            logger.info(f"  - 최고 검증 정확도: {training_result['best_val_accuracy']:.4f}")
            
            # 모델 테스트
            test_results = self.trainer.test_model()
            
            logger.info("모델 테스트 결과:")
            for i, result in enumerate(test_results):
                logger.info(f"  테스트 {i+1}: {result['confidence']:.3f}")
            
            return {
                "model_path": training_result['best_model_path'],
                "accuracy": training_result['best_val_accuracy'],
                "training_history": training_result['training_history'],
                "test_results": test_results,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"모델 훈련 실패: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        logger.info("한국 전통요리 레시피 마스터 파이프라인 시작")
        logger.info(f"데이터 디렉토리: {self.data_dir}")
        logger.info(f"모델 디렉토리: {self.model_dir}")
        logger.info(f"강제 재훈련: {self.force_retrain}")
        
        results = {}
        
        try:
            # 1. 데이터 수집
            results['data_collection'] = self.run_data_collection()
            
            # 2. 데이터 전처리
            results['data_preprocessing'] = self.run_data_preprocessing()
            
            # 3. 모델 훈련
            results['model_training'] = self.run_model_training()
            
            # 4. 최종 결과 저장
            pipeline_result = {
                "pipeline_completed": True,
                "timestamp": "2024-01-01T00:00:00",  # 실제로는 현재 시간
                "results": results,
                "total_recipes": results['data_preprocessing']['recipes_count'],
                "total_qa_pairs": results['data_preprocessing']['qa_pairs_count'],
                "model_accuracy": results['model_training']['accuracy'],
                "model_path": results['model_training']['model_path']
            }
            
            result_file = os.path.join(self.model_dir, 'pipeline_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_result, f, ensure_ascii=False, indent=2)
            
            logger.info("=" * 50)
            logger.info("파이프라인 완료!")
            logger.info("=" * 50)
            logger.info(f"총 레시피: {pipeline_result['total_recipes']}개")
            logger.info(f"총 QA 쌍: {pipeline_result['total_qa_pairs']}개")
            logger.info(f"모델 정확도: {pipeline_result['model_accuracy']:.4f}")
            logger.info(f"모델 경로: {pipeline_result['model_path']}")
            logger.info("Flask 앱을 실행하여 웹 서비스를 시작할 수 있습니다.")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {e}")
            raise
    
    def quick_setup(self) -> Dict[str, Any]:
        """빠른 설정 (데모용)"""
        logger.info("빠른 설정 모드 - 모의 데이터만 사용")
        
        try:
            # 1. 모의 데이터 생성
            mock_data = self.collector.collect_mock_data()
            
            # 2. 간단한 전처리
            result = self.preprocessor.process_all()
            
            logger.info("빠른 설정 완료:")
            logger.info(f"  - 모의 레시피: {len(mock_data)}개")
            logger.info(f"  - QA 쌍: {result['total_qa_pairs']}개")
            logger.info("기본 BERT 모델을 사용하여 웹 서비스를 시작할 수 있습니다.")
            
            return {
                "setup_type": "quick",
                "recipes_count": len(mock_data),
                "qa_pairs_count": result['total_qa_pairs'],
                "model_type": "base_bert"
            }
            
        except Exception as e:
            logger.error(f"빠른 설정 실패: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='한국 전통요리 레시피 마스터 파이프라인')
    parser.add_argument('--mode', choices=['full', 'quick', 'data-only', 'train-only'], 
                       default='quick', help='실행 모드')
    parser.add_argument('--data-dir', default='recipe_data', help='데이터 디렉토리')
    parser.add_argument('--model-dir', default='recipe_qa_model', help='모델 디렉토리')
    parser.add_argument('--force-retrain', action='store_true', help='강제 재훈련')
    parser.add_argument('--api-key-food', help='식약처 API 키')
    parser.add_argument('--api-key-rural', help='농촌진흥청 API 키')
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = RecipeMasterPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        force_retrain=args.force_retrain
    )
    
    # API 키 설정 (제공된 경우)
    if args.api_key_food:
        pipeline.collector.api_keys['food_safety'] = args.api_key_food
    if args.api_key_rural:
        pipeline.collector.api_keys['rural_dev'] = args.api_key_rural
    
    try:
        if args.mode == 'full':
            result = pipeline.run_full_pipeline()
        elif args.mode == 'quick':
            result = pipeline.quick_setup()
        elif args.mode == 'data-only':
            result = {
                'data_collection': pipeline.run_data_collection(),
                'data_preprocessing': pipeline.run_data_preprocessing()
            }
        elif args.mode == 'train-only':
            result = {'model_training': pipeline.run_model_training()}
        
        print("\n" + "="*50)
        print("실행 완료!")
        print("Flask 웹 서버를 시작하려면:")
        print("python app.py")
        print("="*50)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        return 1
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())