import os, sys
from sensor.components.data_ingestion import DataIngestion

from sensor.exception import SensorException
from sensor.logger import logging

from sensor.entity.config_entity import TrainingPipelineConfig
from sensor.entity.config_entity import DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidationArtifact
from sensor.components.data_validation import DataValidation
from sensor.entity.config_entity import DataValidationConfig
from sensor.components.data_transformation import DataTransformation
from sensor.entity.config_entity import DataTransformationConfig
from sensor.entity.artifact_entity import DataTransformationArtifact
from sensor.components.model_trainer import ModelTrainer
from sensor.entity.artifact_entity import ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainingConfig
from sensor.components.model_evaluation import ModelEvaluation
from sensor.entity.artifact_entity import ModelEvaluationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.components.model_pusher import ModelPusher
from sensor.entity.artifact_entity import ModelPusherArtifact
from sensor.entity.config_entity import ModelPusherConfig
from sensor.cloud_storage.s3_syncher import S3Sync
from sensor.constant.s3_bucket import *
from sensor.constant.training_pipeline import *

class TrainPipeline:
    is_pipeline_running=False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

        self.data_ingestion_config=DataIngestionConfig()

        self.data_validation_config = DataValidationConfig()

        self.data_transformation_config = DataTransformationConfig()

        self.model_trainer_config = ModelTrainingConfig()

        self.model_eval_config = ModelEvaluationConfig()

        self.model_pusher_config = ModelPusherConfig()

        self.s3_sync = S3Sync()
        

    def start_data_ingestion(self)->DataIngestionArtifact: #this function should return train and test file path as mentioned in artifact
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )

            logging.info("Getting the data from mongodb")

            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
            logging.info("Got the train_set and test_set from mongodb")

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
                )
            
            return data_ingestion_artifact
            
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e

    
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    )-> DataTransformationArtifact:

        try:
            data_transformation = DataTransformation(
                data_validation_artifact, self.data_transformation_config
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_model_trainer(
            self, data_transformation_artifact: DataTransformationArtifact
        ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config = self.model_trainer_config ,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def start_model_evaluation(
        self, data_validation_artifact: DataValidationArtifact,
        model_training_artifact: ModelTrainerArtifact
        )-> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
                data_validation_artifact=data_validation_artifact,
                model_eval_config=self.model_eval_config,
                model_training_artifact=model_training_artifact
            )

            model_eval_arifact = model_evaluation.initiate_model_evaluation()

            return model_eval_arifact

        except Exception as e:
            raise SensorException(e, sys)

    
    def start_model_pusher(self,
    model_eval_artifact:ModelEvaluationArtifact
        )-> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                self.model_pusher_config, 
                model_eval_artifact
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact

        except  Exception as e:
            raise  SensorException(e,sys)


    def sync_artifact_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"

            self.s3_sync.sync_folder_to_s3(
                folder = self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_buket_url
                )

        except Exception as e:
            raise SensorException(e,sys)

            
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"

            self.s3_sync.sync_folder_to_s3(
                folder = SAVED_MODEL_DIR,
                aws_bucket_url=aws_buket_url
                )

        except Exception as e:
            raise SensorException(e,sys)


    def run_pipeline(self):

        TrainPipeline.is_pipeline_running=True

        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )

            model_training_artifact = self.start_model_trainer(
                data_transformation_artifact
            )

            model_eval_artifact = self.start_model_evaluation(
                data_validation_artifact,
                model_training_artifact
            )
            
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")

            model_pusher_artifact = self.start_model_pusher(
                model_eval_artifact
            )

            TrainPipeline.is_pipeline_running=False

            self.sync_artifact_dir_to_s3()

            self.sync_saved_model_dir_to_s3()
            
        except Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False
            raise SensorException(e, sys)
