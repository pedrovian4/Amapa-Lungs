import os
import pandas as pd
from amapa_ar.config import MODELS_DIR, PROCESSED_DATA_DIR
from amapa_ar.data.extract_features import CNNFeatureExtractor, GLCMFeatureExtractor, ShapeFeatureExtractor
from amapa_ar.pipelines.feature_extract_pipeline import FeatureExtractorPipeline
from amapa_ar.pipelines.training_pipeline import TrainingPipeline
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
def main():
    data_path = os.path.join(PROCESSED_DATA_DIR, "extracted_features.csv")    
    if not os.path.exists(data_path):
        extractors = [GLCMFeatureExtractor(), ShapeFeatureExtractor(), CNNFeatureExtractor()]
        feature_pipeline = FeatureExtractorPipeline(extractors)
        print("Extraindo características das imagens...")
        df_features = feature_pipeline.run()
        print("Criando Diretório")        
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print("Criando salvando características")        
        df_features.to_csv(data_path, index=False)
        print(f"Características salvas em {data_path}")
    else:
        print("Arquivo de características já existe. Pulando extração.")
    print("Iniciando treinamento dos modelos...")
    pipeline = TrainingPipeline(data_path=data_path, models_dir=MODELS_DIR)
    pipeline.run()

if __name__ == "__main__":
    main()
