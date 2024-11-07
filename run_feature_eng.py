import os
from amapa_ar.config import MODELS_DIR, PROCESSED_DATA_DIR
from amapa_ar.pipelines.train_feature_eng import TrainingFeatureEngPipeline

data_path = os.path.join(PROCESSED_DATA_DIR, "extracted_features.csv")    

pipeline = TrainingFeatureEngPipeline(data_path, MODELS_DIR)

pipeline.run()

