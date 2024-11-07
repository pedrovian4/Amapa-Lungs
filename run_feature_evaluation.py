from amapa_ar.config import MODELS_DIR
from amapa_ar.evaluation.evaluate_svm import SVMModelEvaluator


evaluator = SVMModelEvaluator(
    data_path="extracted_features.csv",
    models_dir=MODELS_DIR,
    test_size=0.2,
    random_state=42
)
evaluator.run()

