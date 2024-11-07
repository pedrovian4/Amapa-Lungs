import os
import pickle
from colorama import Fore, Style
from amapa_ar.models.feature_engineering import FeatureEngineeringPipeline, RandomForestModel, SVMModel, XGBoostModel
from amapa_ar.config import MODELS_DIR

class TrainingFeatureEngPipeline:
    """Pipeline para treinamento e avaliação de diferentes modelos de engenharia de características."""

    def __init__(self, data_path, models_dir=MODELS_DIR, test_size=0.2):
        self.data_path = data_path
        self.models_dir = models_dir
        self.test_size = test_size
        self.results = []
    

    def train_feature_engineering_models(self):
        """Treina modelos de engenharia de características (não CNNs) e os salva se não existirem."""
        pipeline = FeatureEngineeringPipeline(self.data_path)
        models = {
            "SVM": SVMModel(),
            "XGBoost": XGBoostModel(),
            "RandomForest": RandomForestModel(),
        }

        print(Fore.BLUE + "🔄 Iniciando treinamento dos modelos de engenharia de características..." + Style.RESET_ALL)

        for model_name, model in models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                print(Fore.GREEN + f"🌟 Treinando modelo {model_name}..." + Style.RESET_ALL)
                
                try:
                    pipeline.run_model(model)
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(Fore.GREEN + f"✅ Modelo {model_name} treinado e salvo em {model_path}." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"❌ Falha ao treinar o modelo {model_name}: {e}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"⚠️ Modelo {model_name} já existe. Pulando o treinamento." + Style.RESET_ALL)

    def run(self):
        """Executa o pipeline de treinamento para os modelos de engenharia de características."""
        self.train_feature_engineering_models()
