import os
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from amapa_ar.data.preprocessing import DataGenerator
from amapa_ar.models.cnn import EfficientNetB0Model, InceptionV3Model, ResNet50Model, VGG16Model
from amapa_ar.models.feature_engineering import FeatureEngineeringPipeline, RandomForestModel, SVMModel, XGBoostModel
from amapa_ar.config import MODELS_DIR, EPOCHS

class TrainingCNNPipeline:
    """Pipeline para treinamento e avaliação das CNN."""

    def __init__(self, data_path, models_dir=MODELS_DIR, test_size=0.2):
        self.data_path = data_path
        self.models_dir = models_dir
        self.test_size = test_size
        self.results = []
    

    def train_cnn_model(self, model_class, model_name):
        """Treina um modelo CNN específico e o salva se não existir."""
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            print(Fore.YELLOW + f"⚠️ Modelo {model_name} já existe. Pulando o treinamento." + Style.RESET_ALL)
            return

        print(Fore.BLUE + "🔄 Configurando geradores de dados para CNN..." + Style.RESET_ALL)
        data_gen = DataGenerator()
        train_generator = data_gen.get_train_generator()
        validation_generator = data_gen.get_validation_generator()

        print(Fore.GREEN + f"🚀 Iniciando treinamento da CNN ({model_name})..." + Style.RESET_ALL)
        model = model_class().get_model()

        for epoch in range(EPOCHS):
            print(Fore.MAGENTA + f"🗓️ Treinando Epoch {epoch + 1}/{EPOCHS}..." + Style.RESET_ALL)
            history = model.fit(train_generator, validation_data=validation_generator, epochs=1, verbose=1)
            val_accuracy = history.history['val_accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            print(Fore.YELLOW + f"🔹 Epoch {epoch + 1}/{EPOCHS} concluído: Val Accuracy={val_accuracy:.4f}, Val Loss={val_loss:.4f}" + Style.RESET_ALL)

        model.save(model_path)
        print(Fore.CYAN + f"💾 Modelo {model_name} salvo em {model_path}" + Style.RESET_ALL)

    def run(self):
        """Executa o pipeline de treinamento, incluindo modelos tradicionais e CNN."""
        cnn_models = {
            "VGG16": VGG16Model,
            "ResNet50": ResNet50Model,
            "InceptionV3": InceptionV3Model,
            "EfficientNetB0": EfficientNetB0Model
        }

        print(Fore.BLUE + "🔄 Iniciando treinamento dos modelos CNN..." + Style.RESET_ALL)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.train_cnn_model, model_class, model_name): model_name for model_name, model_class in cnn_models.items()}
            
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                    print(Fore.GREEN + f"✅ Modelo {model_name} treinado com sucesso." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"❌ Falha ao treinar o modelo {model_name}: {e}" + Style.RESET_ALL)

        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.models_dir, "training_results.csv")
        results_df.to_csv(results_path, index=False)
        print(Fore.CYAN + f"📊 Resultados salvos em {results_path}" + Style.RESET_ALL)
