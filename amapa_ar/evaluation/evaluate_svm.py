import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
from amapa_ar.models.feature_engineering import FeatureEngineeringPipeline, SVMModel
from amapa_ar.config import MODELS_DIR

class SVMModelEvaluator:
    """Evaluator para modelos SVM que calcula a taxa de acurácia."""

    def __init__(self, data_path, models_dir=MODELS_DIR, test_size=0.2, random_state=42):
        """
        Inicializa o avaliador.

        :param data_path: Caminho para o arquivo CSV com os dados
        :param models_dir: Diretório onde os modelos estão armazenados
        :param test_size: Proporção dos dados para teste
        :param random_state: Estado aleatório para reprodutibilidade
        """
        self.model_path = os.path.join(models_dir, "SVM.pkl")
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.pipeline = FeatureEngineeringPipeline(self.data_path)
        self.results = {}

    def load_model(self):
        """Carrega o modelo a partir do arquivo .pkl."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"O modelo SVM não foi encontrado em {self.model_path}.")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(Fore.GREEN + f"✅ Modelo SVM carregado com sucesso." + Style.RESET_ALL)

    def load_data(self):
        """Carrega os dados do CSV."""
        try:
            data = pd.read_csv(self.data_path)
            print(Fore.GREEN + f"✅ Dados carregados de {self.data_path}." + Style.RESET_ALL)
            return data
        except Exception as e:
            print(Fore.RED + f"❌ Falha ao carregar os dados: {e}" + Style.RESET_ALL)
            raise e

    def preprocess_data(self, data):
        """Aplica a pipeline de engenharia de características."""
        try:
            X, y = self.pipeline.transform(data)
            print(Fore.GREEN + "✅ Dados pré-processados com sucesso." + Style.RESET_ALL)
            return X, y
        except Exception as e:
            print(Fore.RED + f"❌ Falha no pré-processamento dos dados: {e}" + Style.RESET_ALL)
            raise e

    def evaluate(self):
        """Avalia o modelo e calcula a acurácia."""
        # Carrega o modelo
        self.load_model()

        # Carrega e processa os dados
        data = self.load_data()
        X, y = self.preprocess_data(data)

        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(Fore.BLUE + f"🔄 Dados divididos em treino ({1 - self.test_size}) e teste ({self.test_size})." + Style.RESET_ALL)

        # Faz as previsões
        try:
            y_pred = self.model.predict(X_test)
            print(Fore.GREEN + "✅ Previsões realizadas com sucesso." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Falha ao fazer previsões: {e}" + Style.RESET_ALL)
            raise e

        # Calcula a acurácia
        acc = accuracy_score(y_test, y_pred)
        self.results['accuracy'] = acc
        print(Fore.BLUE + f"📊 Acurácia do modelo SVM: {acc * 100:.2f}%" + Style.RESET_ALL)
        return acc

    def run(self):
        """Executa todo o processo de avaliação."""
        print(Fore.BLUE + "🔍 Iniciando avaliação do modelo SVM..." + Style.RESET_ALL)
        accuracy = self.evaluate()
        return accuracy

