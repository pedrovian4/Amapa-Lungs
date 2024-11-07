from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from colorama import Fore, Style

from amapa_ar.config import RANDOM_STATE, TEST_SIZE


class FeatureEngineeringModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        """Treina o modelo com os dados de treino."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Realiza predições com os dados de teste."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo e imprime métricas de desempenho.

        Parâmetros:
        - X_test: pd.DataFrame - Conjunto de dados de teste.
        - y_test: pd.Series - Rótulos do conjunto de teste.
        """
        print(Fore.BLUE + "🔍 Avaliando modelo..." + Style.RESET_ALL)
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(Fore.GREEN + f"✅ Accuracy: {accuracy:.4f}" + Style.RESET_ALL)
        print(Fore.CYAN + f"📊 Classification Report:\n{report}" + Style.RESET_ALL)


class RandomForestModel(FeatureEngineeringModel):
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        print(Fore.YELLOW + "🌟 Treinando RandomForest..." + Style.RESET_ALL)
        self.model.fit(X_train, y_train)
        print(Fore.GREEN + "✅ RandomForest treinado com sucesso." + Style.RESET_ALL)

    def predict(self, X_test):
        return self.model.predict(X_test)


class SVMModel(FeatureEngineeringModel):
    def __init__(self, kernel='linear', probability=True):
        super().__init__()
        self.model = SVC(kernel=kernel, probability=probability)

    def train(self, X_train, y_train):
        print(Fore.YELLOW + "🌟 Treinando SVM..." + Style.RESET_ALL)
        self.model.fit(X_train, y_train)
        print(Fore.GREEN + "✅ SVM treinado com sucesso." + Style.RESET_ALL)

    def predict(self, X_test):
        return self.model.predict(X_test)


class XGBoostModel(FeatureEngineeringModel):
    def __init__(self, random_state=42, use_label_encoder=False, eval_metric="mlogloss"):
        super().__init__()
        self.model = XGBClassifier(use_label_encoder=use_label_encoder, eval_metric=eval_metric, random_state=random_state)

    def train(self, X_train, y_train):
        print(Fore.YELLOW + "🌟 Treinando XGBoost..." + Style.RESET_ALL)
        self.model.fit(X_train, y_train)
        print(Fore.GREEN + "✅ XGBoost treinado com sucesso." + Style.RESET_ALL)

    def predict(self, X_test):
        return self.model.predict(X_test)


class FeatureEngineeringPipeline:
    def __init__(self, data_path):
        """
        Inicializa a classe com o caminho para o dataset.

        Parâmetros:
        - data_path (str): Caminho para o arquivo CSV contendo as características extraídas.
        """
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """Carrega os dados de características e separa rótulos e características."""
        print(Fore.BLUE + "🔄 Carregando os dados de características..." + Style.RESET_ALL)
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=["label"])
        y = data["label"]
        class_mapping = {label: idx for idx, label in enumerate(data["label"].unique())}
        y = y.map(class_mapping)
        print(Fore.MAGENTA + "🗓️ Dividindo os dados em conjuntos de treino e teste..." + Style.RESET_ALL)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(Fore.GREEN + "✅ Dados carregados e divididos com sucesso." + Style.RESET_ALL)
        return X_train, X_test, y_train, y_test

    def run_model(self, model):
        """
        Executa o treinamento e avaliação do modelo.

        Parâmetros:
        - model (FeatureEngineeringModel): Instância de um modelo de Feature Engineering.
        """
        print(Fore.CYAN + f"\n🔄 Iniciando treinamento e avaliação para o modelo {model.__class__.__name__}..." + Style.RESET_ALL)
        X_train, X_test, y_train, y_test = self.load_data()
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)
