from sklearn.ensemble import IsolationForest


class AnomalyDetectorIsolationForest:
    def __init__(self, n_estimators=100, contamination=0.1):
        """
        Inicializa o modelo Isolation Forest para detecção de anomalias.

        Parâmetros:
        - n_estimators (int): Número de árvores na floresta.
        - contamination (float): Proporção esperada de anomalias nos dados.
        """
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

    def train(self, feature_data):
        """
        Treina o modelo Isolation Forest usando os dados de características.

        Parâmetros:
        - feature_data: np.array - Características extraídas das imagens (como em `extracted_features.csv`).
        """
        self.model.fit(feature_data)

    def detect_anomaly(self, sample):
        """
        Detecta se uma amostra é anômala.

        Parâmetros:
        - sample: np.array - Amostra de características para verificar.
        
        Retorna:
        - bool - True se for anômala, False caso contrário.
        """
        prediction = self.model.predict(sample.reshape(1, -1))
        return prediction[0] == -1  # quando  é -1  é anômala
