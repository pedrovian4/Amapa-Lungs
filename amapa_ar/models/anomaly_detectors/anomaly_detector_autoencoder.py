import numpy as np
import cv2
from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from sklearn.metrics import mean_squared_error
from amapa_ar.config import IMAGE_SIZE, RAW_DATA_DIR
import os

class AnomalyDetectorAutoencoder:
    def __init__(self, input_shape=IMAGE_SIZE + (3,)):
        self.input_shape = input_shape
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        """
        Constrói um modelo de autoencoder simples para detecção de anomalias.
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, train_images, epochs=50, batch_size=32):
        """
        Treina o autoencoder usando imagens normais.
        
        Parâmetros:
        - train_images: np.array - Imagens de treinamento normais.
        - epochs: int - Número de épocas de treinamento.
        - batch_size: int - Tamanho do lote para treinamento.
        """
        self.autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=batch_size, shuffle=True)

    def detect_anomaly(self, image, threshold=0.01):
        """
        Detecta anomalias comparando a imagem com sua reconstrução.
        
        Parâmetros:
        - image: np.array - Imagem de entrada.
        - threshold: float - Limite de erro de reconstrução para detecção de anomalias.
        
        Retorna:
        - bool - True se for anômala, False caso contrário.
        """
        reconstructed = self.autoencoder.predict(np.expand_dims(image, axis=0))
        error = mean_squared_error(image.flatten(), reconstructed.flatten())
        return error > threshold


""" 
exemplo de como usar

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    return image / 255.0 

def load_normal_images():
    normal_images = []
    normal_class_dir = os.path.join(RAW_DATA_DIR, 'train', '00 Anatomia Normal')  
    for img_name in os.listdir(normal_class_dir):
        img_path = os.path.join(normal_class_dir, img_name)
        image = preprocess_image(img_path)
        normal_images.append(image)
    return np.array(normal_images)

 """
