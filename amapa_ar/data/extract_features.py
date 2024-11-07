import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from amapa_ar.config import IMAGE_SIZE, CLASS_NAMES, RAW_DATA_DIR


class FeatureExtractor:
    def extract(self, image):
        raise NotImplementedError("Subclasses devem implementar o método 'extract'.")


class GLCMFeatureExtractor(FeatureExtractor):
    def extract(self, image):
        """
        Extrai características de textura da imagem usando a Matriz de Co-ocorrência de Níveis de Cinza (GLCM).
        Retorna uma lista de cinco valores correspondentes às propriedades: contraste, dissimilaridade, 
        homogeneidade, energia e correlação.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return [contrast, dissimilarity, homogeneity, energy, correlation]


class ShapeFeatureExtractor(FeatureExtractor):
    def extract(self, image):
        """
        Extrai características de forma da imagem, como área e perímetro, a partir dos contornos.
        Retorna uma lista com dois valores: a área total e o perímetro total dos contornos detectados.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(cnt) for cnt in contours)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        return [area, perimeter]


class CNNFeatureExtractor(FeatureExtractor):
    def __init__(self):
        vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
        self.model = Model(inputs=vgg16.input, outputs=vgg16.output)

    def extract(self, image):
        """
        Extrai características de alto nível da imagem usando a rede neural convolucional VGG16 pré-treinada.
        Retorna um vetor de características da VGG16 achatado.
        """
        img_resized = cv2.resize(image, IMAGE_SIZE)
        img_array = np.expand_dims(img_resized, axis=0)
        features = self.model.predict(img_array).flatten()
        return features.tolist()


