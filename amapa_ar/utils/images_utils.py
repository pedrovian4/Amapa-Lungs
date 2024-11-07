import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import RAW_DATA_DIR, CLASS_ALIASES

class ImageUtils:
    """Classe de utilitários para operações comuns com imagens."""

    @staticmethod
    def load_image(image_path, target_size=None, normalize=True):
        """
        Carrega uma imagem de um caminho especificado e a redimensiona.

        Parâmetros:
        - image_path (str): Caminho da imagem.
        - target_size (tuple): Tamanho alvo (largura, altura) para redimensionamento.
                               None para manter o tamanho original.
        - normalize (bool): Se True, normaliza os pixels para o intervalo [0, 1].

        Retorna:
        - np.ndarray: Imagem carregada e processada.
        """
        image = cv2.imread(image_path)
        if target_size:
            image = cv2.resize(image, target_size)
        if normalize:
            image = image / 255.0
        return image

    @staticmethod
    def show_image(image, title="Image"):
        """
        Exibe uma imagem usando Matplotlib.

        Parâmetros:
        - image (np.ndarray): Imagem a ser exibida.
        - title (str): Título da imagem.
        """
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def load_images_from_class(alias, target_size=None, normalize=True):
        """
        Carrega todas as imagens de uma classe específica usando um alias.

        Parâmetros:
        - alias (str): Alias da classe (ex. 'Normal', 'Pneumonia').
        - target_size (tuple): Tamanho alvo (largura, altura) para redimensionamento.
        - normalize (bool): Se True, normaliza as imagens para o intervalo [0, 1].

        Retorna:
        - list: Lista de imagens carregadas da classe especificada.
        """
        class_name = CLASS_ALIASES.get(alias)
        if class_name is None:
            raise ValueError(f"Alias '{alias}' não encontrado em CLASS_ALIASES.")

        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise ValueError(f"Diretório para a classe '{class_name}' não encontrado em '{RAW_DATA_DIR}'.")

        images = []
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = ImageUtils.load_image(img_path, target_size, normalize)
            images.append(image)
        return images
