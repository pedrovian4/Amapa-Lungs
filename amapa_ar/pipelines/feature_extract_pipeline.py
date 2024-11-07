import os
import cv2
import numpy as np
import pandas as pd
from amapa_ar.config import CLASS_NAMES, IMAGE_SIZE, TRAIN_DIR
from colorama import Fore
class FeatureExtractorPipeline:
    def __init__(self, extractors):
        self.extractors = extractors

    def extract_features_from_image(self, img_path, class_label):
        image = cv2.imread(img_path)
        if image is None:
            return None
        
        features = []
        for extractor in self.extractors:
            features.extend(extractor.extract(image))
        
        return [class_label] + features

    def run(self):
        data = []
        total_images = sum(len(os.listdir(os.path.join(TRAIN_DIR, class_label))) for class_label in CLASS_NAMES)
        processed_images = 0

        for class_label in CLASS_NAMES:
            class_dir = os.path.join(TRAIN_DIR, class_label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                print(Fore.BLUE +"🦖 Extraindo caracteristicas da imagem: ", img_name)
                result = self.extract_features_from_image(img_path, class_label)
                if result is not None:
                    data.append(result)
                processed_images += 1
                print(Fore.GREEN + f"🎃 Processando imagem {processed_images}/{total_images} ({(processed_images / total_images) * 100:.2f}%) concluído.")

        columns = ["label"]
        for extractor in self.extractors:
            num_features = len(extractor.extract(np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)))
            columns.extend([f"{type(extractor).__name__}_{i}" for i in range(num_features)])

        df = pd.DataFrame(data, columns=columns)
        return df
