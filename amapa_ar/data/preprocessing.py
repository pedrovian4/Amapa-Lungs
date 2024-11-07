import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from amapa_ar.config import RAW_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES


class DataGenerator:
    def __init__(self, raw_data_dir=RAW_DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_names=CLASS_NAMES):
        """
        Inicializa a classe DataGenerator com configurações para os geradores de dados de treinamento, validação e teste.

        Parâmetros:
        - raw_data_dir (str): Diretório de dados brutos onde as imagens estão armazenadas.
        - image_size (tuple): Tamanho das imagens para redimensionamento.
        - batch_size (int): Tamanho do lote para os geradores de dados.
        - class_names (List[str]): Lista com os nomes das classes para classificação.
        """
        self.raw_data_dir = raw_data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = class_names

        self.train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.test_datagen = ImageDataGenerator(rescale=1.0/255)

    def get_train_generator(self):
        """
        Cria e retorna o gerador de dados para o conjunto de treinamento.

        Retorna:
        - train_generator (DirectoryIterator): Iterador de dados para treinamento.
        """
        return self.train_datagen.flow_from_directory(
            directory=os.path.join(self.raw_data_dir, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.class_names,
            class_mode='categorical',
            subset='training'
        )

    def get_validation_generator(self):
        """
        Cria e retorna o gerador de dados para o conjunto de validação.

        Retorna:
        - validation_generator (DirectoryIterator): Iterador de dados para validação.
        """
        return self.train_datagen.flow_from_directory(
            directory=os.path.join(self.raw_data_dir, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.class_names,
            class_mode='categorical',
            subset='validation'
        )

    def get_test_generator(self):
        """
        Cria e retorna o gerador de dados para o conjunto de teste.

        Retorna:
        - test_generator (DirectoryIterator): Iterador de dados para teste.
        """
        return self.test_datagen.flow_from_directory(
            directory=os.path.join(self.raw_data_dir, 'test'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            classes=self.class_names,
            class_mode='categorical',
            shuffle=False
        )
