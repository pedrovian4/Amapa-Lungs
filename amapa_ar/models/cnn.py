from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB0  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from colorama import Fore, Style
from amapa_ar.config import IMAGE_SIZE, CLASS_NAMES


class BaseCNN:
    def __init__(self, input_shape=IMAGE_SIZE + (3,), num_classes=len(CLASS_NAMES), learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None

    def compile_model(self):
        print(Fore.BLUE + "🔧 Compilando o modelo CNN com configuração de otimização..." + Style.RESET_ALL)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_model(self):
        print(Fore.CYAN + "🚀 Obtendo modelo CNN configurado para treinamento..." + Style.RESET_ALL)
        self.compile_model()
        return self.model


class VGG16Model(BaseCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(Fore.YELLOW + "🔄 Inicializando VGG16 com pesos pré-treinados..." + Style.RESET_ALL)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.model = self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        print(Fore.GREEN + "⚙️ Adicionando camadas personalizadas ao modelo VGG16..." + Style.RESET_ALL)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=x)


class ResNet50Model(BaseCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(Fore.YELLOW + "🔄 Inicializando ResNet50 com pesos pré-treinados..." + Style.RESET_ALL)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.model = self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        print(Fore.GREEN + "⚙️ Adicionando camadas personalizadas ao modelo ResNet50..." + Style.RESET_ALL)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=x)


class InceptionV3Model(BaseCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(Fore.YELLOW + "🔄 Inicializando InceptionV3 com pesos pré-treinados..." + Style.RESET_ALL)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.model = self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        print(Fore.GREEN + "⚙️ Adicionando camadas personalizadas ao modelo InceptionV3..." + Style.RESET_ALL)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=x)


class EfficientNetB0Model(BaseCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(Fore.YELLOW + "🔄 Inicializando EfficientNetB0 com pesos pré-treinados..." + Style.RESET_ALL)
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.model = self._add_custom_layers(base_model)

    def _add_custom_layers(self, base_model):
        print(Fore.GREEN + "⚙️ Adicionando camadas personalizadas ao modelo EfficientNetB0..." + Style.RESET_ALL)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=x)


def get_model(model_name: str):
    models = {
        'vgg16': VGG16Model,
        'resnet50': ResNet50Model,
        'inceptionv3': InceptionV3Model,
        'efficientnetb0': EfficientNetB0Model,
    }
    if model_name in models:
        print(Fore.CYAN + f"🔍 Selecionando o modelo {model_name} para treinamento..." + Style.RESET_ALL)
        return models[model_name]().get_model()
    else:
        print(Fore.RED + f"❌ Modelo {model_name} não encontrado. Modelos disponíveis: {list(models.keys())}" + Style.RESET_ALL)
        raise ValueError(f"Modelo {model_name} não encontrado.")
