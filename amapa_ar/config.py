import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'test')
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'train')
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 8
IMAGE_SIZE = (224, 224)
EPOCHS = 2
LEARNING_RATE = 0.01
CLASS_NAMES = [
    "00 Anatomia Normal", 
    "01 Processos Inflamatórios Pulmonares (Pneumonia)", 
    "02 Maior Densidade (Derrame Pleural, Consolidação Atelectasica, Hidrotorax, Empiema)",
    "03 Menor Densidade (Pneumotorax, Pneumomediastino, Pneumoperitonio)",
    "04 Doenças Pulmonares Obstrutivas (Enfisema, Broncopneumonia, Bronquiectasia, Embolia)",
    "05 Doenças Infecciosas Degenerativas (Tuberculose, Sarcoidose, Proteinose, Fibrose)",
    "06 Lesões Encapsuladas (Abscessos, Nódulos, Cistos, Massas Tumorais, Metastases)",
    "07 Alterações de Mediastino (Pericardite, Malformações Arteriovenosas, Linfonodomegalias)",
    "08 Alterações do Tórax (Atelectasias, Malformações, Agenesia, Hipoplasias)"
]


CLASS_ALIASES = {
    "Normal": "00 Anatomia Normal",
    "Pneumonia": "01 Processos Inflamatórios Pulmonares (Pneumonia)",
    "HighDensity": "02 Maior Densidade (Derrame Pleural, Consolidação Atelectasica, Hidrotorax, Empiema)",
    "LowDensity": "03 Menor Densidade (Pneumotorax, Pneumomediastino, Pneumoperitonio)",
    "Obstructive": "04 Doenças Pulmonares Obstrutivas (Enfisema, Broncopneumonia, Bronquiectasia, Embolia)",
    "Infectious": "05 Doenças Infecciosas Degenerativas (Tuberculose, Sarcoidose, Proteinose, Fibrose)",
    "Encapsulated": "06 Lesões Encapsuladas (Abscessos, Nódulos, Cistos, Massas Tumorais, Metastases)",
    "Mediastinal": "07 Alterações de Mediastino (Pericardite, Malformações Arteriovenosas, Linfonodomegalias)",
    "Thoracic": "08 Alterações do Tórax (Atelectasias, Malformações, Agenesia, Hipoplasias)"
}