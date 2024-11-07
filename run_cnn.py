import os
from amapa_ar.config import MODELS_DIR, PROCESSED_DATA_DIR
from amapa_ar.pipelines.train_cnn_pipeline import TrainingCNNPipeline
from colorama import Fore, Style
def main():
    cnn_pipeline = TrainingCNNPipeline(
        data_path=os.path.join(PROCESSED_DATA_DIR, "train"),
        models_dir=MODELS_DIR
    ) 
    print(Fore.YELLOW +"🛠️ Iniciado treinamento da CNN " +Style.RESET_ALL )   
    cnn_pipeline.run()

if __name__ == "__main__":
    main()
