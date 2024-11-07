import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from amapa_ar.config import (
    IMAGE_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CLASS_NAMES,
    MODELS_DIR,
    TEST_DIR,
    REPORTS_DIR,
    BATCH_SIZE,
)

class ModelEvaluator:
    def __init__(
        self,
        models_dir=MODELS_DIR,
        test_dir=TEST_DIR,
        reports_dir=REPORTS_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
    ):
        self.models_dir = models_dir
        self.test_dir = test_dir
        self.reports_dir = reports_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = class_names
        self.test_generator = None
        self.model = None
        self.y_true = None
        self.y_pred = None
        self.Y_pred = None
        self.results = {}
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_trained_model(self, model_name):
        model_path = os.path.join(self.models_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            print(f"Modelo {model_name} não encontrado em {model_path}")
            return None
        self.model = load_model(model_path)
        print(f"Modelo {model_name} carregado com sucesso.")
        return self.model

    def prepare_test_generator(self):
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = test_datagen.flow_from_directory(
            directory=self.test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def get_predictions(self):
        self.Y_pred = self.model.predict(self.test_generator, verbose=1)
        self.y_pred = np.argmax(self.Y_pred, axis=1)
        self.y_true = self.test_generator.classes
        self.class_labels = list(self.test_generator.class_indices.keys())

    def plot_confusion_matrix(self, model_name):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            cmap="Blues",
        )
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_fig_path = os.path.join(
            self.reports_dir, f"{model_name}_confusion_matrix.png"
        )
        plt.savefig(cm_fig_path)
        plt.close()
        print(f"Matriz de confusão salva em {cm_fig_path}")

    def print_classification_report(self, model_name):
        report = classification_report(
            self.y_true, self.y_pred, target_names=self.class_labels, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_csv_path = os.path.join(
            self.reports_dir, f"{model_name}_classification_report.csv"
        )
        report_df.to_csv(report_csv_path, index=True)
        print(f"Relatório de classificação salvo em {report_csv_path}")

    def calculate_additional_metrics(self):
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            average=None,
            labels=range(len(self.class_labels)),
        )

        print("\nMétricas adicionais por classe:")
        for i, label in enumerate(self.class_labels):
            print(f"Classe: {label}")
            print(f"  Precisão: {precision[i]:.2f}")
            print(f"  Revocação: {recall[i]:.2f}")
            print(f"  F1-Score: {f1[i]:.2f}")
            print(f"  Suporte: {support[i]}\n")

        accuracy = np.mean(self.y_true == self.y_pred)
        print(f"Acurácia Global: {accuracy:.2f}")

    def plot_roc_curve(self, model_name):
        y_true_bin = label_binarize(self.y_true, classes=range(len(self.class_labels)))
        n_classes = y_true_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Curva ROC micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), self.Y_pred.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"Curva ROC micro-average (AUC = {roc_auc['micro']:.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        for i, label in enumerate(self.class_labels):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=f"Classe {label} (AUC = {roc_auc[i]:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", label="Linha Aleatória (AUC = 0.5)")
        plt.xlabel("Taxa de Falsos Positivos (FPR)")
        plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
        plt.title(f"Curvas ROC por Classe - {model_name}")
        plt.legend(loc="lower right")
        roc_fig_path = os.path.join(self.reports_dir, f"{model_name}_roc_curve.png")
        plt.savefig(roc_fig_path)
        plt.close()
        print(f"Curva ROC salva em {roc_fig_path}")

    def visualize_predictions(self, model_name, num_images=9):
        indices = random.sample(range(len(self.test_generator.filenames)), num_images)
        plt.figure(figsize=(15, 15))
        for i, idx in enumerate(indices):
            img_path = self.test_generator.filepaths[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(
                image, (self.image_size[0], self.image_size[1])
            )

            plt.subplot(3, 3, i + 1)
            plt.imshow(image_resized)
            plt.axis("off")

            true_label = self.class_labels[self.y_true[idx]]
            pred_label = self.class_labels[self.y_pred[idx]]
            color = "green" if true_label == pred_label else "red"

            plt.title(f"Real: {true_label}\nPred: {pred_label}", color=color)
        plt.tight_layout()
        vis_fig_path = os.path.join(
            self.reports_dir, f"{model_name}_predictions.png"
        )
        plt.savefig(vis_fig_path)
        plt.close()
        print(f"Visualização de previsões salva em {vis_fig_path}")

    def evaluate_and_save(self, model_name):
        self.load_trained_model(model_name)
        if self.model is None:
            return
        self.prepare_test_generator()
        self.evaluate_model()
        self.get_predictions()
        self.print_classification_report(model_name)
        self.plot_confusion_matrix(model_name)
        self.calculate_additional_metrics()
        self.plot_roc_curve(model_name)
        self.visualize_predictions(model_name)

        report_df = pd.read_csv(
            os.path.join(self.reports_dir, f"{model_name}_classification_report.csv"),
            index_col=0,
        )
        self.results[model_name] = report_df

    def compare_models(self):
        metrics = ["precision", "recall", "f1-score", "support"]
        summary_df = pd.DataFrame()

        for model_name, report_df in self.results.items():
            report = report_df.iloc[:-3]
            avg_metrics = report[metrics].mean()
            avg_metrics["model"] = model_name
            summary_df = summary_df.append(avg_metrics, ignore_index=True)

        summary_df = summary_df.sort_values(by="f1-score", ascending=False)

        summary_csv_path = os.path.join(self.reports_dir, "models_comparison.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="model", y="f1-score", data=summary_df)
        plt.title("Comparação do F1-Score Médio por Modelo")
        plt.xlabel("Modelo")
        plt.ylabel("F1-Score Médio")
        plt.savefig(os.path.join(self.reports_dir, "models_comparison.png"))
        plt.close()

        print(f"Comparação dos modelos salva em {summary_csv_path}")
        print("Gráfico de comparação salvo em models_comparison.png")
        print(summary_df)

    def run(self, model_names):
        for model_name in model_names:
            print(f"\nIniciando avaliação para o modelo: {model_name}")
            self.evaluate_and_save(model_name)
        self.compare_models()


