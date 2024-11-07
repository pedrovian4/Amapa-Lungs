import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from amapa_ar.config import IMAGE_SIZE, EPOCHS, LEARNING_RATE

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def prepare_test_generator(test_dir, batch_size=32):
    class_labels = [folder for folder in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, folder))]
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        classes=class_labels,
        class_mode='categorical',
        shuffle=False
    )
    return test_gen

def evaluate_model(model, test_gen):
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    return loss, accuracy

def get_predictions(model, test_gen):
    Y_pred = model.predict(test_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    return y_true, y_pred, class_labels

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.title('Matriz de Confusão')
    plt.show()

def print_classification_report(y_true, y_pred, class_labels):
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print('Relatório de Classificação')
    print(report)

def calculate_additional_metrics(y_true, y_pred, class_labels):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_labels)))
    
    print("\nMétricas adicionais por classe:")
    for i, label in enumerate(class_labels):
        print(f"Classe: {label}")
        print(f"  Precisão: {precision[i]:.2f}")
        print(f"  Revocação: {recall[i]:.2f}")
        print(f"  F1-Score: {f1[i]:.2f}")
        print(f"  Suporte: {support[i]}\n")
    
    accuracy = np.mean(y_true == y_pred)
    print(f"Acurácia Global: {accuracy:.2f}")

def plot_roc_curve(y_true, Y_pred, class_labels):
    y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
    Y_pred_bin = label_binarize(np.argmax(Y_pred, axis=1), classes=range(len(class_labels)))

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], Y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Classe {label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Linha Aleatória (AUC = 0.5)")
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curvas ROC por Classe")
    plt.legend(loc="lower right")
    plt.show()

def visualize_predictions(test_gen, y_true, y_pred, class_labels, num_images=9):
    indices = random.sample(range(len(test_gen.filenames)), num_images)
    plt.figure(figsize=(15,15))
    for i, idx in enumerate(indices):
        img_path = test_gen.filepaths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        
        plt.subplot(3, 3, i+1)
        plt.imshow(image_resized)
        plt.axis('off')
        
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f'Real: {true_label}\nPred: {pred_label}', color=color)
    plt.tight_layout()
    plt.show()
