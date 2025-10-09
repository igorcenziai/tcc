import cv2
import numpy as np
from sklearn.utils import shuffle
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

DOWNLOAD_PATH = "./data/temp/"
ZIP_NAME = "fer2013.zip"
TEMP_ZIP_FILE_PATH = os.path.join(DOWNLOAD_PATH, ZIP_NAME)

def load_dataset(PATH, exclude_classes = []):
    """
    Retrieve FER-2013 Dataset.

    Args:
    PATH: [str]: The dataset's path

    Returns:
    (images, labels, label_map)
    images: an array with all the images of all subfolders
    labels: an array with the labels of the images (the name of the folder they are in)
    label_map: a dict with the name of the subfolders
    """
    CLASS_NAMES = sorted(os.listdir(PATH))
    for c in exclude_classes:
        CLASS_NAMES.remove(c)
    LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    images = []
    labels = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(PATH, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(LABEL_MAP[class_name])
    images = np.array(images)
    labels = np.array(labels)
    images, labels = shuffle(images, labels, random_state=42)
    return images, labels, LABEL_MAP

def download_dataset(DATA_PATH):
    """Download do dataset FER2013
    Args:
    DATA_PATH: [str]: The path of the folder to save the dataset
    """
    
    if os.path.exists(DATA_PATH) and os.listdir(DATA_PATH):
        print(f"✅ Dataset já existe em: {DATA_PATH}")
        print("Pulando download...")
        return
    
    print("Fazendo download do dataset...")
    
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    
    cmd = [
        "curl", 
        "-L", 
        "-o", TEMP_ZIP_FILE_PATH,
        "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Download concluído!")
    except subprocess.CalledProcessError:
        print("Erro no download. Certifique-se que o curl está instalado.")
        sys.exit(1)
    
    # Extrair ZIP
    print("Extraindo arquivos...")
    os.makedirs(DATA_PATH, exist_ok=True)
    
    with zipfile.ZipFile(TEMP_ZIP_FILE_PATH, "r") as zip_file:
        zip_file.extractall(DATA_PATH)
    
    # Remover arquivo temporário
    os.remove(TEMP_ZIP_FILE_PATH)
    print("Dataset pronto!")

def create_cnn_model(base_model, OUTPUT_SIZE):
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(OUTPUT_SIZE, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def show_train_loss_accuracy(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history['loss'], label='Loss Treino')
    axes[0].plot(history.history['val_loss'], label='Loss Validação')
    axes[0].set_title(f'Loss - {model_name}')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Acurácia Treino')
        axes[1].plot(history.history['val_accuracy'], label='Acurácia Validação')
        axes[1].set_title(f'Acurácia - {model_name}')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Acurácia')
        axes[1].legend()
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Acurácia não disponível', ha='center', va='center')

    plt.tight_layout()

    save_dir = os.path.join(model_name, "metrics", "training")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "accuracy_loss_plot_basic_completo.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



OUTPUT_CLASSES = 7
INPUT_SHAPE = (48, 48, 3)

def get_callbacks(model_name):
    return [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3),
    ModelCheckpoint(f'{model_name}/best_{model_name}_model_basic.keras', monitor='accuracy', save_best_only=True)
]

def show_metrics(y_true, y_pred, model_name, image_name, LABEL_MAP, is_test=False):
    classes_emocao = list(LABEL_MAP.keys())
    class_acc = {}

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Métricas do modelo: {model_name}")
    print("Accuracy Score:  ", round(acc, 4))
    print("Precision Score: ", round(prec, 4))
    print("Recall Score:    ", round(rec, 4))
    print("F1 Score:        ", round(f1, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    
    for i, emotion in enumerate(classes_emocao):
        idx = np.where(y_true == i)[0]
        correct = np.sum(y_pred[idx] == y_true[idx])
        class_acc[emotion] = correct / len(idx)
    
    print('\nAcurácia por classe:')
    for emotion, acc in class_acc.items():
        print(f'- {emotion}: {acc:.2%}')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_emocao, 
                yticklabels=classes_emocao)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')

    save_dir = os.path.join(model_name, "metrics", "simple")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{"test_" if is_test else "train_"}{image_name}.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def get_preds(X, model):
    y = model.predict(X)
    return np.argmax(y, axis=1)