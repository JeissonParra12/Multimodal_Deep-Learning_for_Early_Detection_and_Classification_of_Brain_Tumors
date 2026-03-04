import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==========================================
# 1. CONFIGURATION (LOCAL PATH)
# ==========================================

CONFIG = {
    'BASE_PATH': '/Users/jeissonparra/Documents/Master_s Degree Florida International University/Data Science & AI/Spring - 2026/Capstone/Datasets/Balanced_Multimodal/CT',
    'INPUT_SHAPE': (224, 224, 4),      # From preprocessing
    'PATCH_SIZE': 28,                   # 224 / 8 = 28
    'NUM_PATCHES': 64,                   # 8 * 8 grid
    'NUM_CLASSES': 2,                    # Binary: Normal vs Tumor
    'BATCH_SIZE': 16,
    'EPOCHS': 2,
    'LEARNING_RATE': 0.0001,
    'FIGURE_PATH': './presentation_figures_ct',
    'MODEL_SAVE_PATH': './saved_models_ct'
}

CLASS_NAMES = ['Normal', 'Tumor']
COLORS = {
    'normal': '#2ECC71',
    'tumor': '#E74C3C',
    'ct_primary': '#FFB347',   # Orange
    'ct_secondary': '#A05195', # Purple
    'background': '#F8F9FA'
}

os.makedirs(CONFIG['FIGURE_PATH'], exist_ok=True)
os.makedirs(CONFIG['MODEL_SAVE_PATH'], exist_ok=True)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 3

# ==========================================
# 2. DATA GENERATOR (CT Specific)
# ==========================================

class CTDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, shuffle=True, input_shape=(224,224,4)):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.file_list = []
        self.labels = []
        self._scan_directory()
        self.on_epoch_end()

    def _scan_directory(self):
        for label_name in ['normal', 'tumor']:
            path = os.path.join(self.directory, label_name)
            if not os.path.exists(path):
                continue
            label_idx = 0 if label_name == 'normal' else 1
            for f in os.listdir(path):
                if f.endswith('.npy'):
                    self.file_list.append(os.path.join(path, f))
                    self.labels.append(label_idx)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        file_batch = [self.file_list[k] for k in indexes]
        label_batch = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(file_batch, label_batch)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_batch, label_batch):
        X = np.empty((self.batch_size, *self.input_shape), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, f_path in enumerate(file_batch):
            img = np.load(f_path).astype(np.float32)
            if img.shape != self.input_shape:
                img = tf.image.resize(img, (self.input_shape[0], self.input_shape[1])).numpy()
            X[i] = img
            y[i] = label_batch[i]

        return X, to_categorical(y, num_classes=CONFIG['NUM_CLASSES'])

# ==========================================
# 3. CORRELATION LEARNING ARCHITECTURE
# ==========================================

class CorrelationLayer(layers.Layer):
    def call(self, inputs):
        return tf.matmul(inputs, inputs, transpose_b=True)

def build_ct_correlation_model(input_shape):
    inputs = Input(shape=input_shape)

    # 1. Extract patches (8x8 grid, 28x28 each)
    patch_size = CONFIG['PATCH_SIZE']
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )   # shape: (batch, 8, 8, patch_size*patch_size*channels)

    patches_reshaped = layers.Reshape((CONFIG['NUM_PATCHES'], -1))(patches)
    patches_img = layers.Reshape((CONFIG['NUM_PATCHES'], patch_size, patch_size, input_shape[-1]))(patches_reshaped)

    # 2. Shared CNN encoder
    encoder = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(patch_size, patch_size, input_shape[-1])),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.LayerNormalization()
    ], name="shared_encoder")

    patch_features = layers.TimeDistributed(encoder)(patches_img)  # (batch, 64, 128)

    # 3. Correlation matrix
    correlation_matrix = CorrelationLayer(name="correlation_layer")(patch_features)  # (batch, 64, 64)

    # 4. Classification head
    flat_matrix = layers.Flatten()(correlation_matrix)  # 4096
    x = layers.Dense(256, activation='relu')(flat_matrix)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(CONFIG['NUM_CLASSES'], activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CT_Correlation_Model")
    return model

# ==========================================
# 4. PUBLICATION-QUALITY VISUALIZATIONS
# ==========================================

def create_ct_training_curves(history):
    """Generate training history plots for CT classifier (Slide 13 style)"""
    epochs = np.arange(1, len(history.history['accuracy']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['background'])

    # Accuracy
    train_acc = np.array(history.history['accuracy']) * 100
    val_acc = np.array(history.history['val_accuracy']) * 100
    ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=3)
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=3)
    ax1.fill_between(epochs, train_acc, val_acc, alpha=0.1, color='gray')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('CT Correlation Model Training Accuracy', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 101])
    ax1.axhline(y=val_acc[-1], color='green', linestyle='--', alpha=0.5,
                label=f'Final Val: {val_acc[-1]:.1f}%')
    ax1.text(epochs[-1]-5, val_acc[-1]+1, f'{val_acc[-1]:.1f}%',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=3)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=3)
    ax2.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('CT Correlation Model Training Loss', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(train_loss+val_loss)*1.1])
    ax2.text(epochs[-1]-5, val_loss[-1]+0.02, f'{val_loss[-1]:.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_training_curves.pdf'), bbox_inches='tight')
    plt.show()
    print("✓ CT training curves saved.")

def create_ct_confusion_matrix(y_true, y_pred, class_names):
    """Generate confusion matrix with metrics (Slide 14 style)"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['background'])

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('CT Confusion Matrix - Raw Counts', fontweight='bold')

    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('CT Confusion Matrix - Percentages', fontweight='bold')

    # Metrics
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if cm[1,1]+cm[1,0] > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if cm[0,0]+cm[0,1] > 0 else 0

    textstr = f'Accuracy: {accuracy*100:.1f}%\nSensitivity: {sensitivity*100:.1f}%\nSpecificity: {specificity*100:.1f}%'
    ax2.text(1.5, 1.5, textstr, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_confusion_matrix.pdf'), bbox_inches='tight')
    plt.show()
    print("✓ CT confusion matrix saved.")
    return accuracy, sensitivity, specificity

def create_ct_roc_curve(y_true, y_pred_proba, class_names):
    """Generate ROC curve with inset (Slide 14 style)"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])

    ax.plot(fpr, tpr, color=COLORS['ct_primary'], lw=3,
            label=f'CT ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['ct_primary'])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('CT ROC Curve', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Inset for low FPR region
    axins = inset_axes(ax, width="30%", height="30%", loc='center right',
                       bbox_to_anchor=(0.1, 0.1, 0.8, 0.8),
                       bbox_transform=ax.transAxes)
    mask = fpr <= 0.2
    axins.plot(fpr[mask], tpr[mask], color=COLORS['ct_primary'], lw=2)
    axins.plot([0, 0.2], [0.95, 0.95], 'k--', alpha=0.3)
    axins.plot([0.2, 0.2], [0, 1], 'k--', alpha=0.3)
    axins.set_xlim([0, 0.2])
    axins.set_ylim([0.8, 1.0])
    axins.grid(True, alpha=0.3)
    axins.set_xlabel('FPR (0-0.2)', fontsize=9)
    axins.set_ylabel('TPR (0.8-1.0)', fontsize=9)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="gray", linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'ct_roc_curve.pdf'), bbox_inches='tight')
    plt.show()
    print("✓ CT ROC curve saved.")
    return roc_auc

def create_comparison_chart(mri_scores, ct_scores):
    """Generate side-by-side comparison bar chart (Slide 15 style)"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS['background'])

    bars1 = ax.bar(x - width/2, mri_scores, width, label='MRI Classifier',
                   color='#1f77b4', edgecolor='black', linewidth=1, alpha=0.8)
    bars2 = ax.bar(x + width/2, ct_scores, width, label='CT Classifier',
                   color='#ff7f0e', edgecolor='black', linewidth=1, alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)')
    ax.set_xlabel('Metrics')
    ax.set_title('Unimodal Models Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([90, 102])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG['FIGURE_PATH'], 'model_comparison.pdf'), bbox_inches='tight')
    plt.show()
    print("✓ Comparison chart saved.")

# ==========================================
# 5. EXECUTION & TRAINING
# ==========================================

def run_ct_classification():
    print("=" * 60)
    print("CT CORRELATION LEARNING MODEL")
    print("=" * 60)

    # Check local path
    if not os.path.exists(CONFIG['BASE_PATH']):
        print(f"❌ Error: Path not found {CONFIG['BASE_PATH']}")
        return

    # Generators
    train_gen = CTDataGenerator(os.path.join(CONFIG['BASE_PATH'], 'train'),
                                 batch_size=CONFIG['BATCH_SIZE'],
                                 input_shape=CONFIG['INPUT_SHAPE'])
    val_gen = CTDataGenerator(os.path.join(CONFIG['BASE_PATH'], 'val'),
                              batch_size=CONFIG['BATCH_SIZE'],
                              shuffle=False,
                              input_shape=CONFIG['INPUT_SHAPE'])
    test_gen = CTDataGenerator(os.path.join(CONFIG['BASE_PATH'], 'test'),
                               batch_size=CONFIG['BATCH_SIZE'],
                               shuffle=False,
                               input_shape=CONFIG['INPUT_SHAPE'])

    print(f"Training samples: {len(train_gen.file_list)}")
    print(f"Validation samples: {len(val_gen.file_list)}")
    print(f"Test samples: {len(test_gen.file_list)}")

    # Build model
    model = build_ct_correlation_model(CONFIG['INPUT_SHAPE'])
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Train
    history = model.fit(
        train_gen,
        epochs=CONFIG['EPOCHS'],
        validation_data=val_gen,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )

    # Save model
    model.save(os.path.join(CONFIG['MODEL_SAVE_PATH'], 'ct_correlation_model.keras'))

    # Generate training curves
    create_ct_training_curves(history)

    # Evaluate on test set
    y_true, y_pred_probs = [], []
    for i in range(len(test_gen)):
        X, y = test_gen[i]
        preds = model.predict_on_batch(X)
        y_true.extend(np.argmax(y, axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    acc, sens, spec = create_ct_confusion_matrix(y_true, y_pred, CLASS_NAMES)

    # ROC curve
    roc_auc = create_ct_roc_curve(y_true, y_pred_probs, CLASS_NAMES)

    # For the comparison chart, you'll need MRI scores from your MRI classifier.
    # Replace these with your actual MRI results.
    mri_scores = [99.6, 99.6, 99.6, 99.5, 99.9]   # Example from your slides
    ct_scores = [acc*100, sens*100, spec*100, (2*sens*spec/(sens+spec))*100 if sens+spec>0 else 0, roc_auc*100]
    create_comparison_chart(mri_scores, ct_scores)

    print("\n" + "=" * 60)
    print("✅ CT PIPELINE COMPLETED")
    print(f"📁 Figures saved in: {CONFIG['FIGURE_PATH']}")
    print("=" * 60)

if __name__ == "__main__":
    run_ct_classification()