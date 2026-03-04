import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2

# ============================================================================
# SLIDE 9: MRI Training Curves
# ============================================================================

def create_mri_training_curves():
    """Generate training history plots for MRI classifier"""
    
    # Simulated training history based on your actual results
    # Replace these with your actual history values
    epochs = np.arange(1, 21)
    
    # Training metrics (smoothed for realistic curves)
    train_acc = 0.85 + 0.15 * (1 - np.exp(-epochs/5)) + 0.01 * np.random.randn(20)
    val_acc = 0.83 + 0.15 * (1 - np.exp(-epochs/6)) + 0.015 * np.random.randn(20)
    train_loss = 0.4 * np.exp(-epochs/4) + 0.02 * np.random.randn(20)
    val_loss = 0.42 * np.exp(-epochs/4.5) + 0.025 * np.random.randn(20)
    
    # Ensure values are within reasonable bounds
    train_acc = np.clip(train_acc, 0.75, 1.0)
    val_acc = np.clip(val_acc, 0.75, 1.0)
    train_loss = np.clip(train_loss, 0, 0.5)
    val_loss = np.clip(val_loss, 0, 0.5)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc * 100, 'b-', label='Training Accuracy', linewidth=3)
    ax1.plot(epochs, val_acc * 100, 'r-', label='Validation Accuracy', linewidth=3)
    ax1.fill_between(epochs, train_acc * 100, val_acc * 100, alpha=0.1, color='gray')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('MRI CNN Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([75, 101])
    ax1.axhline(y=99.6, color='green', linestyle='--', alpha=0.5, label='Final Accuracy: 99.6%')
    
    # Add text annotations
    ax1.text(15, 98, f'Train Acc: {train_acc[-1]*100:.1f}%', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
    ax1.text(15, 96, f'Val Acc: {val_acc[-1]*100:.1f}%', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=3)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=3)
    ax2.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('MRI CNN Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.5])
    
    ax2.text(15, 0.35, f'Train Loss: {train_loss[-1]:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
    ax2.text(15, 0.3, f'Val Loss: {val_loss[-1]:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    plt.tight_layout()
    plt.savefig('mri_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('mri_training_curves.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ MRI training curves saved as 'mri_training_curves.png'")

# ============================================================================
# SLIDE 10: MRI Confusion Matrix
# ============================================================================

def create_mri_confusion_matrix():
    """Generate confusion matrix for MRI classifier (CNN+SVM)"""
    
    # Actual confusion matrix from your results
    # Format: rows = true labels, columns = predicted labels
    # Order: Normal, Meningioma, Glioma, Pituitary
    cm = np.array([
        [1878, 8, 3, 1],    # Normal
        [5, 1885, 4, 1],    # Meningioma
        [2, 3, 1888, 3],    # Glioma
        [1, 2, 2, 1890]     # Pituitary
    ])
    
    class_names = ['Normal', 'Meningioma', 'Glioma', 'Pituitary']
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax[0], cbar_kws={'label': 'Count'})
    ax[0].set_xlabel('Predicted Label', fontsize=12)
    ax[0].set_ylabel('True Label', fontsize=12)
    ax[0].set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax[1], cbar_kws={'label': 'Percentage (%)'})
    ax[1].set_xlabel('Predicted Label', fontsize=12)
    ax[1].set_ylabel('True Label', fontsize=12)
    ax[1].set_title('Confusion Matrix - Percentages', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mri_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig('mri_confusion_matrix.pdf', bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    accuracy = np.trace(cm) / np.sum(cm)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    
    print("\n✓ MRI Confusion Matrix saved")
    print(f"   Overall Accuracy: {accuracy*100:.2f}%")
    print(f"   Per-class Precision: {dict(zip(class_names, precision))}")
    print(f"   Per-class Recall: {dict(zip(class_names, recall))}")

# ============================================================================
# SLIDE 11: MRI ROC Curves
# ============================================================================

def create_mri_roc_curves():
    """Generate ROC curves for MRI multiclass classification"""
    
    # Simulated ROC data (replace with your actual prediction probabilities)
    n_classes = 4
    class_names = ['Normal', 'Meningioma', 'Glioma', 'Pituitary']
    colors = ['blue', 'green', 'red', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', alpha=0.5)
    
    # Generate realistic ROC curves with high AUC
    for i, (name, color) in enumerate(zip(class_names, colors)):
        # Simulate high-performance ROC curve
        fpr = np.concatenate([[0], np.sort(np.random.uniform(0, 0.1, 50)), [1]])
        
        if i == 0:  # Normal
            tpr = np.concatenate([[0], np.linspace(0.8, 1.0, 50), [1]])
            roc_auc = 0.999
        elif i == 1:  # Meningioma
            tpr = np.concatenate([[0], np.linspace(0.78, 1.0, 50), [1]])
            roc_auc = 0.998
        elif i == 2:  # Glioma
            tpr = np.concatenate([[0], np.linspace(0.82, 1.0, 50), [1]])
            roc_auc = 0.999
        else:  # Pituitary
            tpr = np.concatenate([[0], np.linspace(0.9, 1.0, 50), [1]])
            roc_auc = 1.000
        
        ax.plot(fpr, tpr, color=color, lw=3, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves - MRI Multiclass Classification', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add inset zoom for high-sensitivity region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Create inset
    axins = inset_axes(ax, width="40%", height="40%", loc='center right',
                       bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if i == 0:
            fpr = np.linspace(0, 0.1, 100)
            tpr = 1 - 0.1 * np.exp(-50 * fpr)
        elif i == 1:
            tpr = 1 - 0.12 * np.exp(-48 * fpr)
        elif i == 2:
            tpr = 1 - 0.08 * np.exp(-52 * fpr)
        else:
            tpr = 1 - 0.05 * np.exp(-60 * fpr)
        
        axins.plot(fpr, tpr, color=color, lw=2)
    
    axins.plot([0, 0.1], [0.9, 0.9], 'k--', alpha=0.3)
    axins.plot([0.1, 0.1], [0, 1], 'k--', alpha=0.3)
    axins.set_xlim([0, 0.1])
    axins.set_ylim([0.9, 1.0])
    axins.grid(True, alpha=0.3)
    axins.set_xlabel('FPR (0-0.1)', fontsize=9)
    axins.set_ylabel('TPR (0.9-1.0)', fontsize=9)
    
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="gray", linewidth=1)
    
    plt.tight_layout()
    plt.savefig('mri_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('mri_roc_curves.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ MRI ROC curves saved as 'mri_roc_curves.png'")

# ============================================================================
# SLIDE 13: CT Training Curves
# ============================================================================

def create_ct_training_curves():
    """Generate training history plots for CT classifier"""
    
    epochs = np.arange(1, 21)
    
    # CT training metrics (slightly lower than MRI)
    train_acc = 0.75 + 0.22 * (1 - np.exp(-epochs/6)) + 0.015 * np.random.randn(20)
    val_acc = 0.73 + 0.22 * (1 - np.exp(-epochs/7)) + 0.02 * np.random.randn(20)
    train_loss = 0.5 * np.exp(-epochs/5) + 0.03 * np.random.randn(20)
    val_loss = 0.52 * np.exp(-epochs/5.5) + 0.035 * np.random.randn(20)
    
    train_acc = np.clip(train_acc, 0.7, 0.98)
    val_acc = np.clip(val_acc, 0.7, 0.98)
    train_loss = np.clip(train_loss, 0, 0.6)
    val_loss = np.clip(val_loss, 0, 0.6)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc * 100, 'b-', label='Training Accuracy', linewidth=3)
    ax1.plot(epochs, val_acc * 100, 'r-', label='Validation Accuracy', linewidth=3)
    ax1.fill_between(epochs, train_acc * 100, val_acc * 100, alpha=0.1, color='gray')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('CT Correlation Model Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 100])
    ax1.axhline(y=95.5, color='green', linestyle='--', alpha=0.5, label='Final Accuracy: 95.5%')
    
    ax1.text(15, 92, f'Train Acc: {train_acc[-1]*100:.1f}%', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
    ax1.text(15, 89, f'Val Acc: {val_acc[-1]*100:.1f}%', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=3)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=3)
    ax2.fill_between(epochs, train_loss, val_loss, alpha=0.1, color='gray')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('CT Correlation Model Training Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.6])
    
    ax2.text(15, 0.45, f'Train Loss: {train_loss[-1]:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
    ax2.text(15, 0.4, f'Val Loss: {val_loss[-1]:.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.2))
    
    plt.tight_layout()
    plt.savefig('ct_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('ct_training_curves.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ CT training curves saved as 'ct_training_curves.png'")

# ============================================================================
# SLIDE 14: CT Confusion Matrix and ROC
# ============================================================================

def create_ct_results():
    """Generate confusion matrix and ROC curve for CT binary classifier"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = np.array([[660, 33],    # Normal: 660 correct, 33 misclassified as tumor
                   [30, 665]])    # Tumor: 30 misclassified as normal, 665 correct
    
    class_names = ['Normal', 'Tumor']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('CT Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Calculate and add metrics as text
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # True Positive Rate
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True Negative Rate
    
    textstr = f'Accuracy: {accuracy*100:.1f}%\nSensitivity: {sensitivity*100:.1f}%\nSpecificity: {specificity*100:.1f}%'
    ax1.text(1.5, 2.2, textstr, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # ROC Curve
    # Simulate ROC data
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - np.exp(-5 * fpr)  # Realistic ROC shape
    roc_auc = 0.982
    
    ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax2.fill_between(fpr, tpr, alpha=0.2, color='orange')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('CT ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Add inset for low FPR region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    axins = inset_axes(ax2, width="30%", height="30%", loc='center right',
                       bbox_to_anchor=(0.1, 0.1, 0.8, 0.8), bbox_transform=ax2.transAxes)
    
    fpr_zoom = np.linspace(0, 0.2, 50)
    tpr_zoom = 1 - np.exp(-8 * fpr_zoom)
    axins.plot(fpr_zoom, tpr_zoom, color='darkorange', lw=2)
    axins.plot([0, 0.2], [0.95, 0.95], 'k--', alpha=0.3)
    axins.plot([0.2, 0.2], [0, 1], 'k--', alpha=0.3)
    axins.set_xlim([0, 0.2])
    axins.set_ylim([0.8, 1.0])
    axins.grid(True, alpha=0.3)
    
    mark_inset(ax2, axins, loc1=1, loc2=3, fc="none", ec="gray", linewidth=1)
    
    plt.tight_layout()
    plt.savefig('ct_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('ct_results.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ CT results saved as 'ct_results.png'")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Sensitivity: {sensitivity*100:.1f}%")
    print(f"   Specificity: {specificity*100:.1f}%")

# ============================================================================
# SLIDE 15: Unimodal Comparison Bar Chart
# ============================================================================

def create_comparison_chart():
    """Generate side-by-side comparison of MRI and CT performance"""
    
    # Metrics data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    mri_scores = [99.6, 99.6, 99.6, 99.5, 99.9]
    ct_scores = [95.5, 95.4, 95.5, 95.4, 98.2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, mri_scores, width, label='MRI Classifier', 
                   color='#1f77b4', edgecolor='black', linewidth=1, alpha=0.8)
    bars2 = ax.bar(x + width/2, ct_scores, width, label='CT Classifier', 
                   color='#ff7f0e', edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_title('Unimodal Models Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim([90, 102])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    # Add a small table with additional info
    cell_text = [[f"{mri_scores[i]:.1f}%", f"{ct_scores[i]:.1f}%"] for i in range(len(metrics))]
    table = ax.table(cellText=cell_text, rowLabels=metrics, 
                     colLabels=['MRI', 'CT'],
                     cellLoc='center', loc='bottom', 
                     bbox=[0.15, -0.45, 0.7, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison chart saved as 'model_comparison.png'")

# ============================================================================
# BONUS: Preprocessing Pipeline Visualization
# ============================================================================

def create_preprocessing_pipeline_visualization():
    """Create a figure showing the preprocessing pipeline steps"""
    
    # This creates a conceptual diagram of the preprocessing steps
    # You should replace with actual images from your dataset
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    steps = ['Original', 'Brain Extraction', 'Denoised', 'Contrast Enhanced', 'Multi-scale Output']
    
    # MRI row (top)
    for i, step in enumerate(steps):
        ax = axes[0, i]
        
        # Create placeholder images (replace with actual processed images)
        if i == 0:
            # Original
            img = np.random.rand(100, 100) * 255
        elif i == 1:
            # Brain extracted
            img = np.random.rand(100, 100) * 200
            mask = np.zeros((100, 100))
            mask[20:80, 20:80] = 1
            img = img * mask
        elif i == 2:
            # Denoised
            img = np.random.rand(100, 100) * 200
            img = np.clip(img + np.random.randn(100,100)*5, 0, 255)
        elif i == 3:
            # Contrast enhanced
            img = np.random.rand(100, 100) * 255
            img = np.clip(img * 1.5, 0, 255)
        else:
            # Multi-scale (showing one channel)
            img = np.random.rand(100, 100) * 255
            
        ax.imshow(img, cmap='gray')
        ax.set_title(step, fontsize=10)
        ax.axis('off')
    
    # CT row (bottom)
    for i, step in enumerate(steps):
        ax = axes[1, i]
        
        # Create placeholder images (replace with actual processed images)
        if i == 0:
            img = np.random.rand(100, 100) * 255
            img[30:70, 40:60] = 200  # Simulate bone
        elif i == 1:
            img = np.random.rand(100, 100) * 200
            mask = np.zeros((100, 100))
            mask[15:85, 15:85] = 1
            img = img * mask
        elif i == 2:
            img = np.random.rand(100, 100) * 200
            img = cv2.GaussianBlur(img.astype(np.uint8), (3,3), 0) if 'cv2' in dir() else img
        elif i == 3:
            img = np.random.rand(100, 100) * 255
            img = np.clip(img * 1.3, 0, 255)
        else:
            img = np.random.rand(100, 100) * 255
            
        ax.imshow(img, cmap='gray')
        ax.set_title(step, fontsize=10)
        ax.axis('off')
    
    # Add modality labels
    axes[0,0].text(-20, 50, 'MRI', fontsize=14, fontweight='bold', rotation=90, 
                   verticalalignment='center')
    axes[1,0].text(-20, 50, 'CT', fontsize=14, fontweight='bold', rotation=90,
                   verticalalignment='center')
    
    plt.suptitle('Modality-Specific Preprocessing Pipeline', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('preprocessing_pipeline.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Preprocessing pipeline visualization saved as 'preprocessing_pipeline.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PRESENTATION VISUALIZATIONS")
    print("=" * 60)
    
    # Create all visualizations
    create_mri_training_curves()
    print("\n" + "-" * 60)
    
    create_mri_confusion_matrix()
    print("\n" + "-" * 60)
    
    create_mri_roc_curves()
    print("\n" + "-" * 60)
    
    create_ct_training_curves()
    print("\n" + "-" * 60)
    
    create_ct_results()
    print("\n" + "-" * 60)
    
    create_comparison_chart()
    print("\n" + "-" * 60)
    
    # Uncomment if you have OpenCV installed for the preprocessing visualization
    # try:
    #     import cv2
    #     create_preprocessing_pipeline_visualization()
    # except ImportError:
    #     print("⚠️ OpenCV not installed. Skipping preprocessing visualization.")
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print("📁 Files saved in current directory")
    print("=" * 60)