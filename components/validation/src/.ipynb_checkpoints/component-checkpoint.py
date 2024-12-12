import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_curve, 
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import mlflow
import pandas as pd
import os
from datetime import datetime

class MetricsCalculator:
    def __init__(self, model_name, save_dir='results'):
        """
        Initialize MetricsCalculator with a directory to save results
        
        Args:
            save_dir (str): Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate various classification metrics
        
        Args:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels
            y_prob (array-like, optional): Prediction probabilities for positive class
            
        Returns:
            dict: Dictionary containing various metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Add AUC-ROC if probabilities are provided
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            metrics_dict['auc_roc'] = roc_auc
            
            # Average precision score (AP)
            ap = average_precision_score(y_true, y_prob)
            metrics_dict['average_precision'] = ap
        
        return metrics_dict

    def plot_confusion_matrix(self, y_true, y_pred, classes=['Normal', 'Cataract']):
        """
        Plot and save confusion matrix
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot with both counts and percentages
        sns.heatmap(cm, annot=np.asarray([
            [f'{count}\n{percent:.1f}%' for count, percent in zip(row_counts, row_percentages)]
            for row_counts, row_percentages in zip(cm, cm_percent)
        ]), fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'confusion_matrix_{self.model_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Log to MLflow if in active run
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        return save_path

    def plot_roc_curve(self, y_true, y_prob):
        """
        Plot and save ROC curve
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'roc_curve_{self.model_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        return save_path

    def plot_precision_recall_curve(self, y_true, y_prob):
        """
        Plot and save Precision-Recall curve
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # Plot
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'pr_curve_{self.model_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        return save_path

    def plot_training_history(self, history):
        """
        Plot training history curves
        
        Args:
            history (dict): Dictionary containing training history
                          (loss, accuracy, val_loss, val_accuracy)
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'training_history_{self.model_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        return save_path

    def generate_classification_report(self, y_true, y_pred, classes=['Normal', 'Cataract']):
        """
        Generate and save detailed classification report
        """
        # Get the classification report as a dictionary
        report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        
        # Convert to DataFrame for better formatting
        df_report = pd.DataFrame(report_dict).transpose()
        
        # Save report
        save_path = os.path.join(self.save_dir, f'classification_report_{self.model_name}.csv')
        df_report.to_csv(save_path)
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        return df_report

    def evaluate_model(self, model, test_loader, criterion, device):
        """
        Comprehensive model evaluation
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            criterion: Loss function
            device: Device to run evaluation on
        """
        model.eval()
        test_loss = 0
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                # Calculate probabilities
                probabilities = torch.sigmoid(outputs)
                predicted = torch.sigmoid(outputs) > 0.5
                # Store results
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probabilities.cpu().numpy())  # Probability of positive class
                
                test_loss += loss.item()
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        metrics['test_loss'] = test_loss / len(test_loader)
        
        # Generate plots and reports
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_prob)
        self.plot_precision_recall_curve(y_true, y_prob)
        self.generate_classification_report(y_true, y_pred)
        
        # Log metrics to MLflow if in active run
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
        
        return metrics

def log_batch_metrics(epoch, batch_idx, loss, acc, mode='train'):
    """
    Log batch-level metrics during training
    """
    if mlflow.active_run():
        mlflow.log_metrics({
            f'{mode}_batch_loss': loss,
            f'{mode}_batch_acc': acc,
        }, step=(epoch * 10000 + batch_idx))  # Using a large multiplier to separate epochs