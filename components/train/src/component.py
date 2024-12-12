import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from tqdm import tqdm
import yaml
from pathlib import Path
import sys

# Add the project root to Python path to enable imports
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

# Updated imports based on new directory structure
from components.source.src.component import load_and_preprocess_data
from components.transform.src.component import DataAugmentation
from components.train.src.model import get_model, ModelCheckpoint
from components.validation.src.component import MetricsCalculator

class Trainer:
    def __init__(self, config_path='config.yml'):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for artifacts"""
        for dir_path in [
            self.config['paths']['model_dir'],
            self.config['paths']['log_dir'],
            self.config['paths']['results_dir']
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def train_model(self, model_name, train_loader, val_loader, test_loader):
        """Training loop for a single model"""
        # Initialize model
        model = get_model(
            model_name=model_name,
            pretrained=True,
            device=self.device
        )
        print(f"Model {model_name} is loaded")

        # Initialize metrics calculator
        metrics_calculator = MetricsCalculator(model_name, save_dir=self.config['paths']['results_dir'])
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['training']['lr_scheduler_factor'],
            patience=self.config['training']['lr_scheduler_patience'],
            verbose=True
        )
        
        # Initialize checkpoint handler
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.config['paths']['model_dir'], f'best_model_{model_name}.pth'),
            monitor='val_loss',
            mode='min'
        )
        
        # Track metrics history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Set up MLflow experiment for this model
        experiment_name = f"{self.config['mlflow']['experiment_name']}_{model_name}"
        mlflow.set_experiment(experiment_name)
        
        # Training loop
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params({
                **self.config,
                'model_name': model_name
            })
            
            for epoch in range(self.config['training']['epochs']):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                train_pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{self.config["training"]["epochs"]}')
                
                for batch_idx, (inputs, labels) in enumerate(train_pbar):
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                    
                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    preds = torch.sigmoid(outputs) > 0.5
                    train_total += labels.size(0)
                    train_correct += preds.eq(labels).sum().item()
                    
                    # Update progress bar
                    train_acc = 100. * train_correct / train_total
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.3f}',
                        'acc': f'{train_acc:.2f}%'
                    })
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_predictions = []
                val_probs = []
                val_targets = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        probs = torch.sigmoid(outputs)
                        predicted = torch.sigmoid(outputs) > 0.5
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        
                        val_probs.extend(probs.cpu().numpy())
                        val_predictions.extend(predicted.cpu().numpy())
                        val_targets.extend(labels.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total
                
                # Update history
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                # Calculate and log metrics
                metrics = metrics_calculator.calculate_metrics(
                    val_targets, 
                    val_predictions,
                    val_probs
                )
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    **metrics
                }, step=epoch)
                
                print(f'\n[{model_name}] Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
                print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Save checkpoint if improved
                if checkpoint(avg_val_loss, model, epoch):
                    print(f'Model {model_name} saved at epoch {epoch+1}')
            
            # Plot training history
            metrics_calculator.plot_training_history(history)
            
            # Final evaluation on test set
            print(f"\nEvaluating model {model_name} on test set:")
            test_metrics = metrics_calculator.evaluate_model(
                model, 
                test_loader, 
                criterion, 
                self.device
            )
            print(f"\nTest set metrics for {model_name}:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")
                
            return model, history, test_metrics
    
    def train(self):
        """Main training loop for all models"""
        # Get data loaders
        train_loader, val_loader, test_loader = load_and_preprocess_data(
            self.config['data']['base_dir'],
            DataAugmentation.get_train_transforms(),
            DataAugmentation.get_val_transforms(),
            self.config['data']['batch_size']
        )
        print("Data is loaded")
        
        results = {}
        for model_name in self.config['model_names']:
            print(f"\nTraining model: {model_name}")
            model, history, test_metrics = self.train_model(
                model_name,
                train_loader,
                val_loader,
                test_loader
            )
            results[model_name] = {
                'model': model,
                'history': history,
                'test_metrics': test_metrics
            }
            
        return results

if __name__ == '__main__':
    trainer = Trainer()
    results = trainer.train()