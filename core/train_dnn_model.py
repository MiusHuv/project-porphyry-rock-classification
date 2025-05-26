import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

# Ensure optuna is installed: pip install optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not found. Please install it: pip install optuna. "
          "DNN hyperparameter tuning will use default parameters.")
    OPTUNA_AVAILABLE = False

MODEL_SAVE_DIR = "trained_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for PyTorch DNN training.")

# 1. Define the PyTorch Model
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers_config, dropout_rates):
        super(SimpleDNN, self).__init__()
        layers = []
        current_dim = input_dim
        for i, hidden_units in enumerate(hidden_layers_config):
            layers.append(nn.Linear(current_dim, hidden_units))
            layers.append(nn.ReLU())
            if i < len(dropout_rates): # Apply dropout if specified for this layer
                 layers.append(nn.Dropout(dropout_rates[i]))
            current_dim = hidden_units
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        # Softmax will be applied in loss function (CrossEntropyLoss) for multi-class
        # or handled separately for binary (BCEWithLogitsLoss)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 2. Training and Evaluation Functions
def train_epoch(model, dataloader, criterion, optimizer, num_classes):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        if num_classes == 2: # Binary classification
            # BCEWithLogitsLoss expects raw logits and labels as float
            loss = criterion(outputs.squeeze(), labels.float())
        else: # Multi-class
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        if num_classes == 2:
            preds = torch.sigmoid(outputs).squeeze() > 0.5
        else:
            _, preds = torch.max(outputs, 1)
            
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            if num_classes == 2:
                loss = criterion(outputs.squeeze(), labels.float())
                preds = torch.sigmoid(outputs).squeeze() > 0.5
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# 3. Optuna Objective Function
def objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, input_dim, num_classes, epochs):
    # Hyperparameters to tune
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_configs = []
    dropout_configs = []
    for i in range(n_layers):
        hidden_configs.append(trial.suggest_categorical(f'n_units_l{i}', [32, 64, 128, 256]))
        dropout_configs.append(trial.suggest_float(f'dropout_l{i}', 0.1, 0.5))
    
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop']) # SGD can also be an option

    # Create model
    model = SimpleDNN(input_dim, 
                      num_classes if num_classes > 2 else 1, # Output 1 neuron for binary with BCEWithLogitsLoss
                      hidden_configs, 
                      dropout_configs).to(DEVICE)

    # Criterion and Optimizer
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally, more stable
    else:
        criterion = nn.CrossEntropyLoss()
        
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else: # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 7 # Early stopping patience

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, num_classes)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, num_classes)
        
        trial.report(val_acc, epoch) # Report validation accuracy to Optuna

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if trial.should_prune(): # Optuna pruning
            raise optuna.exceptions.TrialPruned()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} for trial {trial.number}")
            break
            
    return best_val_acc # Optuna tries to maximize this

# 4. Main Training Function
def train_pytorch_dnn(X_train, y_train, X_val, y_val, input_dim, num_classes,
                      epochs=50, n_optuna_trials=20, 
                      project_name="pytorch_dnn_optuna", model_filename="pytorch_dnn_model.pth"):
    
    # Convert pandas/numpy to PyTorch tensors
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    if isinstance(y_train, pd.Series): y_train = y_train.values
    if isinstance(X_val, pd.DataFrame): X_val = X_val.values
    if isinstance(y_val, pd.Series): y_val = y_val.values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long if num_classes > 2 else torch.float32) # Long for CrossEntropy, Float for BCE
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long if num_classes > 2 else torch.float32)

    best_params = None
    best_model_state_dict = None

    if OPTUNA_AVAILABLE:
        print(f"\nStarting Optuna hyperparameter search for PyTorch DNN ({n_optuna_trials} trials)...")
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, X_train_tensor, y_train_tensor, 
                                               X_val_tensor, y_val_tensor, 
                                               input_dim, num_classes, epochs=epochs), 
                       n_trials=n_optuna_trials, 
                       timeout=600) # Optional timeout in seconds

        best_params = study.best_trial.params
        print("\nBest Optuna trial:")
        print(f"  Value (Validation Accuracy): {study.best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # Build and train the best model with optimal hyperparameters
        n_layers = best_params.get('n_layers', 1)
        hidden_configs = [best_params.get(f'n_units_l{i}', 64) for i in range(n_layers)]
        dropout_configs = [best_params.get(f'dropout_l{i}', 0.2) for i in range(n_layers)]
        lr = best_params.get('lr', 0.001)
        optimizer_name = best_params.get('optimizer', 'Adam')
        batch_size = best_params.get('batch_size', 64)

        best_model = SimpleDNN(input_dim, 
                               num_classes if num_classes > 2 else 1,
                               hidden_configs, dropout_configs).to(DEVICE)
    else: # Fallback to default parameters if Optuna is not available
        print("\nOptuna not available. Training PyTorch DNN with default parameters.")
        # Default parameters
        hidden_configs = [128, 64]
        dropout_configs = [0.3, 0.2]
        lr = 0.001
        optimizer_name = 'Adam'
        batch_size = 64
        best_model = SimpleDNN(input_dim, 
                               num_classes if num_classes > 2 else 1,
                               hidden_configs, dropout_configs).to(DEVICE)

    # Final training of the best model (or default model)
    print("\nTraining final PyTorch DNN model...")
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(best_model.parameters(), lr=lr)
    else: # RMSprop or other
        optimizer = optim.RMSprop(best_model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor) # For final validation during training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    final_epochs = epochs * 2 # Train for longer on the best/default config
    best_val_acc_final = 0.0
    patience_counter_final = 0
    patience_final = 10 # Early stopping for final model

    for epoch in range(final_epochs):
        train_loss, train_acc = train_epoch(best_model, train_loader, criterion, optimizer, num_classes)
        val_loss, val_acc = evaluate_model(best_model, val_loader, criterion, num_classes)
        print(f"Epoch {epoch+1}/{final_epochs} => Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc_final:
            best_val_acc_final = val_acc
            best_model_state_dict = best_model.state_dict() # Save best performing model state
            patience_counter_final = 0
        else:
            patience_counter_final += 1
        
        if patience_counter_final >= patience_final:
            print(f"Early stopping final training at epoch {epoch+1}.")
            break
    
    # Load the best performing state into the model
    if best_model_state_dict:
        best_model.load_state_dict(best_model_state_dict)

    # Save the trained model
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    try:
        torch.save(best_model.state_dict(), model_save_path)
        print(f"PyTorch DNN model state_dict saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving PyTorch DNN model: {e}")
            
    return best_model # Return the trained model instance