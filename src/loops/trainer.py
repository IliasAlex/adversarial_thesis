import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20):
    model.to(device)
    
    best_val_acc = 0.0
    best_model_wts = None
    best_val_loss = float('inf')  # Track the lowest validation loss
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            if data.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                data = data.unsqueeze(1)

            # Move data and labels to the device (GPU/CPU)
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = model.state_dict()  # Save best model
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
        
        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break  # Stop training
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    print("Training completed.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}, Best Validation Loss: {best_val_loss:.4f}")
    return model

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, _ in tqdm(data_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            if data.dim() == 3:  # Ensure `unsqueeze(1)` is only applied if there is no channel dimension
                data = data.unsqueeze(1)  
            outputs = model(data)  
            loss = criterion(outputs, labels)
            
            # Accumulate metrics
            running_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy

def test(model, test_loader, device):
    print("Testing model...")
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc