"""
Утилиты для обучения нейросетевых моделей.

Обеспечивает циклы обучения, валидацию и управление контрольными точками.
Разработано специально для задач multi-label классификации.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
import numpy as np
from tqdm import tqdm


def train_cnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    patience: int = 3,
    verbose: bool = True
) -> Tuple[List[float], List[float], List[float]]:
    """
    Обучение CNN модели с ранней остановкой (early stopping) по validation loss.
    
    Включает:
    - Прямой проход (forward pass)
    - Расчет функции потерь (BCEWithLogitsLoss для multi-label)
    - Обратное распространение ошибки и шаг оптимизатора
    - Валидацию и проверку критерия early stopping
    """
    model.to(device)
    train_losses = []
    val_losses = []
    epochs_list = []
    
    # Для ранней остановки (early stopping)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # ЭТАП ОБУЧЕНИЯ
        model.train()  # Режим обучения (включает dropout)
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [TRAIN]",
            disable=not verbose
        )
        
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Прямой проход (forward pass)
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            # Расчет функции потерь (BCEWithLogitsLoss комбинирует sigmoid + BCE)
            loss = criterion(predictions, batch_y.float())
            
            # Обратное распространение ошибки и шаг оптимизатора
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # ЭТАП ВАЛИДАЦИИ
        model.eval()  # Режим оценки (отключает dropout)
        val_loss = 0.0
        val_batches = 0
        
        # Отключаем вычисление градиентов для валидации
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [VAL]",
                disable=not verbose
            )
            
            for batch_x, batch_y in progress_bar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y.float())
                
                val_loss += loss.item()
                val_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # ПРОВЕРКА РАННЕЙ ОСТАНОВКИ (EARLY STOPPING)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if verbose:
                print(f"  → Validation loss improved! Saving checkpoint...")
        else:
            patience_counter += 1
            if verbose:
                print(f"  → No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return train_losses, val_losses, epochs_list


def predict_with_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> np.ndarray:
    """
    Генерация предсказаний для тестовой выборки.
    Переводит сырые логиты в независимые вероятности [0, 1] через Sigmoid.
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Generating predictions",
            disable=not verbose
        )
        
        for batch_x, _ in progress_bar:
            batch_x = batch_x.to(device)
            
            # 1. Получение сырых логитов от сети
            logits = model(batch_x)
            
            # 2. Применение Sigmoid для перевода логитов в вероятности
            probs = torch.sigmoid(logits)
            
            all_predictions.append(probs.cpu().numpy())
    
    return np.vstack(all_predictions)


def create_data_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Создает PyTorch DataLoaders для обучающей и валидационной выборок.
    """
    # Конвертация numpy-массивов в тензоры PyTorch
    x_train_tensor = torch.from_numpy(x_train).long()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_val_tensor = torch.from_numpy(x_val).long()
    y_val_tensor = torch.from_numpy(y_val).float()
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True  # Оптимизация передачи данных в память GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_loader(
    x_test: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, int]:
    """
    Создает DataLoader для тестовой выборки.
    """
    x_test_tensor = torch.from_numpy(x_test).long()
    
    # Заглушки для меток (не используются при инференсе)
    y_dummy = torch.zeros((len(x_test), 6))
    test_dataset = TensorDataset(x_test_tensor, y_dummy)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return test_loader, len(x_test)