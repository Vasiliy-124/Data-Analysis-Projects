"""
Метрики оценки для multi-label классификации токсичности.

Модуль реализует метрики, специфичные для задач multi-label классификации.
Основная метрика — ROC AUC, вычисляемая для каждой метки и усредненная (macro-average).
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score, hamming_loss, accuracy_score, 
    precision_recall_fscore_support, roc_curve, auc
)


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Вычисляет ROC AUC для multi-label классификации.
    """
    try:
        # roc_auc_score автоматически обрабатывает multi-label формат
        score = roc_auc_score(y_true, y_pred_proba, average=average, multi_class='ovr')
        return score
    except ValueError as e:
        print(f"Warning: ROC AUC computation failed - {e}")
        return 0.0


def compute_per_label_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: Optional[list] = None
) -> dict:
    """
    Вычисляет ROC AUC для каждой метки отдельно.
    Полезно для выявления классов, которые сложнее или проще предсказать.
    """
    n_labels = y_true.shape[1]
    
    if label_names is None:
        label_names = [f'label_{i}' for i in range(n_labels)]
    
    scores = {}
    for i in range(n_labels):
        try:
            score = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            scores[label_names[i]] = score
        except ValueError:
            # Обработка случаев, когда метка содержит только один класс в выборке
            scores[label_names[i]] = 0.0
    
    return scores


def compute_hamming_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Вычисляет потерю Хэмминга (Hamming loss) — долю неверно предсказанных меток.
    """
    return hamming_loss(y_true, y_pred)


def compute_subset_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Вычисляет точность подмножества (Subset accuracy / exact match ratio).
    Строгая метрика: требует, чтобы ВСЕ метки для примера были предсказаны верно.
    """
    # accuracy_score вычисляет долю полных совпадений (exact match ratio)
    return accuracy_score(y_true, y_pred)


def compute_multi_label_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[list] = None
) -> dict:
    """
    Вычисляет комплексные метрики для multi-label классификации.
    Удобная функция для оценки и сравнения моделей.
    """
    # Преобразование вероятностей в бинарные предсказания по порогу
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    # Глобальные метрики
    metrics['roc_auc_macro'] = compute_roc_auc(y_true, y_pred_proba, average='macro')
    metrics['roc_auc_micro'] = compute_roc_auc(y_true, y_pred_proba, average='micro')
    metrics['hamming_loss'] = compute_hamming_loss(y_true, y_pred)
    metrics['subset_accuracy'] = compute_subset_accuracy(y_true, y_pred)
    
    # Метрики для каждой метки (per-label)
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f'label_{i}' for i in range(n_labels)]
    
    metrics['per_label_roc_auc'] = compute_per_label_roc_auc(
        y_true, y_pred_proba, label_names
    )
    
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    for i in range(n_labels):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        precision_dict[label_names[i]] = precision
        recall_dict[label_names[i]] = recall
        f1_dict[label_names[i]] = f1
    
    metrics['per_label_precision'] = precision_dict
    metrics['per_label_recall'] = recall_dict
    metrics['per_label_f1'] = f1_dict
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func=None
) -> Tuple[float, float]:
    """
    Поиск оптимального порога (threshold) для перевода вероятностей в бинарные метки.
    По умолчанию максимизирует ROC AUC (macro).
    """
    if metric_func is None:
        metric_func = lambda y_true, y_pred: compute_roc_auc(y_true, y_pred)
    
    best_threshold = 0.5
    best_score = 0.0
    
    # Поиск по сетке значений порога (Grid search)
    for threshold in np.arange(0.0, 1.01, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = metric_func(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score