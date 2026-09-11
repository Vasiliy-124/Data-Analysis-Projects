"""
Утилиты предобработки текста для классификации токсичных комментариев.

Модуль предоставляет функции для очистки и подготовки текста без удаления
неанглийских токенов для сохранения мультиязычности контента.
"""

import re
from typing import List, Optional
import numpy as np


def clean_text(text: str, lowercase: bool = True, remove_special: bool = False) -> str:
    """
    Очистка и нормализация текста с сохранением мультиязычного контента.
    
    Пайплайн:
    1. Приведение к нижнему регистру (опционально)
    2. Замена HTML-сущностей
    3. Нормализация пробелов
    """
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    if lowercase:
        text = text.lower()
    
    # Замена HTML-сущностей
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление пробелов по краям
    text = text.strip()
    
    return text


def create_vocabulary(texts: List[str], min_freq: int = 1) -> tuple[dict, dict]:
    """
    Создание словаря (маппинги word2id и id2word) на основе корпуса текстов.
    
    Использует токенизацию по пробелам и фильтрует редкие токены по min_freq.
    Специальные токены: 0 - '<pad>', 1 - '<unk>'.
    """
    from collections import Counter
    
    word_freq: Counter = Counter()
    
    # Токенизация и подсчет частот
    for text in texts:
        tokens = text.split()
        word_freq.update(tokens)
    
    # Фильтрация токенов по минимальной частоте
    filtered_tokens = [word for word, freq in word_freq.items() if freq >= min_freq]
    
    # Создание маппингов (с резервированием 0 для <pad> и 1 для <unk>)
    word2id: dict = {'<pad>': 0, '<unk>': 1}
    for idx, word in enumerate(filtered_tokens, start=2):
        word2id[word] = idx
    
    id2word: dict = {idx: word for word, idx in word2id.items()}
    
    return word2id, id2word


def tokenize_and_encode(
    texts: List[str],
    word2id: dict,
    max_length: Optional[int] = None,
    pad_value: int = 0
) -> np.ndarray:
    """
    Преобразование текстовых строк в последовательности ID токенов.
    
    Выполняет токенизацию по пробелам, замену на ID (и '<unk>') 
    и паддинг/обрезку до max_length.
    """
    unk_id: int = word2id.get('<unk>', 1)
    
    # Определение целевой длины последовательности
    if max_length is None:
        max_length = max(len(text.split()) for text in texts) if texts else 0
    
    # Инициализация массива для закодированных последовательностей
    encoded_sequences = np.full((len(texts), max_length), pad_value, dtype=np.int64)
    
    # Кодирование каждого текста
    for idx, text in enumerate(texts):
        tokens = text.split()
        
        # Обрезка до max_length
        tokens = tokens[:max_length]
        
        # Преобразование токенов в ID
        token_ids = [word2id.get(token, unk_id) for token in tokens]
        
        # Сохранение ID токенов (паддинг уже инициализирован)
        encoded_sequences[idx, :len(token_ids)] = token_ids
    
    return encoded_sequences


def get_dataset_stats(texts: List[str]) -> dict:
    """
    Вычисление базовой статистики по текстовому датасету.
    Полезно для оценки характеристик данных и выбора гиперпараметров.
    """
    lengths = [len(text.split()) for text in texts]
    
    return {
        'num_samples': len(texts),
        'avg_length': float(np.mean(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths))
    }