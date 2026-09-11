"""
Модель PyTorch CNN для multi-label классификации токсичных комментариев.

Архитектура:
    Embedding → Conv1D (несколько ядер) → MaxPool1D → Flatten → Dense → Sigmoid
"""

import torch
import torch.nn as nn
from typing import Optional


class TextCNN(nn.Module):
    """
    1D CNN для классификации текстов.
    Использует фильтры разных размеров для извлечения n-gram паттернов:
    - kernel_size=2: биграммы
    - kernel_size=3: триграммы
    - kernel_size=4: 4-граммы и т.д.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_classes: int = 6,
        num_filters: int = 100,
        kernel_sizes: tuple = (2, 3, 4, 5),
        dropout_rate: float = 0.5
    ):
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Слой эмбеддингов (отображение токенов в плотные векторы)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Сверточные слои (параллельные 1D свертки с разными размерами ядер)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in kernel_sizes
        ])
        
        # Слой пулинга (Global Max Pooling)
        self.pool = nn.MaxPool1d(kernel_size=1)
        
        # Dropout для регуляризации и предотвращения переобучения
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Подсчет общего числа признаков после конкатенации
        total_filters = len(kernel_sizes) * num_filters
        
        # Полносвязные слои (FC)
        self.fc1 = nn.Linear(total_filters, 128)
        self.fc_hidden = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, num_classes)
        
        # Функции активации
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход (forward pass).
        Возвращает сырые логиты (Sigmoid применяется на этапе инференса или внутри функции потерь).
        """
        # Эмбеддинги и перестановка осей для Conv1d: (batch, channels, length)
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        # Применение сверток и пулинга
        conv_outputs = []
        
        for conv in self.convs:
            conv_out = self.relu(conv(embedded))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Конкатенация выходов всех сверток
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        dropped = self.dropout(concatenated)
        
        # Полносвязные слои
        fc1_out = self.relu(self.fc1(dropped))
        fc1_out = self.dropout(fc1_out)
        
        hidden = self.relu(self.fc_hidden(fc1_out))
        hidden = self.dropout(hidden)
        
        logits = self.fc_output(hidden)
        
        return logits