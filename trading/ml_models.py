from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

try:  # Optional metric backend
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore[assignment]


def _ensure_torch_available() -> None:
    if torch is None or nn is None or TensorDataset is None:
        raise RuntimeError("需要安装 PyTorch 才能启用深度模型（LSTM/Transformer）。")


def _build_sequence_tensors(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[pd.Timestamp]]:
    if sequence_length <= 1:
        sequence_length = 2
    values = frame[feature_columns].astype(np.float32).to_numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    targets = frame["_target"].astype(np.float32).to_numpy()
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    index: list[pd.Timestamp] = []
    for i in range(sequence_length - 1, len(values)):
        seq = values[i - sequence_length + 1 : i + 1]
        sequences.append(seq)
        labels.append(float(targets[i]))
        index.append(frame.index[i])
    if not sequences:
        return None, None, []
    seq_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return seq_tensor, label_tensor, index


@dataclass(slots=True)
class LSTMSettings:
    feature_columns: Sequence[str]
    sequence_length: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 3e-3


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2 if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMClassifierWrapper:
    """
    Minimal scikit-learn 风格封装，便于在现有策略管线中调用 PyTorch LSTM。
    """

    def __init__(self, settings: LSTMSettings) -> None:
        _ensure_torch_available()
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _LSTMClassifier | None = None

    def _prepare_frame(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        frame = X.copy()
        if y is not None:
            frame["_target"] = y.values
        else:
            frame["_target"] = 0.0
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMClassifierWrapper":
        frame = self._prepare_frame(X, y)
        seq_tensor, label_tensor, _ = _build_sequence_tensors(
            frame,
            self.settings.feature_columns,
            self.settings.sequence_length,
        )
        if seq_tensor is None or label_tensor is None:
            # 样本不足时 fallback 为常数概率
            self.model = None
            self._fallback = float(y.mean() if len(y) else 0.5)
            return self
        dataset = TensorDataset(seq_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=self.settings.batch_size, shuffle=True, drop_last=False)
        input_dim = len(self.settings.feature_columns)
        self.model = _LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=max(16, int(self.settings.hidden_dim)),
            num_layers=max(1, int(self.settings.num_layers)),
            dropout=max(0.0, float(self.settings.dropout)),
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)
        self.model.train()
        for _ in range(max(1, int(self.settings.epochs))):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._prepare_frame(X)
        prob_series = pd.Series(0.5, index=frame.index, dtype=float)
        seq_tensor, _, index = _build_sequence_tensors(
            frame,
            self.settings.feature_columns,
            self.settings.sequence_length,
        )
        if seq_tensor is not None and self.model is not None:
            with torch.no_grad():
                logits = self.model(seq_tensor.to(self.device))
                probs = torch.sigmoid(logits).cpu().numpy()
            prob_series.loc[index] = probs
        elif hasattr(self, "_fallback"):
            prob_series[:] = getattr(self, "_fallback")
        clipped = np.clip(prob_series.to_numpy(dtype=float), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - clipped, clipped])


@dataclass(slots=True)
class TransformerSettings:
    feature_columns: Sequence[str]
    sequence_length: int = 32
    model_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length, :]


class _TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, settings: TransformerSettings) -> None:
        super().__init__()
        model_dim = max(32, settings.model_dim)
        num_heads = max(1, min(settings.num_heads, model_dim // 4))
        if model_dim % num_heads != 0:
            model_dim += num_heads - (model_dim % num_heads)
        self.proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=settings.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, settings.num_layers))
        self.position = _PositionalEncoding(model_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.position(x)
        encoded = self.encoder(x)
        cls = encoded[:, -1, :]
        return self.head(cls).squeeze(-1)


class TransformerClassifierWrapper:
    def __init__(self, settings: TransformerSettings) -> None:
        _ensure_torch_available()
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _TransformerClassifier | None = None

    def _prepare_frame(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        frame = X.copy()
        frame["_target"] = y.values if y is not None else 0.0
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransformerClassifierWrapper":
        frame = self._prepare_frame(X, y)
        seq_tensor, label_tensor, _ = _build_sequence_tensors(
            frame,
            self.settings.feature_columns,
            self.settings.sequence_length,
        )
        if seq_tensor is None or label_tensor is None:
            self.model = None
            self._fallback = float(y.mean() if len(y) else 0.5)
            return self
        dataset = TensorDataset(seq_tensor, label_tensor)
        loader = DataLoader(dataset, batch_size=self.settings.batch_size, shuffle=True, drop_last=False)
        input_dim = len(self.settings.feature_columns)
        self.model = _TransformerClassifier(input_dim, self.settings).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.settings.learning_rate)
        self.model.train()
        for _ in range(max(1, int(self.settings.epochs))):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._prepare_frame(X)
        prob_series = pd.Series(0.5, index=frame.index, dtype=float)
        seq_tensor, _, index = _build_sequence_tensors(
            frame,
            self.settings.feature_columns,
            self.settings.sequence_length,
        )
        if seq_tensor is not None and self.model is not None:
            with torch.no_grad():
                logits = self.model(seq_tensor.to(self.device))
                probs = torch.sigmoid(logits).cpu().numpy()
            prob_series.loc[index] = probs
        elif hasattr(self, "_fallback"):
            prob_series[:] = getattr(self, "_fallback")
        clipped = np.clip(prob_series.to_numpy(dtype=float), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - clipped, clipped])


class HybridSequenceEnsembleWrapper:
    """Train LSTM + Transformer in parallel and auto-select or blend their outputs."""

    def __init__(
        self,
        lstm_settings: LSTMSettings,
        transformer_settings: TransformerSettings,
        validation_ratio: float = 0.2,
        selection_mode: str = "auto",
        score_tolerance: float = 0.02,
    ) -> None:
        _ensure_torch_available()
        self.lstm = LSTMClassifierWrapper(lstm_settings)
        self.transformer = TransformerClassifierWrapper(transformer_settings)
        self.validation_ratio = float(np.clip(validation_ratio, 0.05, 0.4))
        self.selection_mode = selection_mode.lower()
        self.score_tolerance = max(0.0, float(score_tolerance))
        self.active_model = "blend"
        self.validation_scores: dict[str, float] = {}

    def _min_sequence_length(self) -> int:
        return int(
            max(
                getattr(self.lstm.settings, "sequence_length", 16),
                getattr(self.transformer.settings, "sequence_length", 16),
                16,
            )
        )

    def _score_predictions(self, y_true: pd.Series, probs: np.ndarray) -> float | None:
        if y_true.empty:
            return None
        y_clean = y_true.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if y_clean.empty:
            return None
        aligned_probs = pd.Series(probs, index=y_true.index).loc[y_clean.index].to_numpy(dtype=float)
        if aligned_probs.size == 0:
            return None
        unique = y_clean.nunique(dropna=True)
        if roc_auc_score is not None and unique > 1:
            try:
                return float(roc_auc_score(y_clean.to_numpy(dtype=float), aligned_probs))
            except Exception:
                pass
        preds = (aligned_probs >= 0.5).astype(float)
        return float((preds == y_clean.to_numpy(dtype=float)).mean())

    def _evaluate_models(self, val_x: pd.DataFrame, val_y: pd.Series) -> dict[str, float]:
        scores: dict[str, float] = {}
        if val_x.empty or len(val_y) == 0:
            return scores
        for name, model in ("lstm", self.lstm), ("transformer", self.transformer):
            try:
                probs = model.predict_proba(val_x)[:, 1]
            except Exception:
                continue
            score = self._score_predictions(val_y, probs)
            if score is not None:
                scores[name] = score
        return scores

    def _decide_active_model(self, scores: dict[str, float]) -> None:
        if self.selection_mode in {"lstm", "transformer", "blend"}:
            if self.selection_mode == "lstm":
                self.active_model = "lstm"
                return
            if self.selection_mode == "transformer":
                self.active_model = "transformer"
                return
            self.active_model = "blend"
            return
        if not scores:
            self.active_model = "blend"
            return
        if self.selection_mode == "average":
            self.active_model = "blend"
            return
        if len(scores) == 1:
            self.active_model = next(iter(scores.keys()))
            return
        lstm_score = scores.get("lstm")
        transformer_score = scores.get("transformer")
        if lstm_score is not None and transformer_score is not None:
            if abs(lstm_score - transformer_score) <= self.score_tolerance:
                self.active_model = "blend"
                return
            self.active_model = "lstm" if lstm_score > transformer_score else "transformer"
            return
        self.active_model = max(scores, key=scores.get)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridSequenceEnsembleWrapper":
        total = len(X)
        if total == 0:
            self.lstm.fit(X, y)
            self.transformer.fit(X, y)
            self.active_model = "blend"
            return self
        cutoff = int(total * (1 - self.validation_ratio))
        cutoff = min(max(cutoff, self._min_sequence_length()), total - 5) if total > 10 else max(total - 3, 1)
        if cutoff <= 0:
            cutoff = max(total // 2, 1)
        train_x = X.iloc[:cutoff]
        train_y = y.iloc[:cutoff]
        val_x = X.iloc[cutoff:]
        val_y = y.iloc[cutoff:]
        if val_x.empty or len(val_y) < 5:
            split = max(total - 10, total // 2, self._min_sequence_length())
            split = min(split, total - 1) if total > 1 else 0
            train_x = X.iloc[:split]
            train_y = y.iloc[:split]
            val_x = X.iloc[split:]
            val_y = y.iloc[split:]

        self.lstm.fit(train_x, train_y)
        self.transformer.fit(train_x, train_y)
        self.validation_scores = self._evaluate_models(val_x, val_y)
        self._decide_active_model(self.validation_scores)
        # 重新用全量数据训练以提供更稳健的推理
        self.lstm.fit(X, y)
        self.transformer.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lstm_proba = self.lstm.predict_proba(X)
        transformer_proba = self.transformer.predict_proba(X)
        if self.active_model == "lstm":
            return lstm_proba
        if self.active_model == "transformer":
            return transformer_proba
        # 默认进行平均融合
        return (lstm_proba + transformer_proba) / 2.0


def build_custom_sequence_model(
    model_name: str,
    feature_columns: Sequence[str],
    params: Any,
) -> Tuple[Any, str]:
    name = (model_name or "").lower()
    overrides = dict(getattr(params, "ml_params", {}) or {})
    if name == "lstm":
        settings = LSTMSettings(
            feature_columns=feature_columns,
            sequence_length=int(overrides.get("sequence_length", getattr(params, "dl_sequence_length", 32))),
            hidden_dim=int(overrides.get("hidden_dim", getattr(params, "dl_hidden_dim", 64))),
            num_layers=int(overrides.get("num_layers", getattr(params, "dl_num_layers", 2))),
            dropout=float(overrides.get("dropout", getattr(params, "dl_dropout", 0.2))),
            epochs=int(overrides.get("epochs", getattr(params, "dl_epochs", 12))),
            batch_size=int(overrides.get("batch_size", getattr(params, "dl_batch_size", 64))),
            learning_rate=float(overrides.get("learning_rate", 3e-3)),
        )
        return LSTMClassifierWrapper(settings), f"LSTM 序列模型 (seq={settings.sequence_length}, hidden={settings.hidden_dim})"
    if name == "transformer":
        settings = TransformerSettings(
            feature_columns=feature_columns,
            sequence_length=int(overrides.get("sequence_length", getattr(params, "dl_sequence_length", 32))),
            model_dim=int(overrides.get("model_dim", getattr(params, "dl_hidden_dim", 96))),
            num_heads=int(overrides.get("num_heads", max(2, getattr(params, "dl_num_layers", 2)))),
            num_layers=int(overrides.get("num_layers", getattr(params, "dl_num_layers", 2))),
            dropout=float(overrides.get("dropout", getattr(params, "dl_dropout", 0.1))),
            epochs=int(overrides.get("epochs", getattr(params, "dl_epochs", 10))),
            batch_size=int(overrides.get("batch_size", getattr(params, "dl_batch_size", 64))),
            learning_rate=float(overrides.get("learning_rate", 5e-4)),
        )
        return TransformerClassifierWrapper(settings), f"Transformer 编码器 (layers={settings.num_layers}, heads={settings.num_heads})"
    if name in {"seq_hybrid", "hybrid_seq", "fusion"}:
        sequence_length = int(overrides.get("sequence_length", getattr(params, "dl_sequence_length", 32)))
        lstm_settings = LSTMSettings(
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            hidden_dim=int(overrides.get("lstm_hidden_dim", overrides.get("hidden_dim", getattr(params, "dl_hidden_dim", 64)))),
            num_layers=int(overrides.get("lstm_num_layers", overrides.get("num_layers", getattr(params, "dl_num_layers", 2)))),
            dropout=float(overrides.get("lstm_dropout", overrides.get("dropout", getattr(params, "dl_dropout", 0.2)))),
            epochs=int(overrides.get("lstm_epochs", overrides.get("epochs", getattr(params, "dl_epochs", 12)))),
            batch_size=int(overrides.get("lstm_batch_size", overrides.get("batch_size", getattr(params, "dl_batch_size", 64)))),
            learning_rate=float(overrides.get("lstm_learning_rate", overrides.get("learning_rate", 3e-3))),
        )
        transformer_settings = TransformerSettings(
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            model_dim=int(overrides.get("model_dim", getattr(params, "dl_hidden_dim", 96))),
            num_heads=int(overrides.get("num_heads", max(2, getattr(params, "dl_num_layers", 2)))),
            num_layers=int(overrides.get("transformer_num_layers", overrides.get("num_layers", getattr(params, "dl_num_layers", 2)))),
            dropout=float(overrides.get("transformer_dropout", overrides.get("dropout", getattr(params, "dl_dropout", 0.1)))),
            epochs=int(overrides.get("transformer_epochs", overrides.get("epochs", getattr(params, "dl_epochs", 10)))),
            batch_size=int(overrides.get("transformer_batch_size", overrides.get("batch_size", getattr(params, "dl_batch_size", 64)))),
            learning_rate=float(overrides.get("transformer_learning_rate", overrides.get("learning_rate", 5e-4))),
        )
        validation_ratio = float(overrides.get("validation_ratio", getattr(params, "val_ratio", 0.2)))
        selection_mode = str(overrides.get("selection_mode", "auto"))
        score_tolerance = float(overrides.get("score_tolerance", 0.02))
        wrapper = HybridSequenceEnsembleWrapper(
            lstm_settings,
            transformer_settings,
            validation_ratio=validation_ratio,
            selection_mode=selection_mode,
            score_tolerance=score_tolerance,
        )
        desc = (
            "LSTM+Transformer 自适应融合"
            f" (mode={wrapper.selection_mode}, val_ratio={validation_ratio:.2f})"
        )
        return wrapper, desc
    raise ValueError(f"未知的深度模型 {model_name}")
