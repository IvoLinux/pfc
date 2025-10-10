# schemas.py
from datetime import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, ConfigDict

class JobCreateTrain(BaseModel):
    kind: str
    model_name: str
    dataset_filename: str
    title: Optional[str] = None

    num_epochs:    int = 1
    max_length:    int = 512
    batch_size:    int = 8
    learning_rate: float = 3e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.05

class JobCreateInfer(BaseModel):
    kind: str
    dataset_filename: str
    checkpoint_filename: str
    title: Optional[str] = None

class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    progress: int
    submitted_at: datetime
    title: Optional[str] = None
    metrics_json: Optional[Dict[str, float]] = None
    model_config = ConfigDict(from_attributes=True)

class InferenceResult(BaseModel):
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    predictions: Optional[list] = None
    info: Optional[Dict[str, Any]] = None
    images: List[str]
