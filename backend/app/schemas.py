from datetime import datetime
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, ConfigDict


# Payloads
class JobCreateTrain(BaseModel):
    kind: str                   # "tabular" | "llm"
    model_name: str
    dataset_filename: str


class JobCreateInfer(BaseModel):
    kind: str                   # "tabular" | "llm"
    dataset_filename: str
    checkpoint_filename: str             # *.joblib  or  *.pt


# Responses
class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    progress: int
    submitted_at: datetime
    metrics_json: Optional[Dict[str, float]] = None
    model_config = ConfigDict(from_attributes=True)


class InferenceResult(BaseModel):
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    predictions: Optional[list] = None
    info: Optional[Dict[str, Any]] = None
    plot_filename: Optional[str] = None
