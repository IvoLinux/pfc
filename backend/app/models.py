import datetime
from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict

Base = declarative_base()


class Job(Base):
    """
    One row per background task.  Four kinds are currently supported:

        • tabular_train   • tabular_infer
        • llm_train       • llm_infer
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)

    # what to do
    kind = Column(String, nullable=False)           # e.g. "llm_train"
    model_name = Column(String, nullable=False)     # "distilbert-base-uncased" or any

    # data & artefacts
    dataset_filename = Column(String, nullable=False)
    checkpoint_filename = Column(String, nullable=True)   # tabular  ➜  *.joblib
    checkpoint_dir = Column(String, nullable=True)        # llm      ➜  folder name
    result_path = Column(String, nullable=True)           # JSON produced by inference

    # live status
    status = Column(String, default="QUEUED")     # QUEUED|RUNNING|COMPLETED|FAILED
    progress = Column(Integer, default=0)         # 0-100 %
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow)

    # optional metrics captured by tasks.py (nice for admin UIs)
    metrics_json = Column(MutableDict.as_mutable(JSON), nullable=True, default=dict)
    confusion_matrix = Column(JSON, nullable=True)
