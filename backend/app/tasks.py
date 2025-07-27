"""
Background tasks for both TABULAR *and* LLM model families.

Two job kinds are supported:
• "tabular_train" / "tabular_infer"
• "llm_train"     / "llm_infer"
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Callable

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from .database import SessionLocal
from .models import Job
from .ml_pipeline import infer_model, train_model
from .llm_pipeline import fine_tune_llm, infer_llm

# --------------------------------------------------------------------------- #
#                               PATHS                                         #
# --------------------------------------------------------------------------- #
DATASETS_DIR = "/datasets"
OUT_ROOT = "models_output"

os.makedirs(OUT_ROOT, exist_ok=True)


# --------------------------------------------------------------------------- #
#                          HELPER FUNCTIONS                                   #
# --------------------------------------------------------------------------- #
def _progress_cb_factory(job: Job, db: Session) -> Callable[[int], None]:
    def _cb(info):
        session = SessionLocal()
        try:
            j = session.get(Job, job.id)
            if isinstance(info, dict):
                # integer progress for the ProgressBar
                if "progress" in info:
                    j.progress = int(info["progress"])

                mj = j.metrics_json or {}
                if "loss" in info:
                    mj["current_loss"] = round(info["loss"], 4)
                if "epoch" in info:
                    mj["current_epoch"] = info["epoch"]
                if "batch" in info:
                    mj["current_batch"] = info["batch"]
                if "elapsed" in info:
                    mj["elapsed"] = round(info["elapsed"], 2)
                if "eta" in info and info["eta"] is not None:
                    mj["eta"] = round(info["eta"], 2)
                if "progress" in info:
                    mj["detailed_progress"] = round(info["progress"], 2)

                j.metrics_json = mj
            else:
                j.progress = int(info)
            flag_modified(j, "metrics_json")
            session.commit()    
        finally:
            session.close()
    return _cb


def _fail(job: Job, db: Session, exc: Exception):
    job.status = "FAILED"
    job.progress = 0
    job.result_path = None
    db.commit()
    tb = "".join(traceback.format_exception(exc))
    print(f"[JOB-ERROR] {job.id}\n{tb}")


# --------------------------------------------------------------------------- #
#                                DISPATCHERS                                  #
# --------------------------------------------------------------------------- #
def run_job(job_id: str):
    """
    Central entry-point for the BackgroundTasks pool. Selects the correct
    pipeline based on job.kind which must be one of:
        tabular_train, tabular_infer, llm_train, llm_infer
    """
    db: Session = SessionLocal()
    job = db.query(Job).get(job_id)
    try:
        if job is None:
            raise RuntimeError(f"Job {job_id} not found in DB")

        job.status = "RUNNING"
        db.commit()

        progress_cb = _progress_cb_factory(job, db)
        ds_path = os.path.join(DATASETS_DIR, job.dataset_filename)

        # -------------------------------------------------------- TABULAR ----
        if job.kind == "tabular_train":
            ckpt, metrics, cm = train_model(
                ds_path, job.model_name, OUT_ROOT, progress_cb
            )
            job.result_path = ckpt

        elif job.kind == "tabular_infer":
            metrics, cm, res = infer_model(
                os.path.join(OUT_ROOT, "checkpoints", job.checkpoint_filename),
                ds_path,
                OUT_ROOT,
                progress_cb,
            )
            job.result_path = res

        # ----------------------------------------------------------- LLM -----
        elif job.kind == "llm_train":
            ckpt, metrics, cm = fine_tune_llm(
                job.id, ds_path, job.model_name, OUT_ROOT, progress_cb
            )
            job.result_path = ckpt

        elif job.kind == "llm_infer":
            metrics, cm, res = infer_llm(
                os.path.join(OUT_ROOT, "llm_checkpoints", job.checkpoint_filename),
                ds_path,
                OUT_ROOT,
                progress_cb,
            )
            job.result_path = res
            # job.result_path = None
        # --------------------------------------------------------------------

        job.metrics_json = metrics
        job.confusion_matrix = cm
        job.status = "COMPLETED"
        job.progress = 100
        db.commit()
        print(f"[JOB-DONE] {job.kind} → {job.result_path}")

    except Exception as exc:
        _fail(job, db, exc)
    finally:
        db.close()
