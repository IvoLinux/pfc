# PFC WebApp - Network Traffic ML Platform

A minimal viable product (MVP) for fine-tuning and running inference on network traffic flow datasets.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Start the application**
   ```bash
   docker compose up --build -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Project Structure

```
/pfc-webapp/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── tasks.py
│   │   └── database.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.tsx
│   ├── Dockerfile
│   └── package.json
├── datasets/               # CSV datasets (volume)
│   ├── pfc_webapp.db       # SQLite database
├── models_output/          # Model checkpoints and results
└── docker-compose.yml
```

## Usage Guide

### 1. Upload Datasets
Navigate to the Training/Inference page and use the upload feature to add CSV files to the system.

### 2. Train a Model
- Go to the Training page
- Select a dataset from the dropdown
- Enter a model name (default: `distilbert-base-uncased`)
- Click "Start Training"
- Monitor progress in the Job Status table

### 3. Run Inference
- Go to the Inference page
- Select a trained model checkpoint
- Choose a dataset for inference
- Click "Run Inference"
- View results when the job completes

### 4. View Results
- Click the information icon in the Job Status table for completed inference jobs
- Review accuracy metrics and confusion matrix

## API Endpoints

| Method | Endpoint                       | Description                 |
|--------|--------------------------------|-----------------------------|
| POST   | `/api/jobs/train`              | Start a training job        |
| POST   | `/api/jobs/infer`              | Start an inference job      |
| GET    | `/api/jobs`                    | List all jobs               |
| GET    | `/api/jobs/{job_id}`           | Get job status              |
| GET    | `/api/datasets`                | List available datasets     |
| POST   | `/api/datasets/upload`         | Upload a new dataset        |
| GET    | `/api/models`                  | List trained models         |
| GET    | `/api/results/{job_id}`        | Get inference results       |

## Development

Can be done inside a docker container environment with --reload setting. See backend/Dockerfile
Otherwise, it can be run locally:

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## Potential issues

### Driver issues
- Install latest GPU drivers
- Ensure docker is configured to access the GPU
- Try launching mock containers that only try to access the GPU to troubleshoot what part is failing

### Container Issues
- Check logs: `docker compose logs -f [backend|frontend]`
- Restart services: `docker compose restart`
- Rebuild containers: `docker compose up --build`

### Database Issues
- Database is stored in `pfc_webapp.db`
- To reset: delete the file and restart backend

### Port Conflicts
- Frontend runs on port 3000
- Backend runs on port 8000
- Change ports in `docker-compose.yml` if needed
