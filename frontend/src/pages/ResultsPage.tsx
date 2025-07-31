import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box, Typography, Paper, Grid, Card, CardContent, Button,
  CircularProgress, Alert
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

interface InferenceResult {
  metrics: Record<string, number>;
  confusion_matrix: number[][];
  predictions?: string[];
  images: string[];
}

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { jobId && fetchRes(jobId); }, [jobId]);

  const fetchRes = async (id: string) => {
    try {
      const r = await fetch(`/api/results/${id}`);
      if (!r.ok) throw new Error('Failed to fetch results');
      setResults(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally { setLoading(false); }
  };

  if (loading) return (
    <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
      <CircularProgress />
    </Box>
  );

  if (error) return (
    <Box>
      <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/infer')}>Back to Inference</Button>
    </Box>
  );

  if (!results) return null;

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/infer')}>Back</Button>
        <Box ml={2} display="flex" flexDirection="column">
          <Typography variant="h4">Resultados da inferência</Typography>
          <Typography variant="h6">Job {jobId}</Typography>
        </Box>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography variant="h5" gutterBottom>Métricas do modelo</Typography>
          <Grid container spacing={2}>
            {Object.entries(results.metrics).map(([k, v]) => (
              <Grid item xs={6} key={k}>
                <Card><CardContent>
                  <Typography color="textSecondary" gutterBottom>{k.replace(/_/g, ' ').toUpperCase()}</Typography>
                  <Typography variant="h4">{(v * 100).toFixed(1)}%</Typography>
                </CardContent></Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography variant="h5" gutterBottom>Matriz de Confusão</Typography>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
            {results.images.length > 0 && (
              <img
                src={`/api/inference-results/${jobId}/${results.images[0]}`}
                alt="Confusion Matrix"
                style={{ maxWidth: '100%' }}
              />
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
