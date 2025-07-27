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
      const r = await fetch(`${import.meta.env.VITE_API_URL}/api/results/${id}`);
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

  const classCount = results.confusion_matrix.length;

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/infer')}>Back</Button>
        <Typography variant="h4" sx={{ ml: 2 }}>Inference Results – Job {jobId?.slice(0, 8)}…</Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Typography variant="h5" gutterBottom>Model Metrics</Typography>
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
          <Typography variant="h5" gutterBottom>Confusion Matrix</Typography>
          <Paper sx={{ p: 2, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ border: '1px solid #ddd', padding: 8 }}></th>
                  {Array.from({ length: classCount }).map((_, j) => (
                    <th key={j} style={{ border: '1px solid #ddd', padding: 8 }}>
                      Pred {j}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td style={{ border: '1px solid #ddd', padding: 8, fontWeight: 'bold' }}>
                      Actual {i}
                    </td>
                    {row.map((cell, j) => (
                      <td key={j}
                        style={{
                          border: '1px solid #ddd', padding: 8, textAlign: 'center',
                          backgroundColor: cell === Math.max(...row) ? '#e8f5e9' : '#fff'
                        }}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
