import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Typography, TextField, Select, MenuItem, Button, FormControl,
  InputLabel, Paper, Alert, CircularProgress, Grid
} from '@mui/material';
import JobStatusTable from '../components/JobStatusTable';
import DatasetUploader from '../components/DatasetUploader';

interface Dataset { filename: string; size_mb: number }

export default function TrainingPage() {
  const [kind, setKind] = useState<'tabular' | 'llm'>('tabular');
  const [modelName, setModelName] = useState('distilbert-base-uncased');
  const [dataset, setDataset] = useState('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<{ ok:boolean; text:string } | null>(null);

  const loadDatasets = useCallback(() =>
    fetch('/api/datasets').then(r => r.json()).then(setDatasets), []);

  useEffect(() => { loadDatasets(); }, [loadDatasets]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true); setMsg(null);
    const res = await fetch('/api/jobs/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind,
        ...(kind === 'llm' && { model_name: modelName }),
        dataset_filename: dataset
      })
    });
    setLoading(false);
    if (res.ok) {
      setMsg({ ok: true, text: 'Training job queued!' });
      setDataset('');
    } else {
      setMsg({ ok: false, text: 'Failed to queue job' });
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Treinamento de modelos</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DatasetUploader onUploaded={loadDatasets}/>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p:3 }}>
            <form onSubmit={submit}>
              <Box sx={{ display:'flex', flexDirection:'column', gap:2 }}>
                <FormControl fullWidth>
                  <InputLabel>Família de Modelos</InputLabel>
                  <Select value={kind} label="Família de Modelos"
                          onChange={e => setKind(e.target.value as any)}>
                    <MenuItem value="tabular">Tabular (Random Forest)</MenuItem>
                    <MenuItem value="llm">LLM (DistilBERT)</MenuItem>
                  </Select>
                </FormControl>

                {kind === 'llm' && (
                  <TextField
                    label="Nome do Modelo"
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    required
                    fullWidth
                  />
                )}

                <FormControl fullWidth required>
                  <InputLabel>Conjunto de Dados</InputLabel>
                  <Select value={dataset} label="Conjunto de Dados"
                          onChange={e => setDataset(e.target.value)}>
                    {datasets.map(d => (
                      <MenuItem key={d.filename} value={d.filename}>{d.filename}</MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {msg && <Alert severity={msg.ok?'success':'error'}>{msg.text}</Alert>}

                <Button type="submit" variant="contained"
                  disabled={loading || !dataset}
                  startIcon={loading ? <CircularProgress size={20}/> : null}>
                  Iniciar Treinamento
                </Button>
              </Box>
            </form>
          </Paper>
        </Grid>
      </Grid>

      <JobStatusTable />
    </Box>
  );
}
