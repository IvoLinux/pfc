import React, { useEffect, useState, useCallback } from 'react';
import {
  Typography, Button, FormControl, InputLabel, Select, MenuItem, Box, Grid
} from '@mui/material';
import JobStatusTable from '../components/JobStatusTable';
import DatasetUploader from '../components/DatasetUploader';

interface Model { filename:string; display_name:string; kind:'tabular'|'llm' }
interface Dataset { filename:string }

export default function InferencePage() {
  const [kind, setKind] = useState<'tabular'|'llm'>('tabular');
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [checkpoint, setCheckpoint] = useState('');
  const [dataset, setDataset]       = useState('');

  const loadDatasets = useCallback(async () => {
    try {
      const r = await fetch('/api/datasets');
      if (!r.ok) throw new Error(r.statusText);
      setDatasets(await r.json());
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  }, []);

  useEffect(() => {
  (async () => {
    try {
      const r = await fetch('/api/models');
      if (!r.ok) throw new Error(r.statusText);
      setModels(await r.json());
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  })();
    loadDatasets();
  }, [loadDatasets]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/jobs/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind, checkpoint_filename: checkpoint, dataset_filename: dataset })
      });
      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
      // you might want to reset your form here or show a notification
    } catch (err) {
      console.error('Failed to start inference:', err);
    }
  };

  const modelsOfKind = models.filter(m => m.kind === kind);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Inferência</Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <DatasetUploader onUploaded={loadDatasets}/>
        </Grid>

        <Grid item xs={12} md={8}>
          <Box component="form" onSubmit={submit}
               sx={{ display:'flex', flexDirection:'column', gap:2 }}>

            <FormControl fullWidth>
              <InputLabel>Família de Modelo</ InputLabel>
              <Select value={kind} label="Família de Modelo"
                      onChange={e => { setKind(e.target.value as any); setCheckpoint(''); }}>
                <MenuItem value="tabular">Tabular</MenuItem>
                <MenuItem value="llm">LLM</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Checkpoint</InputLabel>
              <Select value={checkpoint} label="Checkpoint"
                      onChange={e => setCheckpoint(e.target.value)}>
                {modelsOfKind.map(m => (
                  <MenuItem key={m.filename} value={m.filename}>{m.display_name}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Conjunto de dados</InputLabel>
              <Select value={dataset} label="Conjunto de dados"
                      onChange={e => setDataset(e.target.value)}>
                {datasets.map(d => (
                  <MenuItem key={d.filename} value={d.filename}>{d.filename}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button variant="contained" type="submit" disabled={!checkpoint||!dataset}>
              Iniciar Inferência
            </Button>
          </Box>
        </Grid>
      </Grid>

      <JobStatusTable />
    </Box>
  );
}
