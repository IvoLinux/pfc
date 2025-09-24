import React, { useEffect, useState, useCallback } from 'react';
import {
  Typography, Button, FormControl, InputLabel, Select, MenuItem, Box, Grid
} from '@mui/material';
import JobStatusTable from '../components/JobStatusTable';
import DatasetUploader from '../components/DatasetUploader';

interface Model { filename:string; display_name:string; kind:'tabular'|'llm' }
interface Dataset { filename:string }
interface JobMeta { id:string; title?:string }

export default function InferencePage() {
  // Default to LLM since the Tabular option is commented out in the UI
  const [kind, setKind] = useState<'tabular'|'llm'>('llm');
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [checkpoint, setCheckpoint] = useState('');
  const [dataset, setDataset]       = useState('');
  const [jobs, setJobs] = useState<JobMeta[]>([]);

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

  // Load jobs to map LLM checkpoints -> human-friendly job titles
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/jobs');
        if (!r.ok) return;
        setJobs(await r.json());
      } catch (e) {
        console.error('Failed to load jobs:', e);
      }
    })();
  }, []);

  // Given an LLM model entry, extract its jobId and find the job title
  const labelForModel = (m: Model) => {
    if (m.kind !== 'llm') return m.display_name;
    // Backend lists LLM checkpoints as: "<jobId>/<checkpoint-file>.pt"
    const jobId = m.filename.split('/')[0];
    const job = jobs.find(j => j.id === jobId);
    return job?.title || m.display_name;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/jobs/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind, checkpoint_filename: checkpoint, dataset_filename: dataset })
      });
      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
      // Optionally: clear form / show toast
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
              <InputLabel>Família de Modelo</InputLabel>
              <Select
                value={kind}
                label="Família de Modelo"
                onChange={e => { setKind(e.target.value as any); setCheckpoint(''); }}>
                {/* <MenuItem value="tabular">Tabular</MenuItem> */}
                <MenuItem value="llm">LLM</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth required>
              <InputLabel>Checkpoint</InputLabel>
              <Select value={checkpoint} label="Checkpoint" onChange={e => setCheckpoint(e.target.value)}>
                {modelsOfKind.map(m => (
                  <MenuItem key={m.filename} value={m.filename}>
                    {labelForModel(m)}
                  </MenuItem>
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
