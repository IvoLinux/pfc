import { useEffect, useState } from 'react';
import {
  Table, TableBody, TableCell, TableContainer, TableHead, IconButton,
  TableRow, Paper, LinearProgress, Link, Box, Tooltip, Typography
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close'
import SaveIcon from '@mui/icons-material/Save'
import DeleteIcon from '@mui/icons-material/Delete'
import { Link as RouterLink } from 'react-router-dom';

interface Job {
  id: string;
  kind: string;
  status: string;
  progress: number;
  submitted_at: string;
  metrics_json?: {
    current_loss?: number;
    current_epoch?: number;
    current_batch?: number;
    elapsed?: number;
    eta?: number;
    detailed_progress?: number;
  };
}

const formatTime = (secs: number) => {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return `${h>0?h+'h ':''}${m}m ${s}s`;
};

export default function JobStatusTable() {
  const [jobs, setJobs] = useState<Job[]>([]);

  const load = () => fetch('/api/jobs')
    .then(r => r.json()).then(setJobs);

  useEffect(() => { load(); const t = setInterval(load, 2000); return () => clearInterval(t); }, []);

  const prettyKind = (k: string) =>
    k.replace('_', ' → ').replace('tabular', 'Tabular').replace('llm', 'LLM');

  return (
    <>
      <Typography variant="h5" sx={{ mt:3 }}>Jobs</Typography>
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Tipo</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Ações</TableCell>
              <TableCell>Progresso</TableCell>
              <TableCell>Lançado em</TableCell>
              <TableCell>Resultado</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.map(j => (
              <TableRow key={j.id}>
                <TableCell>{prettyKind(j.kind)}</TableCell>
                <TableCell>{j.status}</TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    {j.kind === 'llm_train' && j.status === 'RUNNING' && (
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Salvar checkpoint" placement="top">
                          <IconButton size="small" color="info" onClick={() => saveCheckpoint(j.id)} aria-label="Salvar checkpoint">
                            <SaveIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip placement="top" title="Encerrar Job">
                          <IconButton size="small" color="error" onClick={() => stopJob(j.id)} aria-label="Encerrar job">
                            <CloseIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    )}
                    {j.status === 'FAILED' && (
                    <Tooltip title="Delete Job" placement="top">
                      <IconButton size="small" color="error" aria-label="delete job" onClick={() => deleteJob(j.id)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  </Box>
                </TableCell>
                <TableCell sx={{ minWidth: 220 }}>
                  {j.status === 'RUNNING' ? (
                    <>
                      <LinearProgress variant="determinate" value={j.progress} />
                      <Box sx={{ mt: 1, fontSize: '0.875rem' }}>
                        {(j.metrics_json?.detailed_progress ?? j.progress).toFixed(2)}%
                        {j.metrics_json?.current_loss != null && ` | loss: ${j.metrics_json.current_loss.toFixed(4)}`}
                        {j.metrics_json?.elapsed != null && j.metrics_json?.eta != null && ` | elapsed: ${formatTime(j.metrics_json.elapsed)}`}
                        {j.metrics_json?.elapsed != null && j.metrics_json?.eta != null && ` | ETA: ${formatTime(j.metrics_json.eta)}`}
                      </Box>
                    </>
                  ) : (
                    j.progress === 100 ? 'Concluído' : `${j.progress}%`
                  )}
                </TableCell>
                <TableCell>{new Date(j.submitted_at).toLocaleString()}</TableCell>
                <TableCell>
                  {j.kind.endsWith('_infer') && j.status === 'COMPLETED' && (
                    <Link component={RouterLink} to={`/results/${j.id}`}>View</Link>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
}

const saveCheckpoint = async (jobId: string) => {
  try {
    await fetch(`/api/jobs/${jobId}/checkpoint/save`, { method: 'POST' });
  } catch (err) {
    console.error('Failed to save checkpoint:', err);
  }
};

const stopJob = async (jobId: string) => {
  try {
    await fetch(`/api/jobs/${jobId}/checkpoint/stop`, { method: 'POST' });
  } catch (err) {
    console.error('Failed to stop job:', err);
  }
};

const deleteJob = async (jobId: string) => {
  if (window.confirm('Tem certeza que quer deletar esse job? Esta ação não pode ser desfeita.')) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Falha ao deletar job');
      window.location.reload();
    } catch (err) {
      console.error('Failed to delete job:', err);
    }
  }
}

