import { useState, DragEvent } from 'react';
import {
  Box, Button, Typography, CircularProgress, Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface Props {
  onUploaded: () => void;           // callback to refresh dataset list
}

export default function DatasetUploader({ onUploaded }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [msg, setMsg]           = useState<{ ok:boolean; text:string } | null>(null);

  const upload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setMsg({ ok:false, text:'Only .csv files are accepted' }); return;
    }
    setLoading(true);
    setMsg(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/datasets/upload', { method:'POST', body: form });
      if (!res.ok) throw new Error(`Server responded ${res.status}: ${res.statusText}`);
      setMsg({ ok: true, text: `Uploaded ${file.name}` });
      onUploaded();
    } catch (err) {
      console.error(err);
      setMsg({ ok: false, text: `Upload failed: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); e.stopPropagation(); setDragOver(false);
    if (e.dataTransfer.files?.[0]) upload(e.dataTransfer.files[0]);
  };

  const pickFile = () => {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = '.csv';
    input.onchange = () => input.files?.[0] && upload(input.files[0]);
    input.click();
  };

  return (
    <Box
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      sx={{
        border: '2px dashed',
        borderColor: dragOver ? 'primary.main' : 'grey.400',
        borderRadius: 2,
        p: 3, textAlign: 'center',
        backgroundColor: dragOver ? 'grey.100' : 'transparent',
        transition: 'background-color .2s',
      }}
    >
      <CloudUploadIcon sx={{ fontSize: 40, mb: 1 }} color="action" />
      <Typography>Arraste e solte um CSV ou</Typography>
      <Button onClick={pickFile} variant="outlined" sx={{ mt: 1 }}>Explorar Arquivos</Button>

      {loading && <CircularProgress size={24} sx={{ mt: 2 }} />}
      {msg && <Alert sx={{ mt:2 }} severity={msg.ok ? 'success' : 'error'}>{msg.text}</Alert>}
    </Box>
  );
}
