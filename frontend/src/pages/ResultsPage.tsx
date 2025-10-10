import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box, Typography, Paper, Grid, Card, CardContent, Button,
  CircularProgress, Alert, Chip, Stack, Divider, Collapse, IconButton,
  Tooltip, ToggleButtonGroup, ToggleButton, Table, TableHead, TableRow,
  TableCell, TableBody
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import DownloadIcon from '@mui/icons-material/Download';

type MetricMap = Record<string, number>;

interface InferenceResult {
  metrics: MetricMap;                 // overall metrics (e.g., precision/recall/f1_score[/accuracy])
  confusion_matrix: number[][];       // counts
  predictions?: string[];             // optional
  images: string[];                   // unused now (we render our own SVG)
  info?: {
    date?: string;
    model_name?: string;
    checkpoint?: string;
    dataset?: string;
    total_samples?: number;
  } & Record<string, any>;
}

interface JobMeta {
  id: string;
  title?: string;
  kind: string;
  status: string;
  metrics_json?: {
    checkpoint?: string;         // "<jobId>/<file>.pt"
    checkpoint_title?: string;   // friendly title of training job
    checkpoint_file?: string;
  } & Record<string, any>;
}

type NormalizeMode = 'count' | 'row' | 'col' | 'all';

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<InferenceResult | null>(null);
  const [jobMeta, setJobMeta] = useState<JobMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showPreds, setShowPreds] = useState(false);
  const [norm, setNorm] = useState<NormalizeMode>('count');

  useEffect(() => { jobId && fetchRes(jobId); }, [jobId]);

  const fetchRes = async (id: string) => {
    try {
      const [resR, resJ] = await Promise.all([
        fetch(`/api/results/${id}`),
        fetch(`/api/jobs/${id}`)
      ]);
      if (!resR.ok) throw new Error('Failed to fetch results');
      if (resJ.ok) setJobMeta(await resJ.json());
      setResults(await resR.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally { setLoading(false); }
  };

  const ckptLabel =
    jobMeta?.metrics_json?.checkpoint_title ||
    jobMeta?.metrics_json?.checkpoint_file ||
    undefined;

  const labels = useMemo(() => {
    if (!results) return [];
    const cm = results.confusion_matrix;
    if (cm.length === 2 && cm[0]?.length === 2) return ['Benign', 'Anomaly'];
    const n = Math.max(cm.length, cm[0]?.length ?? 0);
    return Array.from({ length: n }, (_, i) => `C${i + 1}`);
  }, [results]);

  const perClass = useMemo(() => {
    if (!results) return [];
    return computePerClassMetrics(results.confusion_matrix, labels);
  }, [results, labels]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/infer')}>Back to Inference</Button>
      </Box>
    );
  }

  if (!results || !jobId) return null;

  return (
    <Box sx={{ pb: 6 }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3} flexWrap="wrap" gap={2}>
        <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/infer')}>Back</Button>
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 700, lineHeight: 1.1 }}>
              Resultados da Inferência
            </Typography>
            <Typography variant="h6" color="text.secondary">
              {jobMeta?.title || `Job ${jobId}`}
            </Typography>
            <Stack direction="row" spacing={1} mt={1} flexWrap="wrap">
              {ckptLabel && <Chip label={`Checkpoint: ${ckptLabel}`} />}
              {results.info?.dataset && <Chip label={`Dataset: ${results.info.dataset}`} />}
              {typeof results.info?.total_samples === 'number' && <Chip label={`Amostras: ${results.info.total_samples}`} />}
            </Stack>
          </Box>
        </Box>
        {results.info?.date && (
          <Typography color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
            Executado em {results.info.date}
          </Typography>
        )}
      </Box>

      {/* Actions */}
      <Stack direction="row" spacing={1} sx={{ mb: 3 }} flexWrap="wrap">
        <DownloadJsonButton jobId={jobId} />
        {/* Heatmap download button is inside the Heatmap card (needs ref to SVG) */}
        <DownloadCsvButton cm={results.confusion_matrix} labels={labels} />
      </Stack>

      {/* KPIs + Heatmap */}
      <Grid container spacing={3} alignItems="stretch">
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" gutterBottom>Métricas globais</Typography>
              <Grid container spacing={2}>
                {Object.entries(results.metrics).map(([k, v]) => (
                  <Grid item xs={12} sm={6} key={k}>
                    <KpiCard label={prettyMetric(k)} value={v} />
                  </Grid>
                ))}
              </Grid>
              <Divider sx={{ my: 2 }} />
              <Typography variant="body2" color="text.secondary">
                As métricas globais vêm do relatório do modelo. Use a matriz de confusão e a tabela por classe para diagnósticos finos.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%', display:'flex', flexDirection:'column' }}>
            <CardContent sx={{ flex: 1, display:'flex', flexDirection:'column' }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1} gap={2} flexWrap="wrap">
                <Typography variant="h5">Matriz de Confusão</Typography>
                <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                  <Typography variant="body2" color="text.secondary">Normalizar:</Typography>
                  <ToggleButtonGroup
                    size="small"
                    color="primary"
                    exclusive
                    value={norm}
                    onChange={(_, v) => v && setNorm(v)}
                  >
                    <ToggleButton value="count">Contagem</ToggleButton>
                    <ToggleButton value="row">% por linha</ToggleButton>
                    <ToggleButton value="col">% por coluna</ToggleButton>
                    <ToggleButton value="all">% global</ToggleButton>
                  </ToggleButtonGroup>
                </Box>
              </Box>

              <ConfusionHeatmapCard
                cm={results.confusion_matrix}
                labels={labels}
                normalize={norm}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Per-class metrics */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Typography variant="h6">Métricas por classe</Typography>
            <Typography variant="body2" color="text.secondary">
              Precision / Recall / F1 / Suporte
            </Typography>
          </Box>
          <PerClassTable rows={perClass} />
        </CardContent>
      </Card>

      {/* Predictions (optional) */}
      {results.predictions && results.predictions.length > 0 && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
              <Typography variant="h6">Amostra de previsões</Typography>
              <Tooltip title={showPreds ? 'Ocultar' : 'Mostrar'}>
                <IconButton onClick={() => setShowPreds(s => !s)} size="small">
                  {showPreds ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Tooltip>
            </Box>
            <Collapse in={showPreds}>
              <Divider sx={{ mb: 2 }} />
              <MonoList items={results.predictions.slice(0, 80)} />
              {results.predictions.length > 80 && (
                <Typography variant="caption" color="text.secondary">
                  (+{results.predictions.length - 80} linhas ocultas)
                </Typography>
              )}
            </Collapse>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

/* ========================= Helpers & subcomponents ========================= */

function prettyMetric(k: string) {
  const name = k.replace(/_/g, ' ');
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function KpiCard({ label, value }: { label: string; value: number }) {
  const isRatio = value >= 0 && value <= 1;
  const display = isRatio ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  return (
    <Paper elevation={0} variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
      <Typography color="text.secondary" sx={{ mb: 0.5, fontWeight: 500 }}>
        {label}
      </Typography>
      <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: 0.3 }}>
        {display}
      </Typography>
    </Paper>
  );
}

/* ---------- Confusion Matrix utilities ---------- */

function normalizeMatrix(cm: number[][], mode: NormalizeMode): { values: number[][]; format: (x:number)=>string } {
  if (mode === 'count') {
    return { values: cm.map(r => r.slice()), format: (x) => String(x) };
  }
  const rows = cm.length;
  const cols = cm[0]?.length ?? 0;
  const values: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0));

  if (mode === 'row') {
    for (let i = 0; i < rows; i++) {
      const sum = cm[i].reduce((a,b)=>a+b, 0) || 1;
      for (let j = 0; j < cols; j++) values[i][j] = cm[i][j] / sum;
    }
    return { values, format: (x) => `${(x*100).toFixed(1)}%` };
  }

  if (mode === 'col') {
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let i = 0; i < rows; i++) sum += cm[i][j];
      if (sum === 0) sum = 1;
      for (let i = 0; i < rows; i++) values[i][j] = cm[i][j] / sum;
    }
    return { values, format: (x) => `${(x*100).toFixed(1)}%` };
  }

  // 'all'
  const total = cm.flat().reduce((a,b)=>a+b, 0) || 1;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) values[i][j] = cm[i][j] / total;
  return { values, format: (x) => `${(x*100).toFixed(1)}%` };
}

function labelsFromCM(cm: number[][]): string[] {
  if (cm.length === 2 && cm[0]?.length === 2) return ['Benign', 'Anomaly'];
  const n = Math.max(cm.length, cm[0]?.length ?? 0);
  return Array.from({ length: n }, (_, i) => `C${i + 1}`);
}

function computePerClassMetrics(cm: number[][], labels: string[]) {
  // For each class k:
  // TP = cm[k][k]
  // FP = sum_i cm[i][k] - TP
  // FN = sum_j cm[k][j] - TP
  // TN = total - TP - FP - FN
  const n = labels.length;
  const totals = {
    byRow: cm.map(r => r.reduce((a,b)=>a+b, 0)),
    byCol: Array.from({length:n}, (_,j)=> cm.reduce((a,row)=>a+row[j], 0)),
    total: cm.flat().reduce((a,b)=>a+b, 0),
  };
  return labels.map((lab, k) => {
    const TP = cm[k][k];
    const FP = totals.byCol[k] - TP;
    const FN = totals.byRow[k] - TP;
    const TN = totals.total - TP - FP - FN;
    const prec = TP + FP === 0 ? 0 : TP / (TP + FP);
    const rec  = TP + FN === 0 ? 0 : TP / (TP + FN);
    const f1   = (prec + rec) === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    const support = totals.byRow[k];
    return { label: lab, precision: prec, recall: rec, f1, support, TP, FP, FN, TN };
  });
}

/* ---------- Heatmap Card (SVG + download) ---------- */

function ConfusionHeatmapCard({
  cm,
  labels,
  normalize,
}: {
  cm: number[][];
  labels: string[];
  normalize: NormalizeMode;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { values, format } = useMemo(() => normalizeMatrix(cm, normalize), [cm, normalize]);

  const palette = (t: number) => {
    // pleasant sequential from light to rich: #eaf3ff → #76d1c1 → #2da44e
    const clamp = (x:number)=> Math.max(0, Math.min(1,x));
    const c = (r:number,g:number,b:number)=> `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
    t = clamp(t);
    if (t < 0.5) {
      const u = t/0.5; // 0..1
      return c(234 + u*(118-234), 243 + u*(209-243), 255 + u*(193-255));
    } else {
      const u = (t-0.5)/0.5;
      return c(118 + u*(45-118), 209 + u*(164-209), 193 + u*(78-193));
    }
  };

  // determine max for coloring (use max of normalized values or counts)
  const flat = values.flat();
  const maxVal = Math.max(0.00001, ...flat);
  const darkText = (v:number)=> (v / maxVal) > 0.55;

  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const padding = 60;
  const cell = 44; // responsive via viewBox
  const width = padding + cols*cell + 24;
  const height = padding + rows*cell + 56;

  const downloadSvg = () => {
    if (!svgRef.current) return;
    const serializer = new XMLSerializer();
    const src = serializer.serializeToString(svgRef.current);
    const blob = new Blob([src], { type: 'image/svg+xml;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'confusion_matrix.svg';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
      <Box display="flex" justifyContent="flex-end" mb={1}>
        <Button size="small" startIcon={<DownloadIcon />} onClick={downloadSvg}>
          Baixar SVG
        </Button>
      </Box>
      <Box>
        <svg ref={svgRef} width="100%" viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: '100%' }}>
          {/* cells */}
          {values.map((row, i) =>
            row.map((val, j) => {
              const x = padding + j * cell;
              const y = padding + i * cell;
              const fill = palette(maxVal === 0 ? 0 : val / maxVal);
              const labelText = format(val);
              const textDark = darkText(val);
              return (
                <g key={`${i}-${j}`}>
                  <rect x={x} y={y} width={cell-4} height={cell-4} rx={8} ry={8} fill={fill} />
                  <text
                    x={x + (cell-4)/2}
                    y={y + (cell-4)/2 + 4}
                    textAnchor="middle"
                    fontSize={12}
                    fontWeight={700}
                    fill={textDark ? 'white' : '#1b1f24'}
                  >
                    {labelText}
                  </text>
                </g>
              );
            })
          )}

          {/* axis tick labels */}
          {labels.map((lab, j) => (
            <text
              key={`x-${j}`}
              x={padding + j*cell + (cell-4)/2}
              y={padding - 14}
              textAnchor="middle"
              fontSize={11}
              fill="#57606a"
            >
              {lab}
            </text>
          ))}

          {labels.map((lab, i) => (
            <text
              key={`y-${i}`}
              x={padding - 12}
              y={padding + i*cell + (cell-4)/2 + 4}
              textAnchor="end"
              fontSize={11}
              fill="#57606a"
            >
              {lab}
            </text>
          ))}

          {/* axis titles */}
          <text x={padding + (cols*cell)/2} y={height - 18} textAnchor="middle" fontSize={12} fill="#24292f">
            Predito
          </text>
          <text
            x={16}
            y={padding + (rows*cell)/2}
            textAnchor="middle"
            fontSize={12}
            fill="#24292f"
            transform={`rotate(-90, 16, ${padding + (rows*cell)/2})`}
          >
            Real
          </text>

          {/* legend */}
          <defs>
            <linearGradient id="cmLegend" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={palette(0)} />
              <stop offset="50%" stopColor={palette(0.5)} />
              <stop offset="100%" stopColor={palette(1)} />
            </linearGradient>
          </defs>
          <rect x={padding} y={padding + rows*cell + 10} width={cols*cell - 4} height={10} fill="url(#cmLegend)" rx={5}/>
          <text x={padding} y={padding + rows*cell + 28} fontSize={10} fill="#57606a">min</text>
          <text x={padding + (cols*cell - 4)} y={padding + rows*cell + 28} textAnchor="end" fontSize={10} fill="#57606a">
            {normalize === 'count' ? Math.max(0, ...cm.flat()) : '100%'}
          </text>
        </svg>
      </Box>
    </Paper>
  );
}

/* ---------- Per-class table ---------- */

function PerClassTable({ rows }: { rows: ReturnType<typeof computePerClassMetrics> }) {
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Classe</TableCell>
          <TableCell align="right">Precision</TableCell>
          <TableCell align="right">Recall</TableCell>
          <TableCell align="right">F1</TableCell>
          <TableCell align="right">Suporte</TableCell>
          <TableCell align="right">TP</TableCell>
          <TableCell align="right">FP</TableCell>
          <TableCell align="right">FN</TableCell>
          <TableCell align="right">TN</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows.map(r => (
          <TableRow key={r.label}>
            <TableCell>{r.label}</TableCell>
            <TableCell align="right">{(r.precision*100).toFixed(1)}%</TableCell>
            <TableCell align="right">{(r.recall*100).toFixed(1)}%</TableCell>
            <TableCell align="right">{(r.f1*100).toFixed(1)}%</TableCell>
            <TableCell align="right">{r.support}</TableCell>
            <TableCell align="right">{r.TP}</TableCell>
            <TableCell align="right">{r.FP}</TableCell>
            <TableCell align="right">{r.FN}</TableCell>
            <TableCell align="right">{r.TN}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

/* ---------- Downloads ---------- */

function DownloadJsonButton({ jobId }: { jobId: string }) {
  const onClick = async () => {
    const r = await fetch(`/api/results/${jobId}`);
    if (!r.ok) return;
    const blob = new Blob([await r.text()], { type: 'application/json;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `result_${jobId}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <Button startIcon={<DownloadIcon />} variant="outlined" size="small" onClick={onClick}>
      Baixar result.json
    </Button>
  );
}

function DownloadCsvButton({ cm, labels }: { cm: number[][]; labels: string[] }) {
  const onClick = () => {
    const header = ['Actual \\ Pred', ...labels].join(',');
    const rows = cm.map((r, i) => [labels[i], ...r].join(','));
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'confusion_matrix.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <Button startIcon={<DownloadIcon />} variant="outlined" size="small" onClick={onClick}>
      Baixar CM (.csv)
    </Button>
  );
}

/* ---------- Misc UI ---------- */

function MonoList({ items }: { items: (string | number)[] }) {
  return (
    <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, bgcolor: 'grey.50', maxHeight: 320, overflow: 'auto' }}>
      <Box component="pre" sx={{ m: 0, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace', fontSize: 13 }}>
        {items.map((x, i) => `${i + 1}. ${String(x)}\n`)}
      </Box>
    </Paper>
  );
}
