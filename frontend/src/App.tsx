import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container, Box } from '@mui/material';
import TrainingPage from './pages/TrainingPage';
import InferencePage from './pages/InferencePage';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <Router>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Gerenciador de Aprendizado de Máquina
            </Typography>
            <Button color="inherit" component={Link} to="/train">
              Treinamento
            </Button>
            <Button color="inherit" component={Link} to="/infer">
              Inferência
            </Button>
          </Toolbar>
        </AppBar>
        <Container maxWidth="lg" sx={{ mt: 4 }}>
          <Routes>
            <Route path="/" element={<TrainingPage />} />
            <Route path="/train" element={<TrainingPage />} />
            <Route path="/infer" element={<InferencePage />} />
            <Route path="/results/:jobId" element={<ResultsPage />} />
          </Routes>
        </Container>
      </Box>
    </Router>
  );
}

export default App;