import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { globalCSS } from './styles.js'
import TrainMode from './routes/TrainMode.jsx'
import BuildMode from './routes/BuildMode.jsx'
import TestMode from './routes/TestMode.jsx'
import InferMode from './routes/InferMode.jsx'

const MODES = [
  { path: '/build', label: 'Build', icon: '◧' },
  { path: '/test', label: 'Test', icon: '⚙' },
  { path: '/train', label: 'Train', icon: '▶' },
  { path: '/infer', label: 'Infer', icon: '◎' },
]

export default function App() {
  return (
    <BrowserRouter>
      <style>{globalCSS}</style>
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 24px',
        gap: 0,
        maxWidth: 1280,
        margin: '0 auto',
      }}>

        {/* Header */}
        <header style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
            <h1 style={{
              fontFamily: 'Inter, sans-serif',
              fontWeight: 600,
              fontSize: 20,
              letterSpacing: '-0.03em',
              color: '#fff',
            }}>
              2D Robotics <span style={{ color: 'var(--gold)', fontWeight: 300 }}>Sandbox</span>
            </h1>
          </div>
        </header>

        {/* Mode navigation */}
        <nav className="mode-nav">
          {MODES.map(mode => (
            <NavLink
              key={mode.path}
              to={mode.path}
              className={({ isActive }) => `mode-tab ${isActive ? 'active' : ''}`}
            >
              {mode.icon} {mode.label}
            </NavLink>
          ))}
        </nav>

        {/* Route content */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/build" element={<BuildMode />} />
            <Route path="/test" element={<TestMode />} />
            <Route path="/train" element={<TrainMode />} />
            <Route path="/infer" element={<InferMode />} />
            <Route path="*" element={<Navigate to="/train" replace />} />
          </Routes>
        </div>

        {/* Footer */}
        <footer style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          paddingTop: 12, borderTop: '1px solid var(--border)',
          fontSize: 9, color: 'rgba(255,255,255,0.15)', letterSpacing: '0.08em',
          marginTop: 16,
        }}>
          <span>2D ROBOTICS SANDBOX</span>
          <span>BUILD · TEST · TRAIN · INFER</span>
        </footer>

      </div>
    </BrowserRouter>
  )
}
