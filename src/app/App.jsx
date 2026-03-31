import { useState, useCallback } from 'react'
import { HashRouter, Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom'
import { globalCSS } from './styles.js'
import TrainMode, { ENV_CONFIGS } from './routes/TrainMode.jsx'
import BuildMode from './routes/BuildMode.jsx'
import TestMode from './routes/TestMode.jsx'
import InferMode from './routes/InferMode.jsx'
import EnvSelector from '../components/EnvSelector.jsx'

const MODES = [
  { path: '/build', label: 'Build', icon: '◧' },
  { path: '/test', label: 'Test', icon: '⚙' },
  { path: '/train', label: 'Train', icon: '▶' },
  { path: '/infer', label: 'Infer', icon: '◎' },
]

function AppLayout() {
  const location = useLocation()
  const isTrainRoute = location.pathname === '/train'

  // Env state lifted here so header selector and TrainMode share it
  const [envType, setEnvType] = useState('terrain')
  const [trainingActive, setTrainingActive] = useState(false)

  // Build new handler — passed down to EnvSelector, TrainMode provides the sidebar open callback
  const [buildNewCallback, setBuildNewCallback] = useState(null)
  const registerBuildNew = useCallback((fn) => {
    setBuildNewCallback(() => fn)
  }, [])

  const handleEnvSelect = useCallback((key) => {
    setEnvType(key)
  }, [])

  return (
    <div style={{
      minHeight: '100vh',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      padding: isTrainRoute ? '12px 16px 0' : '20px 24px',
      gap: 0,
      maxWidth: isTrainRoute ? 'none' : 1280,
      margin: '0 auto',
      overflow: 'hidden',
    }}>

      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        paddingBottom: 0,
        flexShrink: 0,
      }}>
        <h1 style={{
          fontFamily: 'Inter, sans-serif',
          fontWeight: 600,
          fontSize: 20,
          letterSpacing: '-0.03em',
          color: '#fff',
          whiteSpace: 'nowrap',
        }}>
          2D Robotics <span style={{ color: 'var(--gold)', fontWeight: 300 }}>Sandbox</span>
        </h1>

        {/* Env selector — only on train route */}
        {isTrainRoute && (
          <EnvSelector
            envConfigs={ENV_CONFIGS}
            currentEnv={envType}
            onSelect={handleEnvSelect}
            disabled={trainingActive}
            onBuildNew={() => buildNewCallback?.()}
          />
        )}

        {/* Mode navigation */}
        <nav className="mode-nav" style={{ marginLeft: 'auto', marginBottom: 0, borderBottom: 'none' }}>
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
      </header>

      {/* Thin separator */}
      <div style={{ height: 1, background: 'var(--border)', flexShrink: 0 }} />

      {/* Route content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <Routes>
          <Route path="/build" element={<BuildMode />} />
          <Route path="/test" element={<TestMode />} />
          <Route path="/train" element={
            <TrainMode
              envType={envType}
              setEnvType={setEnvType}
              onTrainingStateChange={setTrainingActive}
              onRegisterBuildNew={registerBuildNew}
            />
          } />
          <Route path="/infer" element={<InferMode />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </div>

      {/* Footer — hidden on train route for max canvas space */}
      {!isTrainRoute && (
        <footer style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          paddingTop: 12, borderTop: '1px solid var(--border)',
          fontSize: 9, color: 'rgba(255,255,255,0.15)', letterSpacing: '0.08em',
          marginTop: 16, flexShrink: 0,
        }}>
          <span>2D ROBOTICS SANDBOX</span>
          <span>BUILD · TEST · TRAIN · INFER</span>
        </footer>
      )}
    </div>
  )
}

export default function App() {
  return (
    <HashRouter>
      <style>{globalCSS}</style>
      <AppLayout />
    </HashRouter>
  )
}
