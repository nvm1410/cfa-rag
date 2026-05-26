import { useState, type FormEvent } from 'react'
import { useAuth } from '../context/AuthContext'

type Mode = 'login' | 'signup'

export default function AuthPage() {
  const { login, signup, busy } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)

    const u = username.trim()
    const p = password.trim()
    if (!u || !p) {
      setError('Please fill in all fields')
      return
    }
    if (u.length < 3) {
      setError('Username must be at least 3 characters')
      return
    }
    if (p.length < 4) {
      setError('Password must be at least 4 characters')
      return
    }

    try {
      if (mode === 'login') {
        await login(u, p)
      } else {
        await signup(u, p)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>CFA RAG</h1>
        <p className="auth-subtitle">Chat with your CFA curriculum</p>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => { setMode('login'); setError(null) }}
          >
            Login
          </button>
          <button
            className={`auth-tab ${mode === 'signup' ? 'active' : ''}`}
            onClick={() => { setMode('signup'); setError(null) }}
          >
            Sign Up
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={e => setUsername(e.target.value)}
            placeholder="alice"
            autoComplete="username"
            disabled={busy}
          />

          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            placeholder="••••••••"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            disabled={busy}
          />

          {error && <p className="auth-error">{error}</p>}

          <button type="submit" className="auth-submit" disabled={busy}>
            {busy ? 'Please wait…' : mode === 'login' ? 'Login' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  )
}
