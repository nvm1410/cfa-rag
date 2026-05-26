import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { api } from '../api/client'
import type { AuthResponse } from '../types'

interface AuthContextValue {
  token: string | null
  username: string | null
  login: (u: string, p: string) => Promise<void>
  signup: (u: string, p: string) => Promise<void>
  logout: () => void
  busy: boolean
  error: string | null
}

const AuthContext = createContext<AuthContextValue | null>(null)

const TK = 'cfa_token'
const UK = 'cfa_username'

function handleAuth(res: AuthResponse) {
  localStorage.setItem(TK, res.token)
  localStorage.setItem(UK, res.username)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TK))
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem(UK))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const login = useCallback(async (u: string, p: string) => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.login(u, p)
      handleAuth(res)
      setToken(res.token)
      setUsername(res.username)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Login failed')
      throw e
    } finally {
      setBusy(false)
    }
  }, [])

  const signup = useCallback(async (u: string, p: string) => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.register(u, p)
      handleAuth(res)
      setToken(res.token)
      setUsername(res.username)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Signup failed')
      throw e
    } finally {
      setBusy(false)
    }
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem(TK)
    localStorage.removeItem(UK)
    setToken(null)
    setUsername(null)
    setError(null)
  }, [])

  return (
    <AuthContext.Provider value={{ token, username, login, signup, logout, busy, error }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be inside AuthProvider')
  return ctx
}
