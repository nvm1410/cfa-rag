import type { AuthResponse, ChatSession, SessionDetail, ChatResponse } from '../types'

const BASE = ''

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  token?: string,
): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })

  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new ApiError(data.detail || `Request failed (${res.status})`, res.status)
  }

  if (res.status === 204) return undefined as T
  return res.json()
}

export const api = {
  register: (u: string, p: string) =>
    request<AuthResponse>('POST', '/auth/register', { username: u, password: p }),

  login: (u: string, p: string) =>
    request<AuthResponse>('POST', '/auth/login', { username: u, password: p }),

  listSessions: (token: string) =>
    request<ChatSession[]>('GET', '/sessions', undefined, token),

  createSession: (token: string, title = 'New Chat') =>
    request<ChatSession>('POST', '/sessions', { title }, token),

  getSession: (id: string, token: string) =>
    request<SessionDetail>('GET', `/sessions/${id}`, undefined, token),

  deleteSession: (id: string, token: string) =>
    request<void>('DELETE', `/sessions/${id}`, undefined, token),

  updateTitle: (id: string, title: string, token: string) =>
    request<ChatSession>('PATCH', `/sessions/${id}/title?title=${encodeURIComponent(title)}`, undefined, token),

  ask: (sessionId: string, question: string, token: string) =>
    request<ChatResponse>('POST', `/chat/sessions/${sessionId}/ask`, { question }, token),
}
