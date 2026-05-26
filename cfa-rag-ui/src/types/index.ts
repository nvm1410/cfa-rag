export interface AuthResponse {
  token: string
  user_id: number
  username: string
}

export interface ChatSession {
  id: string
  title: string
  summary: string
  created_at: string
  updated_at: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface SessionDetail extends ChatSession {
  messages: Message[]
}

export interface ChatResponse {
  answer: string
  session_id: string
  relevant: boolean
}
