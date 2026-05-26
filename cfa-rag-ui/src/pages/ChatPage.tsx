import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { useAuth } from '../context/AuthContext'
import type { ChatSession, Message } from '../types'
import Sidebar from '../components/Sidebar'
import MessageList from '../components/MessageList'
import ChatInput from '../components/ChatInput'

export default function ChatPage() {
  const { token } = useAuth()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load sessions on mount
  useEffect(() => {
    if (!token) return
    setLoadingSessions(true)
    api.listSessions(token)
      .then(s => {
        setSessions(s)
        if (s.length > 0 && !activeId) {
          setActiveId(s[0].id)
        }
      })
      .catch(() => {})
      .finally(() => setLoadingSessions(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  // Load messages when active session changes
  useEffect(() => {
    if (!activeId || !token) {
      setMessages([])
      return
    }
    setLoadingMessages(true)
    setError(null)
    api.getSession(activeId, token)
      .then(detail => setMessages(detail.messages))
      .catch(() => setError('Failed to load messages'))
      .finally(() => setLoadingMessages(false))
  }, [activeId, token])

  const handleNew = useCallback(async () => {
    if (!token) return
    try {
      const session = await api.createSession(token)
      setSessions(prev => [session, ...prev])
      setActiveId(session.id)
      setSidebarOpen(false)
    } catch {
      // ignore
    }
  }, [token])

  const handleDelete = useCallback(async (id: string) => {
    if (!token) return
    try {
      await api.deleteSession(id, token)
      setSessions(prev => prev.filter(s => s.id !== id))
      if (activeId === id) {
        setActiveId(null)
        setMessages([])
      }
    } catch {
      // ignore
    }
  }, [token, activeId])

  const handleSend = useCallback(async (question: string) => {
    if (!token || !activeId) return

    // Optimistic user message
    const userMsg: Message = {
      role: 'user',
      content: question,
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])
    setSending(true)
    setError(null)

    try {
      const res = await api.ask(activeId, question, token)
      const assistantMsg: Message = {
        role: 'assistant',
        content: res.answer,
        created_at: new Date().toISOString(),
      }
      setMessages(prev => [...prev, assistantMsg])

      // Refresh sessions to pick up auto-title
      const updated = await api.listSessions(token)
      setSessions(updated)
    } catch {
      setError('Failed to get answer. Please try again.')
      // Remove optimistic message on error
      setMessages(prev => prev.filter(m => m.content !== question || m.role !== 'user'))
    } finally {
      setSending(false)
    }
  }, [token, activeId])

  return (
    <div className="chat-layout">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNew}
        onDelete={handleDelete}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        loading={loadingSessions}
      />

      <main className="main-panel">
        <header className="main-header">
          <button className="menu-btn" onClick={() => setSidebarOpen(true)}>
            ☰
          </button>
          <span className="main-title">
            {activeId
              ? sessions.find(s => s.id === activeId)?.title || 'Chat'
              : 'CFA RAG Chat'}
          </span>
        </header>

        <MessageList
          messages={messages}
          loading={sending}
          error={error}
        />

        <ChatInput
          onSend={handleSend}
          disabled={sending || !activeId || loadingMessages}
        />
      </main>
    </div>
  )
}
