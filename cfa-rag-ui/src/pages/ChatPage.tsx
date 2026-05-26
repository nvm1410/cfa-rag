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

  const handleSelect = useCallback((id: string) => {
    setMessages([])
    setError(null)
    setActiveId(id)
    setSidebarOpen(false)
  }, [])

  const handleNew = useCallback(async () => {
    if (!token) return
    const tempId = `new-${Date.now()}`
    const optimistic: ChatSession = {
      id: tempId,
      title: 'New Chat',
      summary: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }
    setSessions(prev => [optimistic, ...prev])
    setMessages([])
    setError(null)
    setActiveId(tempId)
    setSidebarOpen(false)
    try {
      const session = await api.createSession(token)
      setSessions(prev => prev.map(s => s.id === tempId ? session : s))
      if (activeId === tempId) setActiveId(session.id)
    } catch {
      setSessions(prev => prev.filter(s => s.id !== tempId))
      if (activeId === tempId) {
        setActiveId(null)
      }
    }
  }, [token, activeId])

  const handleDelete = useCallback(async (id: string) => {
    if (!token) return
    const prevSessions = sessions
    const prevActiveId = activeId
    // Optimistic update
    setSessions(prev => prev.filter(s => s.id !== id))
    if (activeId === id) {
      setActiveId(null)
      setMessages([])
    }
    try {
      await api.deleteSession(id, token)
    } catch {
      // Roll back on failure
      setSessions(prevSessions)
      if (prevActiveId !== null) setActiveId(prevActiveId)
    }
  }, [token, activeId, sessions])

  const handleRename = useCallback(async (id: string, title: string) => {
    if (!token) return
    // Optimistic update
    setSessions(prev => prev.map(s => s.id === id ? { ...s, title } : s))
    try {
      const updated = await api.updateTitle(id, title, token)
      setSessions(prev => prev.map(s => s.id === id ? updated : s))
    } catch {
      // Refresh sessions to revert on failure
      const refreshed = await api.listSessions(token)
      setSessions(refreshed)
    }
  }, [token])

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
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
        onRename={handleRename}
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
          fetching={loadingMessages}
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
