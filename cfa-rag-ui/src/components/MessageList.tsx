import { useEffect, useRef } from 'react'
import type { Message } from '../types'

interface MessageListProps {
  messages: Message[]
  loading: boolean
  error: string | null
}

export default function MessageList({ messages, loading, error }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (error) {
    return (
      <div className="messages">
        <div className="message-error">{error}</div>
      </div>
    )
  }

  if (messages.length === 0) {
    return (
      <div className="messages">
        <div className="messages-empty">
          <h2>CFA RAG Chat</h2>
          <p>Ask a question about the CFA curriculum to get started.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="messages">
      {messages.map((m, i) => (
        <div key={i} className={`message message-${m.role}`}>
          <div className="message-avatar">{m.role === 'user' ? 'U' : 'A'}</div>
          <div className="message-content">{m.content}</div>
        </div>
      ))}
      {loading && (
        <div className="message message-assistant">
          <div className="message-avatar">A</div>
          <div className="message-content thinking">Thinking…</div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
