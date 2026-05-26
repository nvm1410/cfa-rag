import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import type { ChatSession } from '../types'
import { useAuth } from '../context/AuthContext'

interface SidebarProps {
  sessions: ChatSession[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  open: boolean
  onClose: () => void
  loading: boolean
}

function EditableTitle({
  session,
  onRename,
}: {
  session: ChatSession
  onRename: (id: string, title: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(session.title)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  useEffect(() => {
    setDraft(session.title)
  }, [session.title])

  function save() {
    const t = draft.trim()
    if (t && t !== session.title) {
      onRename(session.id, t)
    } else {
      setDraft(session.title)
    }
    setEditing(false)
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') { e.preventDefault(); save() }
    if (e.key === 'Escape') { setDraft(session.title); setEditing(false) }
  }

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="sidebar-edit-input"
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={handleKeyDown}
        onClick={e => e.stopPropagation()}
      />
    )
  }

  return (
    <span
      className="sidebar-item-title"
      onDoubleClick={e => { e.stopPropagation(); setEditing(true) }}
    >
      {session.title}
    </span>
  )
}

export default function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
  open,
  onClose,
  loading,
}: SidebarProps) {
  const { username, logout } = useAuth()

  return (
    <>
      {open && <div className="sidebar-overlay" onClick={onClose} />}
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={onNew}>
            + New Chat
          </button>
        </div>

        <nav className="sidebar-list">
          {loading && sessions.length === 0 && (
            <p className="sidebar-empty">Loading…</p>
          )}
          {!loading && sessions.length === 0 && (
            <p className="sidebar-empty">No sessions yet</p>
          )}
          {sessions.map(s => (
            <div
              key={s.id}
              className={`sidebar-item ${s.id === activeId ? 'active' : ''}`}
              onClick={() => { onSelect(s.id); onClose() }}
            >
              <EditableTitle session={s} onRename={onRename} />
              <button
                className="sidebar-item-del"
                onClick={e => { e.stopPropagation(); onDelete(s.id) }}
                title="Delete session"
              >
                ×
              </button>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-user">{username}</span>
          <button className="logout-btn" onClick={logout}>
            Log out
          </button>
        </div>
      </aside>
    </>
  )
}
