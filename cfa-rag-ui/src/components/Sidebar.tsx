import type { ChatSession } from '../types'
import { useAuth } from '../context/AuthContext'

interface SidebarProps {
  sessions: ChatSession[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  open: boolean
  onClose: () => void
  loading: boolean
}

export default function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onDelete,
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
              <span className="sidebar-item-title">{s.title}</span>
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
