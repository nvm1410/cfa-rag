import { AuthProvider, useAuth } from './context/AuthContext'
import AuthPage from './pages/AuthPage'
import ChatPage from './pages/ChatPage'

function Shell() {
  const { token } = useAuth()
  return token ? <ChatPage /> : <AuthPage />
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  )
}
