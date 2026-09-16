import React from 'react'
import { BrowserRouter, Navigate, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Reports from './pages/Reports'
import Wiki from './pages/Wiki'
import VetLogs from './pages/VetLogs'
import CameraSetup from './pages/CameraSetup'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-[#0a0f0d]">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/monitor" element={<Predict />} />
            <Route path="/predict" element={<Navigate to="/" replace />} />
            <Route path="/history" element={<Reports />} />
            <Route path="/wiki" element={<Wiki />} />
            <Route path="/alerts" element={<VetLogs />} />
            <Route path="/setup" element={<CameraSetup />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
