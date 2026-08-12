import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Train from './pages/Train'
import Predict from './pages/Predict'
import Evaluate from './pages/Evaluate'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-[#0f1117]">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/train" element={<Train />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/evaluate" element={<Evaluate />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
