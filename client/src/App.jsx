import React from 'react';
import './App.css';
import HomePage from './features/components/HomePage';
import BlogPage from './features/components/Blog';
import HospitalPage from './features/components/HospitalCards';
import DetectionPage from './features/components/DetectionPage';
import Navbar from './features/components/Navbar';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/blogs" element={<BlogPage />} />
        <Route path="/hospitals" element={<HospitalPage />} />
        <Route path="/detect" element={<DetectionPage />} />
      </Routes>
    </Router>
  );
}
