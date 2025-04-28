import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import ASLPage from "./ASLPage";
import FingerSignPage from "./FingerSignPage";

export default function App() {
  return (
    <Router>
      <nav style={{ padding:10, background:"#222", display:"flex", justifyContent:"center" }}>
        <Link to="/" style={{ color:"#fff", marginRight:20, fontWeight:"bold" }}>ASL Translator</Link>
        <Link to="/fingersign" style={{ color:"#fff", fontWeight:"bold" }}>Finger Signing</Link>
      </nav>
      <Routes>
        <Route path="/" element={<ASLPage />} />
        <Route path="/fingersign" element={<FingerSignPage />} />
      </Routes>
    </Router>
  );
}
