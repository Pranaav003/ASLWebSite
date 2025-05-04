import React from "react";
import { BrowserRouter, Routes, Route, Link, Navigate } from "react-router-dom";
import ASLPage from "./ASLPage";
import FingerSignPage from "./FingerSignPage";

export default function App() {
  return (
    <BrowserRouter>
      <nav style={{
        display:        "flex",
        justifyContent: "center",
        gap:            "2rem",
        padding:        "1rem 0",
        background:     "#222",
        boxShadow:      "0 2px 5px rgba(0,0,0,0.5)"
      }}>
        <Link 
          to="/asl" 
          style={{
            ...linkStyle,
            padding: "0.5rem 1rem",
            border: "2px solid transparent",
            borderRadius: "5px",
          }}
          onMouseEnter={(e) => {
            e.target.style.color = "#00d4ff";
            e.target.style.textDecoration = "underline";
            e.target.style.borderColor = "#00d4ff";
            e.target.style.backgroundColor = "rgba(0, 212, 255, 0.2)";
          }}
          onMouseLeave={(e) => {
            e.target.style.color = "#fff";
            e.target.style.textDecoration = "none";
            e.target.style.borderColor = "transparent";
            e.target.style.backgroundColor = "transparent";
          }}
        >
          → ASL Gesture Model ←
        </Link>
        <Link 
          to="/finger" 
          style={{
            ...linkStyle,
            padding: "0.5rem 1rem",
            border: "2px solid transparent",
            borderRadius: "5px",
          }}
          onMouseEnter={(e) => {
            e.target.style.color = "#00d4ff";
            e.target.style.textDecoration = "underline";
            e.target.style.borderColor = "#00d4ff";
            e.target.style.backgroundColor = "rgba(0, 212, 255, 0.2)";
          }}
          onMouseLeave={(e) => {
            e.target.style.color = "#fff";
            e.target.style.textDecoration = "none";
            e.target.style.borderColor = "transparent";
            e.target.style.backgroundColor = "transparent";
          }}
        >
          → ASL Alphabet Model ←
        </Link>
      </nav>

      <div style={{
        padding:      "1rem",
        background:   "#111",
        minHeight:    "calc(100vh - 60px)",
        color:        "#fff"
      }}>
        <Routes>
          <Route path="/" element={<Navigate to="/asl" replace />} />
          <Route path="/asl" element={<ASLPage />} />
          <Route path="/finger" element={<FingerSignPage />} />
          {/* Fallback for unknown routes */}
          <Route path="*" element={<Navigate to="/asl" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

// simple shared link styles
const linkStyle = {
  color:       "#fff",
  textDecoration: "none",
  fontSize:    "1.1rem",
  fontWeight:  "500"
};
