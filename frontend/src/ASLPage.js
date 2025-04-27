import React, { useState, useEffect } from "react";

export default function ASLPage() {
  const [gestures, setGestures] = useState([]);
  const [rawSummary, setRawSummary] = useState("");
  const [formattedLines, setFormattedLines] = useState([]);

  // fetch gestures
  useEffect(() => {
    fetch("http://127.0.0.1:5001/actions")
      .then((r) => r.json())
      .then(({ actions = [] }) => setGestures(actions))
      .catch(console.error);
  }, []);

  // fetch summary every 5s
  useEffect(() => {
    async function fetchSummary() {
      try {
        const res = await fetch("http://127.0.0.1:5001/summary");
        const { summary = "" } = await res.json();
        setRawSummary(summary);
        setFormattedLines(formatSummary(summary));
      } catch (err) {
        console.error("Summary fetch error:", err);
      }
    }
    fetchSummary();
    const iv = setInterval(fetchSummary, 5000);
    return () => clearInterval(iv);
  }, []);

  function formatSummary(raw) {
    let lines = raw
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l);
    while (
      lines.length &&
      !/^\d+\.\s*/.test(lines[0]) &&
      !/^[-–*]\s+/.test(lines[0])
    ) {
      lines.shift();
    }
    return lines.map((l, i) => {
      const text = l.replace(/^(\d+\.\s*|[-–*]\s*)/, "").trim();
      return `${i + 1}. ${text}`;
    });
  }

  return (
    <div
      style={{
        textAlign: "center",
        padding: 20,
        background: "#111",
        color: "#fff",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ fontSize: "clamp(1.5rem, 6vw, 3rem)" }}>
        AI-based ASL Translator
      </h1>
      <h2 style={{ fontSize: "clamp(1rem, 3vw, 1.5rem)", marginBottom: 20 }}>
        The model is trained on the following gestures:{" Hello, Thanks, iloveyou"}
      </h2>

      {/* Video */}
      <div
        style={{
          margin: "auto",
          width: "65vw",
          maxWidth: 1200,
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 0 10px rgba(0,0,0,0.5)",
        }}
      >
        <img
          src="http://127.0.0.1:5001/video_feed"
          alt="ASL Stream"
          style={{ width: "100%", display: "block" }}
        />
      </div>

      {/* Summary Header */}
      <div
        style={{
          fontSize: "clamp(1.25rem, 4vw, 2rem)",
          fontWeight: "bold",
          marginTop: 40,
        }}
      >
        LLM Suggestions:
      </div>

      {/* Each suggestion on its own line */}
      <div
        style={{
          fontSize: "clamp(1rem, 2.5vw, 1.5rem)",
          margin: "20px auto",
          maxWidth: "65vw",
          color: "#ddd",
          textAlign: "center",
          lineHeight: 1.8,
        }}
      >
        {formattedLines.length > 0 ? (
          formattedLines.map((line, i) => <div key={i}>{line}</div>)
        ) : (
          <div>Waiting for summary...</div>
        )}
      </div>

      <div style={{ marginTop: 40, fontSize: "clamp(0.875rem, 2vw, 1.1rem)" }}>
        Made with ❤️
      </div>
      <div style={{ fontSize: "clamp(0.875rem, 2vw, 1.1rem)" }}>
        Pranaav Iyer, Michael Nguyen, Oscar Primitivo, Nick Everett, Griffin Collins
      </div>
      <div style={{ marginTop: 10, fontSize: "clamp(0.875rem, 2vw, 1.1rem)" }}>
        Reach out: @pranaav.iyer@gmail.com
      </div>
      <div style={{ marginTop: 10, fontSize: "clamp(0.875rem, 2vw, 1.1rem)" }}>
        Connect with us: https://www.linkedin.com/in/pranaav-iyer/
      </div>
    </div>
  );
}