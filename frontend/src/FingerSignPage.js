import React from "react";

export default function FingerSignPage() {
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
        ASL Finger-Sign Translator
      </h1>

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
          src="http://127.0.0.1:5001/finger_feed"
          alt="Finger Signing Stream"
          style={{ width: "100%", display: "block" }}
        />
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
