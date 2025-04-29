// backend/frontend/src/ASLPage.js

import React, { useRef, useEffect, useState } from "react";
import axios from "axios";

// Mediapipe Holistic & drawing utils & camera helper
import {
  Holistic,
  POSE_CONNECTIONS,
  HAND_CONNECTIONS,
  FACEMESH_TESSELATION
} from "@mediapipe/holistic";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

export default function ASLPage() {
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);

  const [action,      setAction]      = useState("");
  const [probs,       setProbs]       = useState([]);
  const [actionsList, setActionsList] = useState([]);
  const [lines,       setLines]       = useState([]);

  // ── 1) Mediapipe skeleton overlay ─────────────────────────────
  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    holistic.setOptions({
      modelComplexity:        1,
      smoothLandmarks:        true,
      enableSegmentation:     false,
      refineFaceLandmarks:    true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
      selfieMode:            true
    });

    holistic.onResults((results) => {
      const video  = videoRef.current;
      const canvas = overlayRef.current;
      const ctx    = canvas.getContext("2d");

      // 1a) size overlay to displayed video
      const width  = video.clientWidth;
      const height = video.clientHeight;
      canvas.width  = width;
      canvas.height = height;
      canvas.style.width  = `${width}px`;
      canvas.style.height = `${height}px`;

      // 1b) clear
      ctx.clearRect(0, 0, width, height);

      // 1c) flip horizontally to match mirrored video
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);

      // 1d) draw landmarks
      if (results.faceLandmarks) {
        drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, {
          color: "#888",
          lineWidth: 1
        });
      }
      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
          color: "#0f0",
          lineWidth: 4
        });
        drawLandmarks(ctx, results.poseLandmarks, {
          color: "#f00",
          lineWidth: 2
        });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
          color: "#c00",
          lineWidth: 5
        });
        drawLandmarks(ctx, results.leftHandLandmarks, {
          color: "#0f0",
          lineWidth: 2
        });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
          color: "#00c",
          lineWidth: 5
        });
        drawLandmarks(ctx, results.rightHandLandmarks, {
          color: "#0f0",
          lineWidth: 2
        });
      }

      ctx.restore();
    });

    // 1e) start feeding video to Holistic
    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          await holistic.send({ image: videoRef.current });
        },
        width:  640,
        height: 480
      });
      camera.start();
    }
  }, []);

  // ── 2) Gesture inference loop (hidden canvas) ────────────────
  useEffect(() => {
    let timer;
    const vid = videoRef.current;

    async function captureAndPredict() {
      if (!vid || !captureRef.current) return;
      try {
        // draw current video frame to hidden canvas
        const W = vid.videoWidth;
        const H = vid.videoHeight;
        const cnv = captureRef.current;
        cnv.width  = W;
        cnv.height = H;
        cnv.getContext("2d").drawImage(vid, 0, 0, W, H);

        // send to your Flask API
        const img = cnv.toDataURL("image/jpeg", 0.6);
        const res = await axios.post("/process_frame", { image: img });
        setProbs(res.data.probabilities);
        setAction(res.data.action);
      } catch (e) {
        console.error("Inference error:", e);
      }
    }

    async function startLoop() {
      try {
        await vid.play();            // ensure play()
      } catch {}
      timer = setInterval(captureAndPredict, 200);
    }
    startLoop();
    return () => clearInterval(timer);
  }, []);

  // ── 3) Load gesture labels once ───────────────────────────────
  useEffect(() => {
    fetch("/actions")
      .then((r) => r.json())
      .then(({ actions }) => setActionsList(actions))
      .catch(console.error);
  }, []);

  // ── 4) Poll for ChatGPT suggestions every 5s ─────────────────
  useEffect(() => {
    let iv;
    async function getSummary() {
      try {
        const res  = await fetch("/summary");
        const body = await res.json();
        const parsed = (body.summary || "")
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => /^\d+\.\s*/.test(l));
        setLines(parsed);
      } catch (e) {
        console.error("Error fetching summary:", e);
      }
    }
    getSummary();
    iv = setInterval(getSummary, 5000);
    return () => clearInterval(iv);
  }, []);

  // ── Render ────────────────────────────────────────────────────
  return (
    <div style={{
      position:   "relative",
      textAlign:  "center",
      background: "#111",
      color:      "#fff",
      padding:    "2vw"
    }}>
      <h1 style={{ margin:0, fontSize:"clamp(1.5rem,4vw,3rem)", color: "#fff" }}>
        AI-ASL Translator
      </h1>
      <h2 style={{
        fontWeight:400,
        fontSize:  "clamp(1rem,2.5vw,1.5rem)",
        marginTop: "1vh",
        color: "#fff"
      }}>
        Trained on: {actionsList.join(", ")}
      </h2>

      {/* video + overlay */}
      <div style={{
        position:"relative",
        display: "inline-block",
        marginTop:"2vh"
      }}>
        <video
          ref={videoRef}
          style={{
            width:      "90vw",
            maxWidth:   "960px",
            borderRadius:"8px"
          }}
          playsInline
          muted
        />
        <canvas
          ref={overlayRef}
          style={{
            position:      "absolute",
            top:           0,
            left:          0,
            pointerEvents: "none"
          }}
        />
      </div>

      {/* hidden inference canvas */}
      <canvas ref={captureRef} style={{ display:"none" }} />

      {/* predictions */}
      <div style={{
        marginTop: "2vh",
        width:     "90vw",
        maxWidth:  "960px",
        margin:    "auto"
      }}>
        <h3 style={{ fontSize:"clamp(1.25rem,3vw,2rem)", color: "#fff" }}>Predicted Text:</h3>
        <p style={{
          fontSize:"clamp(1.5rem,4vw,2rem)",
          color:   "#fff",
          margin:  "0.5vh 0"
        }}>
          {action || "Waiting…"}
        </p>
        <p style={{ color:"#fff" }}>
          [{probs.map((p) => p.toFixed(2)).join(", ")}]
        </p>
      </div>

      {/* suggestions */}
      <div style={{
        marginTop: "4vh",
        width:     "90vw",
        maxWidth:  "900px",
        margin:    "auto"
      }}>
        <h3 style={{
          fontSize:    "clamp(1.25rem,3vw,2rem)",
          marginBottom:"1vh",
          color: "#fff"
        }}>💡 Suggestions:</h3>
        {lines.length > 0 ? (
          lines.map((l,i) => (
            <p key={i} style={{
              fontSize:"clamp(1rem,2.5vw,1.25rem)",
              margin:  "0.5vh 0",
              color: "#fff"
            }}>{l}</p>
          ))
        ) : (
          <p style={{ fontSize:"clamp(1rem,2.5vw,1.25rem)", color: "#fff" }}>
            Waiting for suggestions…
          </p>
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

