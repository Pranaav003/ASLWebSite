// backend/frontend/src/FingerSignPage.js

import React, { useRef, useEffect, useState } from "react";
import axios from "axios";

// Mediapipe Hands + drawing & camera utils
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

export default function FingerSignPage() {
  const videoRef    = useRef(null);
  const overlayRef  = useRef(null);
  const captureRef  = useRef(null);
  const [sentence, setSentence] = useState("");

  // 1) MediaPipe Hands overlay with horizontal flip correction
  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) => 
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
      maxNumHands:            1,
      modelComplexity:        0,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
      selfieMode:            true
    });

    hands.onResults((results) => {
      const video  = videoRef.current;
      const canvas = overlayRef.current;
      const ctx    = canvas.getContext("2d");
      const width  = video.clientWidth;
      const height = video.clientHeight;

      // size overlay to video
      canvas.width  = width;
      canvas.height = height;
      canvas.style.width  = `${width}px`;
      canvas.style.height = `${height}px`;

      ctx.clearRect(0, 0, width, height);

      // flip overlay to counter video mirror
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length) {
        const lm = results.multiHandLandmarks[0];
        const xs = lm.map(pt => pt.x * width);
        const ys = lm.map(pt => pt.y * height);
        const xMin = Math.min(...xs), xMax = Math.max(...xs);
        const yMin = Math.min(...ys), yMax = Math.max(...ys);
        const padX = 10, padY = 10;

        ctx.strokeStyle = "#0ff";
        ctx.lineWidth   = 4;
        ctx.strokeRect(
          xMin - padX,
          yMin - padY,
          (xMax - xMin) + padX*2,
          (yMax - yMin) + padY*2
        );
      }

      ctx.restore();
    });

    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          await hands.send({ image: videoRef.current });
        },
        width:  640,
        height: 480
      });
      camera.start();
    }
  }, []);

  // 2) Inference loop unchanged
  useEffect(() => {
    let timer;
    const vid = videoRef.current;

    async function captureAndPredict() {
      if (!vid || !captureRef.current) return;
      const W   = vid.videoWidth;
      const H   = vid.videoHeight;
      const cnv = captureRef.current;
      cnv.width  = W;
      cnv.height = H;
      cnv.getContext("2d").drawImage(vid, 0, 0, W, H);
      const img = cnv.toDataURL("image/jpeg", 0.6);

      try {
        const res = await axios.post("/process_finger_frame", { image: img });
        setSentence(res.data.action);
      } catch (e) {
        console.error("Finger inference error:", e);
      }
    }

    async function startLoop() {
      try { await vid.play(); } catch {}
      timer = setInterval(captureAndPredict, 200);
    }
    startLoop();
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{
      display:       "flex",
      flexDirection: "column",
      alignItems:    "center",
      padding:       "2vw",
      background:    "#111",
      color:         "#fff",
      minHeight:     "100vh",
      boxSizing:     "border-box"
    }}>
      <h1 style={{ margin:0, fontSize:"clamp(1.5rem,4vw,3rem)", color: "#fff" }}>
        ASL Finger-Sign Translator
      </h1>

      {/* Video + flipped overlay */}
      <div style={{
        position:  "relative",
        display:   "inline-block",
        marginTop: "2vh"
      }}>
        <video
          ref={videoRef}
          style={{
            width:       "90vw",
            maxWidth:    "960px",
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

      {/* Hidden capture canvas */}
      <canvas ref={captureRef} style={{ display:"none" }} />

      {/* Sentence bar */}
      <div style={{
        marginTop:    "2vh",
        width:        "90vw",
        maxWidth:     "960px",
        background:   "#000",
        color:        "#fff",
        padding:      "1vh",
        fontSize:     "clamp(1rem,2.5vw,1.5rem)",
        borderRadius: "4px",
        textAlign:    "center"
      }}>
        {sentence || "Waiting for finger-sign…"}
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
