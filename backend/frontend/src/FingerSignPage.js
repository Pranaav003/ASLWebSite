import React, { useRef, useEffect, useState } from "react";
import axios from "axios";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

export default function FingerSignPage() {
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const [sentenceList, setSentenceList] = useState([]);

  // Refs to track the last recognized character and its timestamp
  const lastCharRef = useRef("");
  const lastTimeRef = useRef(0);

  // Bounding‐box overlay
  useEffect(() => {
    const hands = new Hands({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({
      maxNumHands:            1,
      modelComplexity:        0,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
      selfieMode:            true
    });
    hands.onResults((results) => {
      const vid    = videoRef.current;
      const canvas = overlayRef.current;
      const ctx    = canvas.getContext("2d");
      const w      = vid.clientWidth;
      const h      = vid.clientHeight;
      canvas.width  = w;
      canvas.height = h;
      canvas.style.width  = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.clearRect(0,0,w,h);
      ctx.save();
      ctx.translate(w,0);
      ctx.scale(-1,1);

      if (results.multiHandLandmarks?.length) {
        const lm   = results.multiHandLandmarks[0];
        const xs   = lm.map(pt => pt.x * w);
        const ys   = lm.map(pt => pt.y * h);
        const xMin = Math.min(...xs), xMax = Math.max(...xs);
        const yMin = Math.min(...ys), yMax = Math.max(...ys);
        const pad  = 10;
        ctx.strokeStyle="#0ff";
        ctx.lineWidth=4;
        ctx.strokeRect(
          xMin - pad, yMin - pad,
          (xMax - xMin) + pad*2,
          (yMax - yMin) + pad*2
        );
      }

      ctx.restore();
    });

    if (videoRef.current) {
      new Camera(videoRef.current, {
        onFrame: async () => await hands.send({ image: videoRef.current }),
        width: 640, height: 480
      }).start();
    }
  }, []);

  // Finger inference with hold‐to‐repeat logic (downsampled)
  useEffect(() => {
    let timer;
    const vid = videoRef.current;

    async function loop() {
      try { await vid.play(); } catch {}
      timer = setInterval(async () => {
        const W = 320;
        const H = vid.videoHeight * (320 / vid.videoWidth);
        const cnv = captureRef.current;
        cnv.width  = W;
        cnv.height = H;
        cnv.getContext("2d").drawImage(vid, 0, 0, W, H);
        const img = cnv.toDataURL("image/jpeg", 0.6);

        try {
          const res = await axios.post("/process_finger_frame", { image: img });
          const char = res.data.action || "";
          const now  = Date.now();

          if (char && char === lastCharRef.current) {
            // Same char as before: check hold duration
            if (now - lastTimeRef.current >= 1500) {
              setSentenceList(prev => [...prev, char]);
              lastTimeRef.current = now;
            }
          } else if (char) {
            // New char detected: append immediately
            setSentenceList(prev => [...prev, char]);
            lastCharRef.current  = char;
            lastTimeRef.current  = now;
          }
        } catch (e) {
          console.error("Finger inference error:", e);
        }
      }, 300);
    }

    loop();
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{
      display:       "flex",
      flexDirection: "column",
      alignItems:    "center",
      justifyContent: "center",
      padding:       "2vw",
      background:    "#111",
      color:         "#fff",
      minHeight:     "100vh",
      boxSizing:     "border-box",
      textAlign:     "center"
    }}>
      <h1 style={{ margin:0, fontSize:"clamp(1.5rem,4vw,3rem)" }}>
        ASL Alphabet Translator
      </h1>
      <p style={{
        margin: "1vh 0",
        fontSize: "clamp(1rem,2.5vw,1.25rem)",
        maxWidth: "800px",
        marginLeft: "auto",
        marginRight: "auto",
        color: "#ccc"
      }}>
        Welcome! Position your hand fully in view of your webcam to spell out letters A–Z. Each recognized letter appears below—hold a sign for 1.5 seconds to repeat it. To translate full-body ASL gestures instead, switch back via the navbar.
      </p>
      <img
        src="/ASLAlphabet.png.webp"
        alt="Placeholder"
        style={{
          width: "90vw",
          maxWidth: "600px",
          maxHeight: "400px",
          margin: "2vh 0",
          borderRadius: "8px"
        }}
      />

      <div style={{ position:"relative", display:"inline-block", marginTop:"2vh" }}>
        <video
          ref={videoRef}
          style={{ width:"90vw", maxWidth:"960px", borderRadius:"8px" }}
          playsInline
          muted
        />
        <canvas
          ref={overlayRef}
          style={{ position:"absolute", top:0, left:0, pointerEvents:"none" }}
        />
      </div>

      <canvas ref={captureRef} style={{ display:"none" }} />

      <div style={{
        marginTop:   "2vh",
        width:       "90vw",
        maxWidth:    "960px",
        background:  "#000",
        color:       "#fff",
        padding:     "1vh",
        fontSize:    "clamp(1rem,2.5vw,1.5rem)",
        borderRadius:"4px",
        textAlign:   "center"
      }}>
        {sentenceList.join("") || "Waiting for finger-sign…"}
      </div>

      <footer style={{ marginTop:40, fontSize:"clamp(0.875rem,2vw,1.1rem)", textAlign: "center" }}>
        <div>Made with ❤️</div>
        <div>Pranaav Iyer, Michael Nguyen, Oscar Primitivo, Nick Everett, Griffin Collins</div>
        <div>Reach out: pranaav.iyer@gmail.com, 408-863-2110</div>
        <div>LinkedIn: linkedin.com/in/pranaav-iyer/</div>
      </footer>
    </div>
  );
}
