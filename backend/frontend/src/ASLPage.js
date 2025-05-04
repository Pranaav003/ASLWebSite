import React, { useRef, useEffect, useState } from "react";
import axios from "axios";
import {
  Holistic,
  POSE_CONNECTIONS,
  HAND_CONNECTIONS,
  FACEMESH_TESSELATION
} from "@mediapipe/holistic";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

function useClientId() {
  const [id] = useState(() => {
    let v = localStorage.getItem("clientId");
    if (!v) {
      v = Math.random().toString(36).substr(2, 9);
      localStorage.setItem("clientId", v);
    }
    return v;
  });
  return id;
}

export default function ASLPage() {
  const clientId = useClientId();
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);

  const [sentenceList, setSentenceList] = useState([]);
  const [probs, setProbs] = useState([]);
  const [actionsList, setActionsList] = useState([]);
  const [lines, setLines] = useState([]);

  // Skeleton overlay
  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
    });
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      selfieMode: true
    });
    holistic.onResults((results) => {
      const vid = videoRef.current;
      const canvas = overlayRef.current;
      const ctx = canvas.getContext("2d");
      const w = vid.clientWidth;
      const h = vid.clientHeight;
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);

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
        drawLandmarks(ctx, results.poseLandmarks, { color: "#f00", lineWidth: 2 });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
          color: "#c00",
          lineWidth: 5
        });
        drawLandmarks(ctx, results.leftHandLandmarks, { color: "#0f0", lineWidth: 2 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
          color: "#00c",
          lineWidth: 5
        });
        drawLandmarks(ctx, results.rightHandLandmarks, { color: "#0f0", lineWidth: 2 });
      }
      ctx.restore();
    });

    if (videoRef.current) {
      new Camera(videoRef.current, {
        onFrame: async () => await holistic.send({ image: videoRef.current }),
        width: 640,
        height: 480
      }).start();
    }
  }, []);

  // Gesture inference (downscaled to 320px)
  useEffect(() => {
    let timer;
    const vid = videoRef.current;
    async function loop() {
      try {
        await vid.play();
      } catch {}
      timer = setInterval(async () => {
        const W = 320;
        const H = vid.videoHeight * (320 / vid.videoWidth);
        const cnv = captureRef.current;
        cnv.width = W;
        cnv.height = H;
        cnv.getContext("2d").drawImage(vid, 0, 0, W, H);
        const img = cnv.toDataURL("image/jpeg", 0.6);
        const res = await axios.post("/process_frame", {
          image: img,
          clientId
        });
        setProbs(res.data.probabilities);
        setSentenceList(res.data.action.split(" ").filter(Boolean));
      }, 300);
    }
    loop();
    return () => clearInterval(timer);
  }, [clientId]);

  // Load labels
  useEffect(() => {
    fetch("/actions")
      .then((r) => r.json())
      .then(({ actions }) => setActionsList(actions))
      .catch(console.error);
  }, []);

  // Poll ChatGPT
  useEffect(() => {
    let iv;
    async function fetchSummary() {
      const res = await fetch("/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: sentenceList })
      });
      const { summary } = await res.json();
      const parsed = (summary || "")
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => /^\d+\./.test(l));
      setLines(parsed);
    }
    fetchSummary();
    iv = setInterval(fetchSummary, 5000);
    return () => clearInterval(iv);
  }, [sentenceList]);

  return (
    <div
      style={{
        position: "relative",
        textAlign: "center",
        background: "#111",
        color: "#fff",
        padding: "2vw"
      }}
    >
      <h1 style={{ margin: 0, fontSize: "clamp(1.5rem,4vw,3rem)" }}>
        ASL Gesture Translator
      </h1>
      <p
        style={{
          margin: "1vh 0",
          fontSize: "clamp(1rem,2.5vw,1.25rem)",
          maxWidth: "800px",
          marginLeft: "auto",
          marginRight: "auto",
          color: "#ccc"
        }}
      >
        Welcome! Allow camera access, then sign naturally in front of your
        webcam—phrases like “hello” or “thank you” will appear in real time.
        Scroll down for example sentences generated by our language model. Ready
        to spell out words letter by letter? Switch to the Alphabet mode via
        the navbar!
      </p>
      <p
        style={{
          margin: "1vh 0",
          fontSize: "clamp(0.875rem,2vw,1rem)",
          maxWidth: "800px",
          marginLeft: "auto",
          marginRight: "auto",
          color: "#ccc"
        }}
      >
        IMPORTANT NOTE: Because we’re running on an entry-level $25 server,
        prediction speeds may be slower than optimal. If you’re interested in
        sponsoring the project, I’d love to hear from you—contact details are at
        the bottom of the page.
      </p>
      {/* Image placeholders inline */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "1vw", // Increased gap between images
          margin: "2vh 0",
          flexWrap: "wrap"
        }}
      >
        <div style={{ flex: 1, maxWidth: "150px", maxHeight: "100px", aspectRatio: "1/1" }}>
          <img
            src="/hello.png"
            alt="Example 1"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </div>
        <div style={{ flex: 1, maxWidth: "150px", maxHeight: "100px", aspectRatio: "1/1" }}>
          <img
            src="/iloveyou.png"
            alt="Example 2"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </div>
        <div style={{ flex: 1, maxWidth: "150px", maxHeight: "100px", aspectRatio: "1/1" }}>
          <img
            src="/thanks.png"
            alt="Example 3"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </div>
      </div>
      <h2
        style={{
          fontWeight: 400,
          fontSize: "clamp(1rem,2.5vw,1.5rem)"
        }}
      >
        Trained on: {actionsList.join(", ")}
      </h2>

      <div
        style={{
          position: "relative",
          display: "inline-block",
          marginTop: "2vh",
          width: "100%",
          maxWidth: "1000px"
        }}
      >
        <video
          ref={videoRef}
          style={{
            width: "100%",
            height: "auto",
            borderRadius: "8px"
          }}
          playsInline
          muted
        />
        <canvas
          ref={overlayRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
            width: "100%",
            height: "100%"
          }}
        />
      </div>

      <canvas ref={captureRef} style={{ display: "none" }} />

      <div
        style={{
          marginTop: "2vh",
          width: "90vw",
          maxWidth: "960px",
          margin: "auto"
        }}
      >
        <h3 style={{ fontSize: "clamp(1.25rem,3vw,2rem)" }}>Predicted Text:</h3>
        <p
          style={{
            fontSize: "clamp(1.5rem,4vw,2rem)",
            color: "#fff",
            margin: "0.5vh 0"
          }}
        >
          {sentenceList.join(" ") || "Waiting…"}
        </p>
        <p style={{ color: "#bbb" }}>
          [{probs.map((p) => p.toFixed(2)).join(", ")}]
        </p>
      </div>

      <div
        style={{
          marginTop: "4vh",
          width: "90vw",
          maxWidth: "900px",
          margin: "auto"
        }}
      >
        <h3
          style={{
            fontSize: "clamp(1.25rem,3vw,2rem)",
            marginBottom: "1vh"
          }}
        >
          💡 Suggestions:
        </h3>
        {lines.length > 0 ? (
          lines.map((l, i) => (
            <p
              key={i}
              style={{
                fontSize: "clamp(1rem,2.5vw,1.25rem)",
                margin: "0.5vh 0"
              }}
            >
              {l}
            </p>
          ))
        ) : (
          <p style={{ fontSize: "clamp(1rem,2.5vw,1.25rem)" }}>
            Waiting for suggestions…
          </p>
        )}
      </div>

      <footer
        style={{
          marginTop: 40,
          fontSize: "clamp(0.875rem,2vw,1.1rem)"
        }}
      >
        <div>Made with ❤️</div>
        <div>
          Pranaav Iyer, Michael Nguyen, Oscar Primitivo, Nick Everett, Griffin
          Collins
        </div>
        <div>Reach out: pranaav.iyer@gmail.com, 408-863-2110</div>
        <div>LinkedIn: linkedin.com/in/pranaav-iyer/</div>
        </footer>
      </div>
    );
  }
