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

// generate or persist a client ID
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
  const clientId   = useClientId();
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);

  const [sentenceList, setSentenceList] = useState([]);
  const [probs,        setProbs]        = useState([]);
  const [actionsList,  setActionsList]  = useState([]);
  const [lines,        setLines]        = useState([]);

  // 1) Mediapipe skeleton overlay
  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
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
      const vid = videoRef.current;
      const canvas = overlayRef.current;
      const ctx    = canvas.getContext("2d");
      const w      = vid.clientWidth;
      const h      = vid.clientHeight;
      canvas.width  = w; canvas.height = h;
      canvas.style.width  = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.clearRect(0,0,w,h);
      ctx.save();
      ctx.translate(w,0); ctx.scale(-1,1);

      if (results.faceLandmarks) {
        drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, { color:"#888", lineWidth:1 });
      }
      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color:"#0f0", lineWidth:4 });
        drawLandmarks (ctx, results.poseLandmarks, { color:"#f00", lineWidth:2 });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color:"#c00", lineWidth:5 });
        drawLandmarks (ctx, results.leftHandLandmarks, { color:"#0f0", lineWidth:2 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color:"#00c", lineWidth:5 });
        drawLandmarks (ctx, results.rightHandLandmarks, { color:"#0f0", lineWidth:2 });
      }

      ctx.restore();
    });

    if (videoRef.current) {
      new Camera(videoRef.current, {
        onFrame: async () => { await holistic.send({ image: videoRef.current }); },
        width: 640, height: 480
      }).start();
    }
  }, []);

  // 2) Gesture inference per‐client
  useEffect(() => {
    let timer;
    const vid = videoRef.current;
    async function loop() {
      try { await vid.play(); } catch {}
      timer = setInterval(async () => {
        const W = vid.videoWidth, H = vid.videoHeight;
        const cnv = captureRef.current;
        cnv.width = W; cnv.height = H;
        cnv.getContext("2d").drawImage(vid, 0, 0, W, H);
        const img = cnv.toDataURL("image/jpeg", 0.6);

        const res = await axios.post("/process_frame", {
          image: img,
          clientId
        });
        setProbs(res.data.probabilities);
        setSentenceList(res.data.action.split(" ").filter(Boolean));
      }, 200);
    }
    loop();
    return () => clearInterval(timer);
  }, [clientId]);

  // 3) Load labels
  useEffect(() => {
    fetch("/actions")
      .then(r => r.json())
      .then(({ actions }) => setActionsList(actions))
      .catch(console.error);
  }, []);

  // 4) Poll summary with client sentences
  useEffect(() => {
    let iv;
    async function fetchSummary() {
      const res = await fetch("/summary", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ sentences: sentenceList })
      });
      const { summary } = await res.json();
      const parsed = (summary||"")
        .split("\n").map(l=>l.trim()).filter(l=>/^\d+\.\s*/.test(l));
      setLines(parsed);
    }
    fetchSummary();
    iv = setInterval(fetchSummary, 5000);
    return () => clearInterval(iv);
  }, [sentenceList]);

  return (
    <div style={{
      position:"relative", textAlign:"center", background:"#111",
      color:"#fff", padding:"2vw"
    }}>
      <h1 style={{margin:0,fontSize:"clamp(1.5rem,4vw,3rem)"}}>AI-ASL Translator</h1>
      <h2 style={{
        fontWeight:400,fontSize:"clamp(1rem,2.5vw,1.5rem)",marginTop:"1vh"
      }}>
        Trained on: {actionsList.join(", ")}
      </h2>

      <div style={{position:"relative",display:"inline-block",marginTop:"2vh"}}>
        <video
          ref={videoRef}
          style={{width:"90vw",maxWidth:"960px",borderRadius:"8px"}}
          playsInline muted
        />
        <canvas
          ref={overlayRef}
          style={{position:"absolute",top:0,left:0,pointerEvents:"none"}}
        />
      </div>

      <canvas ref={captureRef} style={{display:"none"}}/>

      <div style={{
        marginTop:"2vh",width:"90vw",maxWidth:"960px",margin:"auto"
      }}>
        <h3 style={{fontSize:"clamp(1.25rem,3vw,2rem)"}}>Predicted Text:</h3>
        <p style={{
          fontSize:"clamp(1.5rem,4vw,2rem)",color:"#0f0",margin:"0.5vh 0"
        }}>
          {sentenceList.join(" ")||"Waiting…"}
        </p>
        <p style={{color:"#bbb"}}>
          [{probs.map(p=>p.toFixed(2)).join(", ")}]
        </p>
      </div>

      <div style={{
        marginTop:"4vh",width:"90vw",maxWidth:"900px",margin:"auto"
      }}>
        <h3 style={{
          fontSize:"clamp(1.25rem,3vw,2rem)",marginBottom:"1vh"
        }}>💡 Suggestions:</h3>
        {lines.length>0
          ? lines.map((l,i)=><p key={i} style={{
              fontSize:"clamp(1rem,2.5vw,1.25rem)",margin:"0.5vh 0"
            }}>{l}</p>)
          : <p style={{fontSize:"clamp(1rem,2.5vw,1.25rem)"}}>
              Waiting for suggestions…
            </p>
        }
      </div>
    </div>
  );
}
