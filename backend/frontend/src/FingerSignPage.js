import React, { useRef, useEffect, useState } from "react";
import axios from "axios";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

export default function FingerSignPage() {
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const [sentenceList, setSentenceList] = useState([]);

  // 1) Bounding-box overlay
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
      canvas.width  = w; canvas.height = h;
      canvas.style.width  = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.clearRect(0,0,w,h);
      ctx.save();
      ctx.translate(w,0); ctx.scale(-1,1);

      if (results.multiHandLandmarks?.length) {
        const lm   = results.multiHandLandmarks[0];
        const xs   = lm.map(pt=>pt.x*w);
        const ys   = lm.map(pt=>pt.y*h);
        const xMin = Math.min(...xs), xMax = Math.max(...xs);
        const yMin = Math.min(...ys), yMax = Math.max(...ys);
        const pad  = 10;
        ctx.strokeStyle="#0ff"; ctx.lineWidth=4;
        ctx.strokeRect(
          xMin-pad, yMin-pad,
          (xMax-xMin)+pad*2,
          (yMax-yMin)+pad*2
        );
      }

      ctx.restore();
    });

    if (videoRef.current) {
      new Camera(videoRef.current, {
        onFrame: async ()=>await hands.send({ image: videoRef.current }),
        width:640, height:480
      }).start();
    }
  }, []);

  // 2) Finger inference + client‐state
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
        const res = await axios.post("/process_finger_frame", { image: img });
        const char = res.data.action || "";
        setSentenceList(prev => prev[prev.length-1] === char ? prev : [...prev, char]);
      }, 200);
    }
    loop();
    return ()=>clearInterval(timer);
  }, []);

  return (
    <div style={{
      display:"flex", flexDirection:"column", alignItems:"center",
      padding:"2vw", background:"#111", color:"#fff",
      minHeight:"100vh", boxSizing:"border-box"
    }}>
      <h1 style={{margin:0,fontSize:"clamp(1.5rem,4vw,3rem)"}}>
        ASL Finger-Sign Translator
      </h1>

      <div style={{
        position:"relative", display:"inline-block", marginTop:"2vh"
      }}>
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
        marginTop:"2vh", width:"90vw", maxWidth:"960px",
        background:"#000", color:"#0ff",
        padding:"1vh", fontSize:"clamp(1rem,2.5vw,1.5rem)",
        borderRadius:"4px", textAlign:"center"
      }}>
        {sentenceList.join("") || "Waiting for finger-sign…"}
      </div>
    </div>
  );
}
