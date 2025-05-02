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
      v = Math.random().toString(36).substr(2,9);
      localStorage.setItem("clientId",v);
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
  const [probs,        setProbs       ] = useState([]);
  const [actionsList,  setActionsList ] = useState([]);
  const [lines,        setLines       ] = useState([]);

  // 1) Skeleton overlay
  useEffect(()=>{
    const holistic = new Holistic({
      locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
    });
    holistic.setOptions({
      modelComplexity:1,
      smoothLandmarks:true,
      refineFaceLandmarks:true,
      minDetectionConfidence:0.5,
      minTrackingConfidence:0.5,
      selfieMode:true
    });
    holistic.onResults(results=>{
      const vid = videoRef.current;
      const c   = overlayRef.current;
      const ctx = c.getContext("2d");
      const w   = vid.clientWidth, h = vid.clientHeight;
      c.width=w; c.height=h;
      c.style.width=`${w}px`; c.style.height=`${h}px`;
      ctx.clearRect(0,0,w,h);
      ctx.save();
      ctx.translate(w,0); ctx.scale(-1,1);
      if(results.faceLandmarks){
        drawConnectors(ctx,results.faceLandmarks,FACEMESH_TESSELATION,{color:"#888",lineWidth:1});
      }
      if(results.poseLandmarks){
        drawConnectors(ctx,results.poseLandmarks,POSE_CONNECTIONS,{color:"#0f0",lineWidth:4});
        drawLandmarks(ctx,results.poseLandmarks,{color:"#f00",lineWidth:2});
      }
      if(results.leftHandLandmarks){
        drawConnectors(ctx,results.leftHandLandmarks,HAND_CONNECTIONS,{color:"#c00",lineWidth:5});
        drawLandmarks(ctx,results.leftHandLandmarks,{color:"#0f0",lineWidth:2});
      }
      if(results.rightHandLandmarks){
        drawConnectors(ctx,results.rightHandLandmarks,HAND_CONNECTIONS,{color:"#00c",lineWidth:5});
        drawLandmarks(ctx,results.rightHandLandmarks,{color:"#0f0",lineWidth:2});
      }
      ctx.restore();
    });
    if(videoRef.current){
      new Camera(videoRef.current,{
        onFrame:async()=>await holistic.send({image:videoRef.current}),
        width:640,height:480
      }).start();
    }
  },[]);

  // 2) Inference: only start when video is ready
  useEffect(()=>{
    let intervalId;
    const vid = videoRef.current;
    const startLoop = () => {
      intervalId = setInterval(async()=>{
        if(vid.videoWidth===0||vid.videoHeight===0) return;
        const W=320;
        const H=vid.videoHeight*(W/vid.videoWidth);
        const cnv=captureRef.current;
        cnv.width=W; cnv.height=H;
        cnv.getContext("2d").drawImage(vid,0,0,W,H);
        const img=cnv.toDataURL("image/jpeg",0.6);
        const res=await axios.post("/process_frame",{image:img,clientId});
        setProbs(res.data.probabilities);
        const words = res.data.action.split(" ").filter(Boolean);
        // append only new words
        setSentenceList(prev=>
          words.length>prev.length ? words : prev
        );
      },300);
    };
    vid.addEventListener("playing", startLoop);
    return ()=>{
      vid.removeEventListener("playing", startLoop);
      clearInterval(intervalId);
    };
  },[clientId]);

  // load labels
  useEffect(()=>{
    fetch("/actions").then(r=>r.json()).then(d=>setActionsList(d.actions));
  },[]);

  // poll ChatGPT
  useEffect(()=>{
    const iv = setInterval(async()=>{
      const res = await fetch("/summary",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({sentences:sentenceList})
      });
      const {summary} = await res.json();
      const parsed = (summary||"").split("\n")
        .map(l=>l.trim())
        .filter(l=>/^\d+\.\s*/.test(l));
      setLines(parsed);
    },5000);
    return ()=>clearInterval(iv);
  },[sentenceList]);

  return (
    <div style={{padding:"2vw",background:"#111",color:"#fff",textAlign:"center"}}>
      <h1>AI-ASL Translator</h1>
      <h2>Trained on: {actionsList.join(", ")}</h2>
      <div style={{position:"relative",display:"inline-block",marginTop:20}}>
        <video ref={videoRef} style={{width:"90vw",maxWidth:960,borderRadius:8}} playsInline muted />
        <canvas ref={overlayRef} style={{position:"absolute",top:0,left:0,pointerEvents:"none"}}/>
      </div>
      <canvas ref={captureRef} style={{display:"none"}}/>
      <div style={{marginTop:20}}>
        <h3>Predicted:</h3>
        <p style={{fontSize:24}}>{sentenceList.join(" ")||"Waiting…"}</p>
        <p style={{fontSize:14,color:"#bbb"}}>[{probs.map(p=>p.toFixed(2)).join(", ")}]</p>
      </div>
      <div style={{marginTop:40}}>
        <h3>💡 Suggestions</h3>
        {lines.map((l,i)=><p key={i}>{l}</p>)}
      </div>
      <footer style={{ marginTop:40, fontSize:"clamp(0.875rem,2vw,1.1rem)" }}>
        <div>Made with ❤️</div>
        <div>Pranaav Iyer, Michael Nguyen, Oscar Primitivo, Nick Everett, Griffin Collins</div>
        <div>Reach out: pranaav.iyer@gmail.com</div>
        <div>LinkedIn: linkedin.com/in/pranaav-iyer/</div>
      </footer>
    </div>
  );
}
