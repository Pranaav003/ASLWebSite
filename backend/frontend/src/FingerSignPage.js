import React, { useRef, useEffect, useState } from "react";
import axios from "axios";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

export default function FingerSignPage() {
  const videoRef   = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const [sentenceList, setSentenceList] = useState([]);
  const lastCharRef  = useRef("");
  const firstTimeRef = useRef(0);

  // bounding‐box
  useEffect(()=>{
    const hands = new Hands({
      locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({
      maxNumHands:1,
      modelComplexity:0,
      minDetectionConfidence:0.5,
      minTrackingConfidence:0.5,
      selfieMode:true
    });
    hands.onResults(res=>{
      const vid=videoRef.current, c=overlayRef.current, ctx=c.getContext("2d");
      const w=vid.clientWidth, h=vid.clientHeight;
      c.width=w; c.height=h;
      ctx.clearRect(0,0,w,h);
      ctx.save(); ctx.translate(w,0); ctx.scale(-1,1);
      if(res.multiHandLandmarks?.length){
        const lm=res.multiHandLandmarks[0],
              xs=lm.map(p=>p.x*w),
              ys=lm.map(p=>p.y*h),
              xMin=Math.min(...xs), xMax=Math.max(...xs),
              yMin=Math.min(...ys), yMax=Math.max(...ys),
              pad=10;
        ctx.strokeStyle="#0ff"; ctx.lineWidth=4;
        ctx.strokeRect(xMin-pad,yMin-pad,(xMax-xMin)+pad*2,(yMax-yMin)+pad*2);
      }
      ctx.restore();
    });
    if(videoRef.current){
      new Camera(videoRef.current,{
        onFrame:async()=>await hands.send({image:videoRef.current}),
        width:640,height:480
      }).start();
    }
  },[]);

  // inference + hold-to-repeat
  useEffect(()=>{
    let id;
    const vid=videoRef.current;
    const startLoop=()=>{
      id=setInterval(async()=>{
        if(vid.videoWidth===0) return;
        const W=320, H=vid.videoHeight*(W/vid.videoWidth);
        const c=captureRef.current;
        c.width=W; c.height=H;
        c.getContext("2d").drawImage(vid,0,0,W,H);
        const img=c.toDataURL("image/jpeg",0.6);
        const res=await axios.post("/process_finger_frame",{image:img});
        const char=res.data.action||"";
        const now=Date.now();
        if(char===lastCharRef.current){
          if(now-firstTimeRef.current>=1500){
            setSentenceList(prev=>[...prev,char]);
            firstTimeRef.current=now;
          }
        } else if(char){
          setSentenceList(prev=>[...prev,char]);
          lastCharRef.current=char;
          firstTimeRef.current=now;
        }
      },300);
    };
    vid.addEventListener("playing",startLoop);
    return ()=>{
      vid.removeEventListener("playing",startLoop);
      clearInterval(id);
    };
  },[]);

  return (
    <div style={{padding:"2vw",background:"#111",color:"#fff",textAlign:"center"}}>
      <h1>ASL Finger-Sign Translator</h1>
      <div style={{position:"relative",display:"inline-block",marginTop:20}}>
        <video ref={videoRef} style={{width:"90vw",maxWidth:960,borderRadius:8}} playsInline muted/>
        <canvas ref={overlayRef} style={{position:"absolute",top:0,left:0,pointerEvents:"none"}}/>
      </div>
      <canvas ref={captureRef} style={{display:"none"}}/>
      <div style={{
        marginTop:20,
        background:"#000",padding:10,
        fontSize:"clamp(1rem,2.5vw,1.5rem)",
        borderRadius:4
      }}>
        {sentenceList.join("")||"Waiting for finger-sign…"}
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
