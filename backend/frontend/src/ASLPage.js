import React, { useState, useEffect } from "react";

export default function ASLPage() {
  const [actions, setActions] = useState([]);
  const [lines, setLines] = useState([]);

  // fetch actions once
  useEffect(() => {
    fetch("/actions")
      .then(r=>r.json())
      .then(({actions})=>setActions(actions))
      .catch(console.error);
  }, []);

  // poll summary
  useEffect(() => {
    async function getSum() {
      const res = await fetch("/summary");
      const {summary=""} = await res.json();
      const raw = summary.split("\n")
                         .map(l=>l.trim())
                         .filter(l=>l)
                         .filter(l=>/^\d+\.\s*/.test(l)||/^[-–*]\s+/.test(l));
      setLines(
        raw.map((l,i)=>`${i+1}. ${l.replace(/^(\d+\.\s*|[-–*]\s*)/,"").trim()}`)
      );
    }
    getSum();
    const iv = setInterval(getSum,5000);
    return ()=>clearInterval(iv);
  },[]);

  return (
    <div style={{textAlign:"center",padding:20,background:"#111",color:"#fff"}}>
      <h1 style={{fontSize:"clamp(2rem,5vw,4rem)"}}>AI-ASL Translator</h1>
      <h2>The model is trained on: {actions.join(", ")}</h2>
      <div style={{margin:"auto",width:"65vw",maxWidth:1200,overflow:"hidden",borderRadius:12}}>
        <img src="/video_feed" alt="ASL" style={{width:"100%"}} />
      </div>
      <h3 style={{marginTop:20}}>LLM Suggestions:</h3>
      <div style={{lineHeight:1.6}}>
        {lines.length? lines.map((l,i)=><div key={i}>{l}</div>)
                       : <div>Waiting for summary...</div>}
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