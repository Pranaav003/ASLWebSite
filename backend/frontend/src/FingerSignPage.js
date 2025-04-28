import React from "react";

export default function FingerSignPage() {
  return (
    <div style={{textAlign:"center",padding:20,background:"#111",color:"#fff"}}>
      <h1 style={{fontSize:"clamp(2rem,5vw,4rem)"}}>ASL Finger-Sign Translator</h1>
      <div style={{margin:"auto",width:"65vw",maxWidth:1200,overflow:"hidden",borderRadius:12}}>
        <img src="/finger_feed" alt="Finger Sign" style={{width:"100%"}}/>
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
