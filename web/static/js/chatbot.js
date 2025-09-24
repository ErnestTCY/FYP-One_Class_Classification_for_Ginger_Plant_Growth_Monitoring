
(function(){
const RAW = (window.BACKEND || "").trim();
const API = RAW ? RAW.replace(/\/+$/,"") : ""; // empty => same origin
console.log("[Chat] Using API base:", API || "(same origin)");
  const panel = document.getElementById("chat-panel");
  const toggle = document.getElementById("chat-toggle");
  const chevron = document.getElementById("chat-chevron");
  const body = document.getElementById("chat-body");
  const input = document.getElementById("chat-text");
  const sendBtn = document.getElementById("chat-send");
  const weatherSwitch = document.getElementById("chat-weather-switch");
  const suggest = document.getElementById("chat-suggest");

  if (!panel || !toggle) return;

  // --- UI: expand/collapse ---------------------------------------------------
  let open = false;
  function setOpen(v){
    open = v;
    panel.style.display = open ? "block" : "none";
    chevron.textContent = open ? "▼" : "▲";
  }
  toggle.addEventListener("click", ()=> setOpen(!open));
  setOpen(false);

  // --- History handling (client-side) ---------------------------------------
  const KEY = "ginger_chat_history";
  function loadHist(){
    try{
      const raw = localStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : [];
    }catch(_){ return []; }
  }
  function saveHist(h){ localStorage.setItem(KEY, JSON.stringify(h)); }

  let history = loadHist(); // [{role:'user'|'assistant'|'system', content:'...'}]

  // Render all messages
  function render(){
    body.innerHTML = "";
    for(const m of history){
      const row = document.createElement("div");
      row.className = "msg " + (m.role === "user" ? "me" : (m.role==="assistant" ? "ai" : "sys"));
      const b = document.createElement("div");
      b.className = "bubble";
      b.textContent = m.content;
      row.appendChild(b);
      body.appendChild(row);
    }
    body.scrollTop = body.scrollHeight;
  }

  // Seed a friendly system message if first time
  if(history.length === 0){
    history.push({
      role: "system",
      content: "Hi! I'm your ginger-care assistant. Ask me about watering schedules, fertilizer, pests, or whether today's weather is suitable for activity."
    });
    saveHist(history);
  }
  render();

  // Suggestion chips
  const chips = [
    "How much should I water this week?",
    "What pests should I watch for now?",
    "Soil tips for VegetativeGrowth?",
    "Is today’s weather good for transplanting?",
  ];
  suggest.innerHTML = "";
  for(const c of chips){
    const btn = document.createElement("button");
    btn.className = "btn btn-sm btn-outline-secondary";
    btn.textContent = c;
    btn.addEventListener("click", ()=>{
      input.value = c;
      input.focus();
    });
    suggest.appendChild(btn);
  }

  // --- Send flow -------------------------------------------------------------
  async function send(){
    const text = (input.value || "").trim();
    if(!text) return;
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;

    history.push({role:"user", content:text});
    render(); saveHist(history);

    // Typing indicator
    const typing = {role:"assistant", content:"…"};
    history.push(typing);
    render();

    try{
      const payload = {
        messages: history.filter(m => m.role !== "assistant" || m.content !== "…"), // no dots
        include_weather: !!weatherSwitch?.checked
      };
      const res = await fetch(`${API}/api/chat`, {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify(payload)
      });
      const j = await res.json();
      // replace typing
      history.pop();
      if(!res.ok || !j.ok){
        history.push({role:"assistant", content: "Sorry, I couldn't reach the AI right now."});
      }else{
        history.push({role:"assistant", content: j.reply || "(no answer)"});
      }
    }catch(_){
      history.pop();
      history.push({role:"assistant", content: "Network error. Please try again."});
    }
    render(); saveHist(history);

    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e)=>{
    if(e.key === "Enter") { e.preventDefault(); send(); }
  });
})();
