"""
The Acity Oracle – Academic City Knowledge Assistant
============================================
Streamlit UI for the RAG chatbot.

Displays (Part D requirements):
  • Chat interface with conversation history
  • Retrieved context chunks with semantic, keyword, and combined scores
  • Exact prompt sent to the LLM (toggleable)
  • Pipeline execution log (toggleable)
  • User feedback buttons – 👍 / 👎 (Part G)

Run locally:
    streamlit run app.py

Author : Michael Nana Kwame Osei-Dei  (10022300168)
Course : IT3241 – Introduction to Artificial Intelligence (2026)
"""

import os
import sys
import logging
import textwrap

import streamlit as st
import streamlit.components.v1 as components

# ── Ensure src/ is importable as a package ───────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_APP_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)

from src.pipeline import RAGPipeline   # noqa: E402
from src.feedback import FeedbackStore # noqa: E402


def render_arena_widget() -> None:
    """
    Render an enhanced SVG-based academic animation banner.
    Theme: Matrix Rain (gold + green), holographic Ghanaian figure, 
    floating data cards, lightbulb pulses, and data stream animations.
    """
    modern_html = textwrap.dedent("""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <style>
            body { margin:0; padding:0; background:transparent; font-family:sans-serif; overflow:hidden; }
            .wrap { width:100%; max-width:1150px; margin:0 auto; position:relative; perspective:1200px; }
            .box {
              width:100%; aspect-ratio:1100/480; border-radius:16px; overflow:hidden;
              border:1px solid #1a4d2e; background:linear-gradient(180deg, #0a1a12 0%, #0d2818 50%, #06140c 100%);
              box-shadow: 0 0 40px rgba(0,200,120,0.15), inset 0 0 80px rgba(0,0,0,0.5);
            }
            svg { width:100%; height:100%; display:block; transform-style:preserve-3d; }
            .hud {
              position:absolute; right:14px; top:10px; background:rgba(0,40,20,0.85);
              border:1px solid #00ff88; border-radius:10px; padding:6px 12px; font-size:12px; 
              color:#00ff88; font-family:'Courier New',monospace; text-shadow:0 0 8px #00ff88;
            }
            canvas#matrix { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; opacity:0.35; }
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="box" id="box">
              <canvas id="matrix" width="1100" height="480"></canvas>
              <svg id="arena" viewBox="0 0 1100 480" preserveAspectRatio="xMidYMid meet">
                <defs>
                  <linearGradient id="holoGrad" x1="0" x2="1" y1="0" y2="1">
                    <stop offset="0%" stop-color="#00ff88" stop-opacity="0.9"/>
                    <stop offset="50%" stop-color="#00ccff" stop-opacity="0.7"/>
                    <stop offset="100%" stop-color="#ffdd00" stop-opacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="goldGrad" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stop-color="#ffd700"/>
                    <stop offset="100%" stop-color="#b8860b"/>
                  </linearGradient>
                  <radialGradient id="holoGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#00ff88" stop-opacity="0.8"/>
                    <stop offset="60%" stop-color="#00ccff" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="#000" stop-opacity="0"/>
                  </radialGradient>
                  <radialGradient id="energyRing" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#ffdd00" stop-opacity="0.9"/>
                    <stop offset="100%" stop-color="#ff8800" stop-opacity="0"/>
                  </radialGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                  <filter id="holoGlowFilter">
                    <feGaussianBlur stdDeviation="4" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                </defs>
                <rect x="0" y="0" width="1100" height="480" fill="transparent"/>
                <g id="layerBg"></g>
                <g id="layerDataCards"></g>
                <g id="layerCharacter"></g>
                <g id="layerEnergy"></g>
                <g id="layerDataStream"></g>
                <g id="layerFx"></g>
              </svg>
            </div>
            <div id="hud" class="hud">Mode: Idle | Data: 0%</div>
          </div>
          <script>
            const NS = "http://www.w3.org/2000/svg";
            const W = 1100, H = 480, TAU = Math.PI * 2;
            const box = document.getElementById("box");
            const canvas = document.getElementById("matrix");
            const ctx = canvas.getContext("2d");
            const bgLayer = document.getElementById("layerBg");
            const dataCardsLayer = document.getElementById("layerDataCards");
            const charLayer = document.getElementById("layerCharacter");
            const energyLayer = document.getElementById("layerEnergy");
            const dataStreamLayer = document.getElementById("layerDataStream");
            const fxLayer = document.getElementById("layerFx");
            const hud = document.getElementById("hud");

            // Matrix Rain Setup
            const matrixChars = "アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプエェケセテネヘメレヱゲゼデベペオォコソトノホモヨョロヲゴゾドボポヴッン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            const goldChars = "₵GHANAオアシタナハマヤラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプ";
            const fontSize = 16;
            const columns = Math.floor(W / fontSize);
            const drops = Array(columns).fill(1);
            const dropsGold = Array(Math.floor(columns/3)).fill(1);

            function drawMatrix() {
              ctx.fillStyle = "rgba(10, 26, 18, 0.08)";
              ctx.fillRect(0, 0, W, H);
              const speed = window.isThinking ? 1.5 : (window.isTyping ? 1.2 : 1.0);
              
              for (let i = 0; i < drops.length; i++) {
                if (Math.random() > 0.97) {
                  const char = matrixChars[Math.floor(Math.random() * matrixChars.length)];
                  ctx.fillStyle = `rgba(0, ${Math.floor(180 + Math.random() * 75)}, ${Math.floor(100 + Math.random() * 56)}, ${0.4 + Math.random() * 0.4})`;
                  ctx.font = fontSize + "px monospace";
                  ctx.fillText(char, i * fontSize, drops[i] * fontSize);
                  if (drops[i] * fontSize > H && Math.random() > 0.975) drops[i] = 0;
                  drops[i] += speed;
                }
              }
              
              for (let i = 0; i < dropsGold.length; i++) {
                if (Math.random() > 0.985) {
                  const char = goldChars[Math.floor(Math.random() * goldChars.length)];
                  ctx.fillStyle = `rgba(255, ${Math.floor(180 + Math.random() * 75)}, 0, ${0.5 + Math.random() * 0.3})`;
                  ctx.font = (fontSize + 4) + "px monospace";
                  ctx.fillText(char, i * fontSize * 3, dropsGold[i] * fontSize);
                  if (dropsGold[i] * fontSize > H && Math.random() > 0.98) dropsGold[i] = 0;
                  dropsGold[i] += speed * 0.8;
                }
              }
            }

            const rand = (a, b) => a + Math.random() * (b - a);
            const el = (t, a = {}) => {
              const n = document.createElementNS(NS, t);
              for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
              return n;
            };

            // Background grid
            for (let x = 0; x <= W; x += 40) bgLayer.appendChild(el("line", {x1:x, y1:0, x2:x, y2:H, stroke:"#00ff88", "stroke-opacity":"0.08", "stroke-width":"1"}));
            for (let y = 0; y <= H; y += 40) bgLayer.appendChild(el("line", {x1:0, y1:y, x2:W, y2:y, stroke:"#00ff88", "stroke-opacity":"0.08", "stroke-width":"1"}));

            // Ghana subtle outline
            const ghanaPath = el("path", {
              d: "M280 120 L310 110 L350 118 L390 105 L430 125 L450 150 L480 170 L475 200 L495 230 L465 255 L435 268 L400 300 L360 310 L310 290 L280 295 L250 275 L230 245 L210 218 L225 190 L215 155 L235 128 Z",
              fill: "none", stroke: "#00ff88", "stroke-opacity": "0.15", "stroke-width": "1.5", "stroke-dasharray": "4 4"
            });
            bgLayer.appendChild(ghanaPath);

            // Holographic Ghanaian Character
            const holoCenterX = 550, holoCenterY = 240;
            
            // Holographic base glow
            const holoBase = el("ellipse", {cx: holoCenterX, cy: 380, rx: 80, ry: 20, fill: "url(#holoGlow)", opacity: 0.6, filter: "url(#holoGlowFilter)"});
            
            // Character body (holographic style)
            const bodyGlow = el("ellipse", {cx: holoCenterX, cy: holoCenterY, rx: 70, ry: 90, fill: "url(#holoGrad)", opacity: 0.25, filter: "url(#holoGlowFilter)"});
            
            // Robe (kente-inspired pattern)
            const robe = el("path", {
              fill: "none", stroke: "url(#holoGrad)", "stroke-width": "3", opacity: 0.9,
              d: `M${holoCenterX-45} ${holoCenterY+50} Q${holoCenterX} ${holoCenterY+120} ${holoCenterX+45} ${holoCenterY+50}`
            });
            
            // Head
            const headGlow = el("circle", {cx: holoCenterX, cy: holoCenterY-45, r: 35, fill: "url(#holoGrad)", opacity: 0.3, filter: "url(#holoGlowFilter)"});
            const head = el("circle", {cx: holoCenterX, cy: holoCenterY-45, r: 22, fill: "none", stroke: "#00ff88", "stroke-width": 2, opacity: 0.85});
            
            // Eyes (holographic)
            const eyeL = el("ellipse", {cx: holoCenterX-8, cy: holoCenterY-48, rx: 4, ry: 2, fill: "#00ff88", opacity: 0.9});
            const eyeR = el("ellipse", {cx: holoCenterX+8, cy: holoCenterY-48, rx: 4, ry: 2, fill: "#00ff88", opacity: 0.9});
            
            // Crown (Adinkra-inspired)
            const crown = el("path", {
              fill: "none", stroke: "#ffd700", "stroke-width": 2, opacity: 0.9,
              d: `M${holoCenterX-25} ${holoCenterY-70} L${holoCenterX} ${holoCenterY-85} L${holoCenterX+25} ${holoCenterY-70}`
            });
            
            // Arms (holographic lines)
            const leftArm = el("line", {x1: holoCenterX-30, y1: holoCenterY, x2: holoCenterX-60, y2: holoCenterY+30, stroke: "#00ccff", "stroke-width": 2, opacity: 0.7});
            const rightArm = el("line", {x1: holoCenterX+30, y1: holoCenterY, x2: holoCenterX+60, y2: holoCenterY+30, stroke: "#00ccff", "stroke-width": 2, opacity: 0.7});
            
            // Data orbs around character
            const orbs = [];
            for (let i = 0; i < 6; i++) {
              const orb = el("circle", {r: rand(4, 8), fill: i % 2 === 0 ? "#00ff88" : "#ffd700", opacity: 0.7, filter: "url(#glow)"});
              dataCardsLayer.appendChild(orb);
              orbs.push({el: orb, angle: rand(0, TAU), radius: rand(60, 100), speed: rand(0.3, 0.8)});
            }

            // Floating holographic data cards
            const dataCards = [];
            const cardLabels = ["GHANA", "VOTE", "DATA", "2024", "REGION", "RESULT"];
            for (let i = 0; i < 5; i++) {
              const cardGroup = el("g", {filter: "url(#holoGlowFilter)"});
              const card = el("rect", {x: -35, y: -20, width: 70, height: 40, rx: 4, fill: "rgba(0,40,20,0.7)", stroke: "#00ff88", "stroke-width": 1, opacity: 0.8});
              const label = el("text", {x: 0, y: 5, fill: "#00ff88", "font-size": "10", "text-anchor": "middle", "font-family": "monospace"});
              label.textContent = cardLabels[i];
              cardGroup.appendChild(card);
              cardGroup.appendChild(label);
              dataCardsLayer.appendChild(cardGroup);
              dataCards.push({
                g: cardGroup, 
                bx: rand(150, 950), 
                by: rand(80, 350), 
                amp: rand(8, 25), 
                ph: rand(0, TAU), 
                orbit: rand(0.2, 0.6),
                orbitRadius: rand(30, 60)
              });
            }

            // Data flow lines
            const dataLines = [];
            for (let i = 0; i < 8; i++) {
              const line = el("path", {
                fill: "none", stroke: i % 2 === 0 ? "#00ff88" : "#ffd700",
                "stroke-width": 1, opacity: 0.4, "stroke-dasharray": "3 6"
              });
              dataCardsLayer.appendChild(line);
              dataLines.push({el: line, offset: rand(0, 100), speed: rand(0.5, 1.5)});
            }

            // Lightbulb (thinking state)
            const lightbulb = el("g", {opacity: 0});
            const bulb = el("path", {
              fill: "#ffd700", opacity: 0.9,
              d: "M0,-20 C-12,-20 -18,-10 -18,0 C-18,12 -10,20 0,20 C10,20 18,12 18,0 C18,-10 12,-20 0,-20"
            });
            const bulbBase = el("rect", {x: -6, y: 20, width: 12, height: 6, fill: "#888"});
            const bulbGlow = el("circle", {cx: 0, cy: 0, r: 30, fill: "url(#energyRing)", opacity: 0.5});
            lightbulb.appendChild(bulbGlow);
            lightbulb.appendChild(bulb);
            lightbulb.appendChild(bulbBase);
            energyLayer.appendChild(lightbulb);

            // Energy rings (thinking state)
            const energyRings = [];
            for (let i = 0; i < 3; i++) {
              const ring = el("circle", {cx: holoCenterX, cy: holoCenterY-20, r: 40 + i * 25, fill: "none", stroke: "#ffd700", "stroke-width": 2, opacity: 0, "stroke-dasharray": "20 10"});
              energyLayer.appendChild(ring);
              energyRings.push({el: ring, baseR: 40 + i * 25, phase: i * 2});
            }

            charLayer.append(holoBase, bodyGlow, robe, headGlow, head, eyeL, eyeR, crown, leftArm, rightArm);

            // State management
            window.isTyping = false;
            window.isThinking = false;
            window.isResponding = false;
            
            const mode = () => window.isResponding ? "Responding" : window.isThinking ? "Thinking" : window.isTyping ? "Typing" : "Idle";
            
            window.setFight = () => { 
              window.isTyping = false; 
              window.isThinking = false; 
              window.isResponding = false; 
            };
            
            window.setTyping = () => { 
              window.isTyping = true; 
              window.isThinking = false; 
              window.isResponding = false; 
            };
            
            window.setThinking = (ms = 2600) => {
              window.isTyping = false; 
              window.isThinking = true; 
              window.isResponding = false;
              clearTimeout(window.__t);
              window.__t = setTimeout(() => window.setFight(), ms);
            };
            
            window.setResponding = (ms = 3000) => {
              window.isTyping = false; 
              window.isThinking = false; 
              window.isResponding = true;
              clearTimeout(window.__r);
              window.__r = setTimeout(() => window.setFight(), ms);
            };

            // Bind to parent Streamlit
            function bindParent() {
              try {
                const d = window.parent.document;
                d.addEventListener("input", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    const has = !!(t.value && t.value.trim().length);
                    if (has) window.setTyping();
                    else if (!window.isThinking && !window.isResponding) window.setFight();
                  }
                }, true);
                d.addEventListener("keydown", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    if (e.key === "Enter" && !e.shiftKey) window.setThinking(2600);
                  }
                }, true);
                d.addEventListener("click", (e) => {
                  const b = e.target.closest("button");
                  if (!b) return;
                  const txt = (b.innerText || b.textContent || "").toLowerCase();
                  if (txt.includes("submit") || txt.includes("ask") || txt.includes("send")) {
                    window.setResponding(3000);
                  }
                }, true);
              } catch (_) {}
            }
            bindParent();

            // Animation loop
            let last = performance.now();
            function frame(now) {
              const t = now / 1000;
              const dt = Math.min(0.05, (now - last) / 1000);
              last = now;
              const m = mode();

              // Draw Matrix rain
              drawMatrix();

              // Animate orbs
              const orbSpeed = window.isThinking ? 2.5 : (window.isTyping ? 1.5 : 1.0);
              for (const o of orbs) {
                o.angle += o.speed * dt * orbSpeed;
                const x = holoCenterX + Math.cos(o.angle) * o.radius;
                const y = holoCenterY + Math.sin(o.angle) * (o.radius * 0.5) - 20;
                o.el.setAttribute("cx", x);
                o.el.setAttribute("cy", y);
                o.el.setAttribute("opacity", 0.5 + Math.sin(t * 3 + o.angle) * 0.3);
              }

              // Animate data cards
              const cardSpeed = window.isThinking ? 1.8 : (window.isTyping ? 1.3 : 1.0);
              for (const c of dataCards) {
                const y = c.by + Math.sin(t * c.orbit * cardSpeed + c.ph) * c.amp;
                const x = c.bx + Math.cos(t * c.orbit * 0.5 + c.ph) * c.orbitRadius;
                c.g.setAttribute("transform", `translate(${x}, ${y}) rotate(${Math.sin(t + c.ph) * 5})`);
              }

              // Animate data flow lines
              for (const dl of dataLines) {
                const offset = (t * dl.speed * 20 + dl.offset) % 100;
                dl.el.setAttribute("stroke-dashoffset", offset);
                const pathD = `M${holoCenterX + 60} ${holoCenterY} Q${holoCenterX + 150} ${holoCenterY - 50 + Math.sin(t + dl.offset) * 30} ${holoCenterX + 250} ${holoCenterY + Math.sin(t * 2 + dl.offset) * 20}`;
                dl.el.setAttribute("d", pathD);
              }

              // Character animation
              const charPulse = window.isThinking ? 1.15 : (window.isTyping ? 1.05 : 1.0);
              bodyGlow.setAttribute("opacity", 0.2 + Math.sin(t * 2) * 0.1);
              headGlow.setAttribute("opacity", 0.25 + Math.sin(t * 2.5) * 0.1);
              
              // Eye flicker
              const eyeFlicker = window.isThinking ? 0.6 + Math.sin(t * 8) * 0.4 : 0.9;
              eyeL.setAttribute("opacity", eyeFlicker);
              eyeR.setAttribute("opacity", eyeFlicker);

              // Lightbulb (thinking state)
              if (window.isThinking) {
                lightbulb.setAttribute("opacity", 1);
                lightbulb.setAttribute("transform", `translate(${holoCenterX + 80}, ${holoCenterY - 80})`);
                const bulbScale = 0.8 + Math.sin(t * 6) * 0.2;
                bulb.setAttribute("transform", `scale(${bulbScale})`);
                bulbGlow.setAttribute("opacity", 0.4 + Math.sin(t * 4) * 0.3);
                
                // Energy rings
                for (const ring of energyRings) {
                  ring.el.setAttribute("opacity", 0.5 + Math.sin(t * 3 + ring.phase) * 0.3);
                  const ringScale = 1 + Math.sin(t * 2 + ring.phase) * 0.15;
                  ring.el.setAttribute("r", ring.baseR * ringScale);
                  ring.el.setAttribute("transform", `rotate(${t * 30 + ring.phase * 60} ${holoCenterX} ${holoCenterY-20})`);
                }
              } else {
                lightbulb.setAttribute("opacity", 0);
                for (const ring of energyRings) {
                  ring.el.setAttribute("opacity", 0);
                }
              }

              // Data stream (responding state)
              if (window.isResponding) {
                if (Math.random() < 0.15) {
                  const particle = el("circle", {r: rand(2, 5), fill: ["#00ff88", "#ffd700", "#00ccff"][Math.floor(Math.random()*3)], opacity: 0.9});
                  particle.setAttribute("cx", holoCenterX + 30);
                  particle.setAttribute("cy", holoCenterY);
                  dataStreamLayer.appendChild(particle);
                  
                  const targetX = rand(400, 700);
                  const targetY = rand(300, 400);
                  const startTime = now;
                  const duration = 1500;
                  
                  const animateParticle = () => {
                    const elapsed = performance.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const ease = progress * (2 - progress);
                    const px = holoCenterX + 30 + (targetX - holoCenterX - 30) * ease;
                    const py = holoCenterY + (targetY - holoCenterY) * ease + Math.sin(progress * Math.PI) * 30;
                    particle.setAttribute("cx", px);
                    particle.setAttribute("cy", py);
                    particle.setAttribute("opacity", 0.9 * (1 - progress));
                    
                    if (progress < 1) requestAnimationFrame(animateParticle);
                    else particle.remove();
                  };
                  requestAnimationFrame(animateParticle);
                }
              }

              // Update HUD
              const dataLevel = window.isResponding ? Math.floor(70 + Math.sin(t * 5) * 30) : (window.isThinking ? Math.floor(40 + Math.sin(t * 3) * 20) : Math.floor(Math.sin(t) * 10 + 10));
              hud.textContent = `Mode: ${m} | Data: ${dataLevel}%`;
              
              requestAnimationFrame(frame);
            }
            requestAnimationFrame(frame);
          </script>
        </body>
        </html>
                    <stop offset="0%" stop-color="#a9d8ff" stop-opacity="0.95"/>
                    <stop offset="100%" stop-color="#7ab8f5" stop-opacity="0.12"/>
                  </radialGradient>
                  <radialGradient id="capGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#ffe082" stop-opacity="0.95"/>
                    <stop offset="100%" stop-color="#ffd54f" stop-opacity="0"/>
                  </radialGradient>
                </defs>
                <rect x="0" y="0" width="1100" height="420" fill="url(#bgGrad)"/>
                <g id="layerSky"></g>
                <g id="layerMap"></g>
                <g id="layerMid"></g>
                <g id="layerFloat"></g>
                <g id="layerCharacter"></g>
                <g id="layerFx"></g>
              </svg>
            </div>
            <div id="hud" class="hud">Mode: Idle | Depth: 0</div>
          </div>
          <script>
            const NS = "http://www.w3.org/2000/svg";
            const W = 1100, H = 420, TAU = Math.PI * 2;
            const box = document.getElementById("box");
            const sky = document.getElementById("layerSky");
            const mapLayer = document.getElementById("layerMap");
            const mid = document.getElementById("layerMid");
            const floatLayer = document.getElementById("layerFloat");
            const charLayer = document.getElementById("layerCharacter");
            const fx = document.getElementById("layerFx");
            const hud = document.getElementById("hud");

            const rand = (a, b) => a + Math.random() * (b - a);
            const el = (t, a = {}) => {
              const n = document.createElementNS(NS, t);
              for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
              return n;
            };

            // Base layers
            for (let x = 0; x <= W; x += 36) sky.appendChild(el("line", {x1:x, y1:0, x2:x, y2:H, stroke:"#c8d4e1", "stroke-opacity":"0.10", "stroke-width":"1"}));
            for (let y = 0; y <= H; y += 28) sky.appendChild(el("line", {x1:0, y1:y, x2:W, y2:y, stroke:"#c8d4e1", "stroke-opacity":"0.10", "stroke-width":"1"}));
            sky.appendChild(el("rect", {x:0, y:0, width:W, height:14, fill:"#ce1126", "fill-opacity":"0.07"}));
            sky.appendChild(el("rect", {x:0, y:14, width:W, height:9, fill:"#fcd116", "fill-opacity":"0.07"}));
            sky.appendChild(el("rect", {x:0, y:23, width:W, height:12, fill:"#006b3f", "fill-opacity":"0.07"}));
            mid.appendChild(el("rect", {x:0, y:322, width:W, height:H - 322, fill:"#dce6f2"}));

            // Ghana map silhouette (stylized polygon)
            const ghana = el("path", {
              d: "M320 110 L350 102 L386 110 L420 100 L458 118 L470 138 L500 156 L494 185 L512 214 L482 236 L454 246 L428 276 L394 286 L350 270 L318 272 L286 254 L268 228 L246 204 L260 182 L252 150 L270 126 Z",
              fill: "#1f4f7a",
              "fill-opacity": "0.20",
              stroke: "#1d466d",
              "stroke-opacity": "0.35",
              "stroke-width": "2"
            });
            mapLayer.appendChild(ghana);
            mapLayer.appendChild(el("text", {
              x: 350, y: 306, fill: "#325f84", "font-size": "18", "font-weight": "600", "letter-spacing": "1"
            })).textContent = "GHANA";

            // Floating books + orbs
            const books = [];
            for (let i = 0; i < 7; i++) {
              const g = el("g");
              const w = rand(42, 62), h = rand(14, 20);
              g.appendChild(el("rect", {x:-w/2, y:-h/2, width:w, height:h, rx:3, fill:["#5e81ac","#6f9dd3","#7aa2d8"][i%3], opacity:.92}));
              g.appendChild(el("line", {x1:-w/2+6, y1:0, x2:w/2-6, y2:0, stroke:"#e8f1ff", "stroke-width":"1.3"}));
              floatLayer.appendChild(g);
              books.push({g, bx:rand(90,530), by:rand(95,280), amp:rand(7,20), ph:rand(0,TAU), dr:rand(.3,.9)});
            }
            const oc = {x:760, y:180}, orbs = [];
            for (let i = 0; i < 8; i++) {
              const o = el("circle", {r:rand(10,17), fill:"url(#orbGlow)"});
              const c = el("circle", {r:2.2, fill:"#dff2ff", opacity:.95});
              floatLayer.appendChild(o); floatLayer.appendChild(c);
              orbs.push({o, c, r:50 + i*12, s:.35 + i*.06, p:rand(0,TAU)});
            }

            // Foreground character (oracle)
            const aura = el("circle",{cx:760, cy:142, r:64, fill:"url(#capGlow)", opacity:.55});
            const robe = el("path",{fill:"#335c88", d:"M720 312 L800 312 L832 366 L688 366 Z"});
            const neck = el("rect",{x:750, y:236, width:20, height:12, fill:"#f3c9a8"});
            const head = el("circle",{cx:760, cy:220, r:22, fill:"#f3c9a8"});
            const eyeL = el("circle",{cx:752, cy:218, r:1.8, fill:"#2f3e52"});
            const eyeR = el("circle",{cx:768, cy:218, r:1.8, fill:"#2f3e52"});
            const mouth = el("path",{d:"M752 227 Q760 232 768 227", fill:"none", stroke:"#946d57", "stroke-width":"1.4"});
            const cap = el("path",{fill:"#284f7a", d:"M732 202 L788 202 L760 186 Z"});
            const capBand = el("rect",{x:740, y:201, width:40, height:6, fill:"#223f62"});
            const tassel = el("line",{x1:786, y1:202, x2:796, y2:220, stroke:"#ffd54f", "stroke-width":"2"});
            const leftArm = el("path",{fill:"none", stroke:"#335c88", "stroke-width":"8", "stroke-linecap":"round"});
            const rightArm = el("path",{fill:"none", stroke:"#335c88", "stroke-width":"8", "stroke-linecap":"round"});
            const bookBack = el("rect",{x:708, y:286, width:104, height:58, rx:5, fill:"#6f9dd3"});
            const bookFront = el("path",{fill:"#8cb6e6"});
            const page1 = el("line",{x1:722, y1:304, x2:798, y2:304, stroke:"#e9f2ff", "stroke-width":"1.5"});
            const page2 = el("line",{x1:722, y1:318, x2:798, y2:318, stroke:"#e9f2ff", "stroke-width":"1.5"});
            charLayer.append(aura, robe, neck, head, eyeL, eyeR, mouth, cap, capBand, tassel, leftArm, rightArm, bookBack, bookFront, page1, page2);

            // Modes from parent app interaction
            window.isTyping = false;
            window.isThinking = false;
            window.isClapping = false;
            const mode = () => window.isClapping ? "Celebrating" : window.isThinking ? "Thinking" : window.isTyping ? "Typing" : "Idle";
            window.setFight = () => { window.isTyping = false; window.isThinking = false; window.isClapping = false; };
            window.setTyping = () => { window.isTyping = true; window.isThinking = false; window.isClapping = false; };
            window.setThinking = (ms = 2600) => {
              window.isTyping = false; window.isThinking = true; window.isClapping = false;
              clearTimeout(window.__t);
              window.__t = setTimeout(() => window.setFight(), ms);
            };
            window.setClapping = (ms = 2000) => {
              window.isTyping = false; window.isThinking = false; window.isClapping = true;
              clearTimeout(window.__c);
              window.__c = setTimeout(() => window.setFight(), ms);
            };

            function bindParent() {
              try {
                const d = window.parent.document;
                d.addEventListener("input", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    const has = !!(t.value && t.value.trim().length);
                    if (has) window.setTyping();
                    else if (!window.isThinking && !window.isClapping) window.setFight();
                  }
                }, true);
                d.addEventListener("keydown", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    if (e.key === "Enter" && !e.shiftKey) window.setThinking(2600);
                  }
                }, true);
                d.addEventListener("click", (e) => {
                  const b = e.target.closest("button");
                  if (!b) return;
                  const txt = (b.innerText || b.textContent || "").toLowerCase();
                  if (txt.includes("boosted") || txt.includes("penal") || txt.includes("yes") || txt.includes("no") || txt.includes("👍") || txt.includes("👎")) window.setClapping(2000);
                }, true);
              } catch (_) {}
            }
            bindParent();

            // 3D parallax target from mouse position
            const parallax = { tx: 0, ty: 0, x: 0, y: 0 };
            function updateMouseTarget(ev) {
              const r = box.getBoundingClientRect();
              const nx = ((ev.clientX - r.left) / r.width) * 2 - 1;
              const ny = ((ev.clientY - r.top) / r.height) * 2 - 1;
              parallax.tx = Math.max(-1, Math.min(1, nx));
              parallax.ty = Math.max(-1, Math.min(1, ny));
            }
            box.addEventListener("mousemove", updateMouseTarget);
            box.addEventListener("mouseleave", () => { parallax.tx = 0; parallax.ty = 0; });

            let last = performance.now();
            function frame(now) {
              const t = now / 1000;
              const dt = Math.min(0.05, (now - last) / 1000);
              last = now;
              const m = mode();

              // Smooth parallax interpolation
              parallax.x += (parallax.tx - parallax.x) * Math.min(1, dt * 9);
              parallax.y += (parallax.ty - parallax.y) * Math.min(1, dt * 9);
              const px = parallax.x * 18;
              const py = parallax.y * 12;

              sky.setAttribute("transform", `translate(${px * 0.12}, ${py * 0.10})`);
              mapLayer.setAttribute("transform", `translate(${px * 0.25}, ${py * 0.20})`);
              mid.setAttribute("transform", `translate(${px * 0.38}, ${py * 0.30})`);
              floatLayer.setAttribute("transform", `translate(${px * 0.55}, ${py * 0.45})`);
              charLayer.setAttribute("transform", `translate(${px * 0.85}, ${py * 0.68})`);

              // Depth tilt for mild 3D feel
              const rotX = (-parallax.y * 2.5).toFixed(2);
              const rotY = (parallax.x * 3.8).toFixed(2);
              box.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;

              // Floating animation
              for (const b of books) {
                const y = b.by + Math.sin(t * b.dr + b.ph) * b.amp;
                const x = b.bx + Math.cos(t * 0.6 + b.ph) * 8;
                const r = Math.sin(t * 0.8 + b.ph) * 7;
                const speedBoost = window.isTyping ? 1.6 : 1.0;
                b.g.setAttribute("transform", `translate(${x + Math.sin(t * speedBoost + b.ph) * 3},${y}) rotate(${r})`);
              }

              const orbSpeed = window.isThinking ? 1.9 : (window.isTyping ? 1.3 : 1.0);
              for (const o of orbs) {
                const a = t * o.s * orbSpeed + o.p;
                const x = oc.x + Math.cos(a) * o.r;
                const y = oc.y + Math.sin(a) * (o.r * 0.6);
                o.o.setAttribute("cx", x); o.o.setAttribute("cy", y);
                o.c.setAttribute("cx", x); o.c.setAttribute("cy", y);
              }

              const nod = (window.isThinking ? 4.2 : 2.0) * Math.sin(t * 2.2);
              head.setAttribute("cy", 220 + nod);
              eyeL.setAttribute("cy", 218 + nod);
              eyeR.setAttribute("cy", 218 + nod);
              mouth.setAttribute("d", `M752 ${227 + nod} Q760 ${232 + nod} 768 ${227 + nod}`);
              cap.setAttribute("d", `M732 ${202 + nod} L788 ${202 + nod} L760 ${186 + nod} Z`);
              capBand.setAttribute("y", 201 + nod);
              tassel.setAttribute("y1", 202 + nod);
              tassel.setAttribute("y2", 220 + nod + Math.sin(t * 3) * 2);

              const flip = (Math.sin(t * (window.isThinking ? 7.5 : 3.5)) + 1) * 0.5;
              const fold = 20 + flip * 38;
              bookFront.setAttribute("d", `M708 286 L${812 - fold} 286 L812 344 L708 344 Z`);
              leftArm.setAttribute("d", "M732 262 Q710 282 715 310");
              rightArm.setAttribute("d", `M788 262 Q808 280 ${790 + flip * 8} 308`);
              aura.setAttribute("opacity", (window.isThinking ? 0.85 + Math.sin(t * 5) * 0.18 : 0.48 + Math.sin(t * 2) * 0.08).toFixed(3));

              if (window.isClapping && Math.random() < 0.28) {
                const p = el("circle", {cx: rand(660, 860), cy: rand(70, 170), r: rand(1.5, 3.0), fill:"#ffd54f", opacity:.95});
                fx.appendChild(p);
                setTimeout(() => p.remove(), 650);
              }

              hud.textContent = `Mode: ${m} | Depth: ${Math.abs(px).toFixed(1)}`;
              requestAnimationFrame(frame);
            }
            requestAnimationFrame(frame);
          </script>
        </body>
        </html>
        """
    )
    components.html(modern_html, height=490, scrolling=False)
    return

    html = textwrap.dedent(
        """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <style>
            :root { --w: 1100; --h: 470; }
            body { margin: 0; padding: 0; background: transparent; font-family: sans-serif; }
            .arena-wrap { width: 100%; max-width: 1150px; margin: 0 auto; }
            .arena-box {
              width: 100%;
              aspect-ratio: 1100 / 470;
              border-radius: 16px;
              border: 1px solid #d4deea;
              overflow: hidden;
              background: #f5f8fc;
              box-shadow: inset 0 0 0 1px rgba(255,255,255,0.5), 0 2px 12px rgba(35, 52, 70, 0.08);
            }
            svg { width: 100%; height: 100%; display: block; }
            .hud {
              position: absolute;
              right: 14px;
              top: 10px;
              background: rgba(255, 255, 255, 0.82);
              border: 1px solid #d2dce9;
              border-radius: 10px;
              padding: 6px 8px;
              font-size: 12px;
              color: #455a70;
            }
            .frame { position: relative; }
          </style>
        </head>
        <body>
          <div class="arena-wrap">
            <div class="frame">
              <div class="arena-box">
                <svg id="arena" viewBox="0 0 1100 470" preserveAspectRatio="xMidYMid meet" aria-label="SVG stickman combat arena">
                  <defs>
                    <linearGradient id="bgGrad" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stop-color="#f9fcff"/>
                      <stop offset="100%" stop-color="#edf3fa"/>
                    </linearGradient>
                  </defs>
                  <rect x="0" y="0" width="1100" height="470" fill="url(#bgGrad)"/>
                  <g id="bgLayer"></g>
                  <g id="sparkLayer"></g>
                  <g id="fighterLayer"></g>
                  <g id="fxLayer"></g>
                </svg>
              </div>
              <div id="hud" class="hud">Mode: fight | Alive: 0</div>
            </div>
          </div>
          <script>
            const NS = "http://www.w3.org/2000/svg";
            const CONFIG = {
              w: 1100,
              h: 470,
              groundY: 360,
              thinkMs: 2600,
              clapMs: 2000,
              roundResetMs: 2200,
              palette: {
                red: "#e74c3c", orange: "#f39c12", teal: "#16a085", blue: "#3498db",
                purple: "#8e44ad", green: "#27ae60", rose: "#e84393", cyan: "#00bcd4",
                sword: "#6d7a89", spark: "#ffd870", dead: "#8d97a2", text: "#455a70"
              },
              size: { head: 10, body: 35, arm: 24, leg: 25, sword: 32, stroke: 3.2 }
            };

            const arena = document.getElementById("arena");
            const fighterLayer = document.getElementById("fighterLayer");
            const sparkLayer = document.getElementById("sparkLayer");
            const fxLayer = document.getElementById("fxLayer");
            const bgLayer = document.getElementById("bgLayer");
            const hud = document.getElementById("hud");

            function svgEl(tag, attrs = {}) {
              const n = document.createElementNS(NS, tag);
              for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
              return n;
            }
            const rand = (a, b) => a + Math.random() * (b - a);
            const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
            const TAU = Math.PI * 2;

            function drawStaticBackground() {
              // Subtle grid.
              for (let x = 0; x <= CONFIG.w; x += 36) {
                bgLayer.appendChild(svgEl("line", { x1: x, y1: 0, x2: x, y2: CONFIG.h, stroke: "#c8d4e1", "stroke-opacity": "0.10", "stroke-width": "1" }));
              }
              for (let y = 0; y <= CONFIG.h; y += 28) {
                bgLayer.appendChild(svgEl("line", { x1: 0, y1: y, x2: CONFIG.w, y2: y, stroke: "#c8d4e1", "stroke-opacity": "0.10", "stroke-width": "1" }));
              }
              // Subtle Ghana accent.
              bgLayer.appendChild(svgEl("rect", { x: 0, y: 0, width: CONFIG.w, height: 14, fill: "#ce1126", "fill-opacity": "0.07" }));
              bgLayer.appendChild(svgEl("rect", { x: 0, y: 14, width: CONFIG.w, height: 9, fill: "#fcd116", "fill-opacity": "0.07" }));
              bgLayer.appendChild(svgEl("rect", { x: 0, y: 23, width: CONFIG.w, height: 12, fill: "#006b3f", "fill-opacity": "0.07" }));
              bgLayer.appendChild(svgEl("rect", { x: 0, y: CONFIG.groundY, width: CONFIG.w, height: CONFIG.h - CONFIG.groundY, fill: "#dce6f2" }));
            }
            drawStaticBackground();

            window.isTyping = false;
            window.isThinking = false;
            window.isClapping = false;
            function mode() {
              if (window.isClapping) return "clap";
              if (window.isThinking) return "think";
              if (window.isTyping) return "typing";
              return "fight";
            }
            window.setFight = function() { window.isTyping = false; window.isThinking = false; window.isClapping = false; };
            window.setTyping = function() { window.isTyping = true; window.isThinking = false; window.isClapping = false; };
            window.setThinking = function(ms = 2600) {
              window.isTyping = false; window.isThinking = true; window.isClapping = false;
              clearTimeout(window.__thinkTimer);
              window.__thinkTimer = setTimeout(() => { if (window.isThinking) window.setFight(); }, ms);
            };
            window.setClapping = function(ms = 2000) {
              window.isTyping = false; window.isThinking = false; window.isClapping = true;
              clearTimeout(window.__clapTimer);
              window.__clapTimer = setTimeout(() => { if (window.isClapping) window.setFight(); }, ms);
            };
            class Spark {
              constructor(x, y) {
                this.x = x;
                this.y = y;
                this.vx = rand(-2.3, 2.3);
                this.vy = rand(-2.8, 1.0);
                this.life = rand(0.18, 0.38);
                this.age = 0;
                this.node = svgEl("circle", { cx: x, cy: y, r: rand(1.2, 2.4), fill: CONFIG.palette.spark });
                sparkLayer.appendChild(this.node);
              }
              update(dt) {
                this.age += dt;
                this.x += this.vx * 60 * dt;
                this.y += this.vy * 60 * dt;
                this.vy += 0.09;
                const alpha = Math.max(0, 1 - this.age / this.life);
                this.node.setAttribute("cx", this.x);
                this.node.setAttribute("cy", this.y);
                this.node.setAttribute("fill-opacity", alpha.toFixed(3));
                if (this.age >= this.life) {
                  this.node.remove();
                  return false;
                }
                return true;
              }
            }
            const sparks = [];
            function spawnSparks(x, y, n = 8) { for (let i = 0; i < n; i++) sparks.push(new Spark(x, y)); }

            function segDistSq(a, b, c, d) {
              const ux = b.x - a.x, uy = b.y - a.y;
              const vx = d.x - c.x, vy = d.y - c.y;
              const wx = a.x - c.x, wy = a.y - c.y;
              const A = ux * ux + uy * uy;
              const B = ux * vx + uy * vy;
              const C = vx * vx + vy * vy;
              const D = ux * wx + uy * wy;
              const E = vx * wx + vy * wy;
              const den = A * C - B * B;
              let s = 0, t = 0;
              if (den !== 0) s = clamp((B * E - C * D) / den, 0, 1);
              t = (B * s + E) / C;
              if (t < 0) { t = 0; s = clamp(-D / A, 0, 1); }
              else if (t > 1) { t = 1; s = clamp((B - D) / A, 0, 1); }
              const dx = (a.x + ux * s) - (c.x + vx * t);
              const dy = (a.y + uy * s) - (c.y + vy * t);
              return dx * dx + dy * dy;
            }

            class Fighter {
              constructor(opts) {
                this.id = opts.id;
                this.team = opts.team;
                this.baseX = opts.x;
                this.baseY = opts.y;
                this.x = opts.x;
                this.y = opts.y;
                this.phase = opts.phase;
                this.color = opts.color;
                this.vx = 0;
                this.facing = this.team === "L" ? 1 : -1;
                this.attackCooldown = rand(0.4, 1.3);
                this.attackPhase = 0;
                this.health = 100;
                this.alive = true;
                this.deathPose = 0;
                this.swordStart = { x: this.x, y: this.y };
                this.swordEnd = { x: this.x, y: this.y };
                this.damageLock = 0;
                this.group = svgEl("g");
                this.head = svgEl("circle", { r: CONFIG.size.head, fill: this.color });
                this.body = svgEl("line", { "stroke-width": CONFIG.size.stroke, stroke: this.color, "stroke-linecap": "round" });
                this.armL = svgEl("line", { "stroke-width": CONFIG.size.stroke, stroke: this.color, "stroke-linecap": "round" });
                this.armR = svgEl("line", { "stroke-width": CONFIG.size.stroke, stroke: this.color, "stroke-linecap": "round" });
                this.legL = svgEl("line", { "stroke-width": CONFIG.size.stroke, stroke: this.color, "stroke-linecap": "round" });
                this.legR = svgEl("line", { "stroke-width": CONFIG.size.stroke, stroke: this.color, "stroke-linecap": "round" });
                this.sword = svgEl("line", { "stroke-width": "2.7", stroke: CONFIG.palette.sword, "stroke-linecap": "round" });
                this.eyeL = svgEl("circle", { r: "1.2", fill: "#fff" });
                this.eyeR = svgEl("circle", { r: "1.2", fill: "#fff" });
                this.hpBg = svgEl("rect", { width: "34", height: "4", rx: "2", fill: "#c8d2de" });
                this.hpFg = svgEl("rect", { width: "34", height: "4", rx: "2", fill: "#3dbb6a" });
                this.group.append(this.hpBg, this.hpFg, this.body, this.armL, this.armR, this.legL, this.legR, this.sword, this.head, this.eyeL, this.eyeR);
                fighterLayer.appendChild(this.group);
              }
              reset() {
                this.x = this.baseX;
                this.y = this.baseY;
                this.vx = 0;
                this.health = 100;
                this.alive = true;
                this.deathPose = 0;
                this.attackPhase = 0;
                this.attackCooldown = rand(0.2, 1.0);
                this.damageLock = 0;
                this.applyColor(this.color);
              }
              applyColor(c) {
                this.head.setAttribute("fill", c);
                this.body.setAttribute("stroke", c);
                this.armL.setAttribute("stroke", c);
                this.armR.setAttribute("stroke", c);
                this.legL.setAttribute("stroke", c);
                this.legR.setAttribute("stroke", c);
              }
              findTarget(all) {
                let best = null, dMin = 1e9;
                for (const f of all) {
                  if (f === this || !f.alive || f.team === this.team) continue;
                  const d = Math.abs(f.x - this.x);
                  if (d < dMin) { dMin = d; best = f; }
                }
                return [best, dMin];
              }
              receiveDamage(dmg) {
                if (!this.alive || this.damageLock > 0) return;
                this.health = Math.max(0, this.health - dmg);
                this.damageLock = 0.12;
                if (this.health <= 0) {
                  this.alive = false;
                  this.applyColor(CONFIG.palette.dead);
                  this.hpFg.setAttribute("width", "0");
                }
              }
              update(t, dt, m, all) {
                if (this.damageLock > 0) this.damageLock -= dt;
                if (!this.alive) {
                  this.deathPose = clamp(this.deathPose + dt * 0.9, 0, 1);
                  return;
                }
                if (m === "fight") {
                  const [target, dist] = this.findTarget(all);
                  if (target) {
                    const dir = target.x > this.x ? 1 : -1;
                    const desired = 98 + rand(-6, 8);
                    const drive = clamp((dist - desired) * 0.014, -1.9, 1.9);
                    this.vx = this.vx * 0.82 + dir * drive;
                    this.x += this.vx;
                    this.facing = dir;
                  }
                  this.x = clamp(this.x, 42, CONFIG.w - 42);
                  this.y = this.baseY + Math.cos(t * 4 + this.phase) * 1.8;
                  this.attackCooldown -= dt;
                  if (this.attackCooldown <= 0) {
                    this.attackPhase = 1;
                    this.attackCooldown = rand(0.55, 1.25);
                  }
                  this.attackPhase = Math.max(0, this.attackPhase - dt * 1.55);
                } else if (m === "typing") {
                  this.x += (this.baseX - this.x) * 0.12;
                  this.y += (this.baseY - this.y) * 0.12;
                  this.facing = this.team === "L" ? 1 : -1;
                  this.attackPhase = 0;
                } else if (m === "think") {
                  this.x += (this.baseX - this.x) * 0.12;
                  this.y += (this.baseY - this.y) * 0.12;
                  this.facing = this.team === "L" ? 1 : -1;
                } else if (m === "clap") {
                  this.x += (this.baseX - this.x) * 0.1;
                  this.y = this.baseY + Math.sin(t * 8 + this.phase) * 2.5;
                  this.facing = this.team === "L" ? 1 : -1;
                }
              }
              render(t, m, idx) {
                const S = CONFIG.size;
                let armL = -0.9, armR = 0.6, legL = -0.7, legR = 0.7, swordA = -0.3, eye = 0;
                if (!this.alive) {
                  const fall = this.deathPose * (Math.PI * 0.45) * (this.team === "L" ? 1 : -1);
                  const x = this.x, y = this.y + this.deathPose * 16;
                  const bodyX2 = x + Math.sin(fall) * S.body;
                  const bodyY2 = y + Math.cos(fall) * S.body;
                  this.body.setAttribute("x1", x); this.body.setAttribute("y1", y);
                  this.body.setAttribute("x2", bodyX2); this.body.setAttribute("y2", bodyY2);
                  this.head.setAttribute("cx", x + Math.sin(fall) * 12);
                  this.head.setAttribute("cy", y - 8 + Math.cos(fall) * 10);
                  this.eyeL.setAttribute("cx", x + Math.sin(fall) * 12 - 2);
                  this.eyeL.setAttribute("cy", y - 8 + Math.cos(fall) * 10 - 1);
                  this.eyeR.setAttribute("cx", x + Math.sin(fall) * 12 + 2);
                  this.eyeR.setAttribute("cy", y - 8 + Math.cos(fall) * 10 - 1);
                  this.armL.setAttribute("x1", x); this.armL.setAttribute("y1", y + 6);
                  this.armL.setAttribute("x2", x - 16); this.armL.setAttribute("y2", y + 18);
                  this.armR.setAttribute("x1", x); this.armR.setAttribute("y1", y + 6);
                  this.armR.setAttribute("x2", x + 16); this.armR.setAttribute("y2", y + 18);
                  this.sword.setAttribute("x1", x + 16); this.sword.setAttribute("y1", y + 18);
                  this.sword.setAttribute("x2", x + 36); this.sword.setAttribute("y2", y + 24);
                  this.legL.setAttribute("x1", bodyX2); this.legL.setAttribute("y1", bodyY2);
                  this.legL.setAttribute("x2", bodyX2 - 14); this.legL.setAttribute("y2", bodyY2 + 10);
                  this.legR.setAttribute("x1", bodyX2); this.legR.setAttribute("y1", bodyY2);
                  this.legR.setAttribute("x2", bodyX2 + 14); this.legR.setAttribute("y2", bodyY2 + 10);
                  this.hpBg.setAttribute("x", x - 17); this.hpBg.setAttribute("y", y - 30);
                  this.hpFg.setAttribute("x", x - 17); this.hpFg.setAttribute("y", y - 30);
                  this.swordStart = { x: x + 16, y: y + 18 };
                  this.swordEnd = { x: x + 36, y: y + 24 };
                  return;
                }
                if (m === "fight") {
                  const strike = Math.sin((1 - this.attackPhase) * Math.PI);
                  const walk = Math.sin(t * 6 + this.phase);
                  armL = -0.9 + Math.cos(t * 4 + this.phase) * 0.08;
                  armR = 0.2 + strike * 0.95 + Math.cos(t * 4 + this.phase) * 0.08;
                  legL = -0.75 + walk * 0.2;
                  legR = 0.75 - walk * 0.2;
                  swordA = (-0.25 + strike * 1.35) * this.facing;
                  eye = 0.8 * this.facing;
                } else if (m === "typing") {
                  armL = 0.1; armR = 0.2; legL = -0.55; legR = 0.55; swordA = 1.5; eye = 1.4;
                } else if (m === "think") {
                  legL = -0.5; legR = 0.5; eye = 1.0;
                  if (idx === 0) { armR = -1.5 + Math.sin(t * 10) * 0.2; armL = 0.15; swordA = 1.45; }
                  else { armL = 0.25; armR = 0.38; swordA = 1.35; }
                } else if (m === "clap") {
                  const cl = Math.sin(t * 16 + this.phase);
                  armL = -0.2 + cl * 0.45;
                  armR = 0.2 - cl * 0.45;
                  legL = -0.5; legR = 0.5; swordA = 1.55; eye = 0.5;
                }
                const x = this.x, y = this.y;
                const sy = y + 7;
                const lx2 = x + Math.cos(Math.PI + armL) * S.arm;
                const ly2 = sy + Math.sin(Math.PI + armL) * S.arm;
                const rx2 = x + Math.cos(armR) * S.arm;
                const ry2 = sy + Math.sin(armR) * S.arm;
                const sx2 = rx2 + Math.cos(swordA) * S.sword;
                const sy2 = ry2 + Math.sin(swordA) * S.sword;
                this.body.setAttribute("x1", x); this.body.setAttribute("y1", y);
                this.body.setAttribute("x2", x); this.body.setAttribute("y2", y + S.body);
                this.head.setAttribute("cx", x); this.head.setAttribute("cy", y - S.head - 2);
                this.eyeL.setAttribute("cx", x - 2 + eye); this.eyeL.setAttribute("cy", y - S.head - 4);
                this.eyeR.setAttribute("cx", x + 2 + eye); this.eyeR.setAttribute("cy", y - S.head - 4);
                this.armL.setAttribute("x1", x); this.armL.setAttribute("y1", sy);
                this.armL.setAttribute("x2", lx2); this.armL.setAttribute("y2", ly2);
                this.armR.setAttribute("x1", x); this.armR.setAttribute("y1", sy);
                this.armR.setAttribute("x2", rx2); this.armR.setAttribute("y2", ry2);
                this.sword.setAttribute("x1", rx2); this.sword.setAttribute("y1", ry2);
                this.sword.setAttribute("x2", sx2); this.sword.setAttribute("y2", sy2);
                this.legL.setAttribute("x1", x); this.legL.setAttribute("y1", y + S.body);
                this.legL.setAttribute("x2", x + Math.cos(Math.PI + legL) * S.leg); this.legL.setAttribute("y2", y + S.body + Math.sin(Math.PI + legL) * S.leg);
                this.legR.setAttribute("x1", x); this.legR.setAttribute("y1", y + S.body);
                this.legR.setAttribute("x2", x + Math.cos(legR) * S.leg); this.legR.setAttribute("y2", y + S.body + Math.sin(legR) * S.leg);
                this.hpBg.setAttribute("x", x - 17); this.hpBg.setAttribute("y", y - 28);
                this.hpFg.setAttribute("x", x - 17); this.hpFg.setAttribute("y", y - 28);
                this.hpFg.setAttribute("width", (34 * (this.health / 100)).toFixed(2));
                this.swordStart = { x: rx2, y: ry2 };
                this.swordEnd = { x: sx2, y: sy2 };
              }
            }

            const fighters = [
              new Fighter({ id: 1, team: "L", x: 120, y: 300, phase: 0.2, color: CONFIG.palette.red }),
              new Fighter({ id: 2, team: "L", x: 210, y: 255, phase: 1.0, color: CONFIG.palette.orange }),
              new Fighter({ id: 3, team: "L", x: 295, y: 320, phase: 1.8, color: CONFIG.palette.purple }),
              new Fighter({ id: 4, team: "L", x: 370, y: 275, phase: 2.4, color: CONFIG.palette.green }),
              new Fighter({ id: 5, team: "R", x: 970, y: 300, phase: 0.6, color: CONFIG.palette.blue }),
              new Fighter({ id: 6, team: "R", x: 895, y: 250, phase: 1.5, color: CONFIG.palette.teal }),
              new Fighter({ id: 7, team: "R", x: 815, y: 322, phase: 2.2, color: CONFIG.palette.rose }),
              new Fighter({ id: 8, team: "R", x: 730, y: 275, phase: 3.1, color: CONFIG.palette.cyan }),
            ];

            function aliveCounts() {
              let l = 0, r = 0;
              for (const f of fighters) {
                if (!f.alive) continue;
                if (f.team === "L") l++; else r++;
              }
              return [l, r];
            }
            let roundOverAt = 0;
            function resetRound() {
              sparks.forEach((s) => s.node.remove());
              sparks.length = 0;
              fxLayer.innerHTML = "";
              fighters.forEach((f) => f.reset());
              roundOverAt = 0;
            }
            function showWinnerText(text) {
              fxLayer.innerHTML = "";
              const label = svgEl("text", {
                x: CONFIG.w / 2,
                y: 84,
                "text-anchor": "middle",
                "font-size": "28",
                "font-weight": "700",
                "fill": "#3f5368",
                "fill-opacity": "0.93"
              });
              label.textContent = text;
              fxLayer.appendChild(label);
            }

            function drawThinkingBubble(t) {
              fxLayer.innerHTML = "";
              const g = svgEl("g", { "fill-opacity": "0.92" });
              const x = 160, y = 100;
              const p = (Math.sin(t * 4) + 1) * 0.5;
              g.appendChild(svgEl("circle", { cx: x, cy: y, r: 14, fill: "#fff" }));
              g.appendChild(svgEl("circle", { cx: x + 24, cy: y - 9, r: 9, fill: "#fff" }));
              g.appendChild(svgEl("circle", { cx: x + 38, cy: y - 15, r: 6, fill: "#fff" }));
              g.appendChild(svgEl("circle", { cx: x - 4, cy: y, r: 2 + p * 0.4, fill: "#7f8c8d" }));
              g.appendChild(svgEl("circle", { cx: x + 3, cy: y, r: 2 + p * 0.4, fill: "#7f8c8d" }));
              g.appendChild(svgEl("circle", { cx: x + 10, cy: y, r: 2 + p * 0.4, fill: "#7f8c8d" }));
              fxLayer.appendChild(g);
            }

            function bindParentEvents(){
              try {
                const doc = window.parent.document;
                let inputEl = null;
                const scanInput = () => {
                  inputEl = doc.querySelector("textarea[aria-label*='chat' i], textarea[placeholder*='Type your question' i], textarea, input[type='text']");
                };
                scanInput();
                const observer = new MutationObserver(() => {
                  if (!inputEl || !doc.contains(inputEl)) scanInput();
                });
                observer.observe(doc.body, { childList: true, subtree: true });

                doc.addEventListener("input", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    const hasText = !!(t.value && t.value.trim().length);
                    if (hasText) window.setTyping(); else if (!window.isThinking && !window.isClapping) window.setFight();
                  }
                }, true);

                doc.addEventListener("keydown", (e) => {
                  const t = e.target;
                  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text"))) {
                    if (e.key === "Enter" && !e.shiftKey) window.setThinking(2600);
                  }
                }, true);

                doc.addEventListener("click", (e) => {
                  const b = e.target.closest("button");
                  if (!b) return;
                  const txt = (b.innerText || b.textContent || "").trim().toLowerCase();
                  if (txt.includes("boosted") || txt.includes("penal") || txt.includes("yes") || txt.includes("no") || txt.includes("👍") || txt.includes("👎")) {
                    window.setClapping(2000);
                  }
                }, true);
              } catch (err) {
                // If parent DOM access is blocked, component still runs in default loop.
              }
            }
            bindParentEvents();

            let last = performance.now();
            function frame(now) {
              const dt = Math.min(0.05, (now - last) / 1000);
              const t = now / 1000;
              last = now;
              const m = mode();
              if (m !== "think") fxLayer.innerHTML = "";

              for (const f of fighters) f.update(t, dt, m, fighters);
              for (let i = 0; i < fighters.length; i++) fighters[i].render(t, m, i);

              if (m === "fight") {
                for (let i = 0; i < fighters.length; i++) {
                  for (let j = i + 1; j < fighters.length; j++) {
                    const a = fighters[i], b = fighters[j];
                    if (!a.alive || !b.alive || a.team === b.team) continue;
                    const d2 = segDistSq(a.swordStart, a.swordEnd, b.swordStart, b.swordEnd);
                    if (d2 < 30) {
                      const mx = (a.swordEnd.x + b.swordEnd.x) * 0.5;
                      const my = (a.swordEnd.y + b.swordEnd.y) * 0.5;
                      spawnSparks(mx, my, 7);
                      // Apply damage from active strike swings only.
                      if (a.attackPhase > 0.42) b.receiveDamage(rand(8, 16));
                      if (b.attackPhase > 0.42) a.receiveDamage(rand(8, 16));
                    }
                  }
                }
                const [lAlive, rAlive] = aliveCounts();
                if ((lAlive === 0 || rAlive === 0) && roundOverAt === 0) {
                  roundOverAt = now;
                  showWinnerText(lAlive > 0 ? "Left Team Wins" : "Right Team Wins");
                }
                if (roundOverAt && now - roundOverAt > CONFIG.roundResetMs) {
                  resetRound();
                }
              } else if (m === "think") {
                drawThinkingBubble(t);
              } else if (m === "clap" && Math.random() < 0.28) {
                const alive = fighters.filter((f) => f.alive);
                if (alive.length) {
                  const p = alive[(Math.random() * alive.length) | 0];
                  spawnSparks(p.x, p.y - 30, 10);
                }
              }

              for (let i = sparks.length - 1; i >= 0; i--) {
                if (!sparks[i].update(dt)) sparks.splice(i, 1);
              }

              const [lAlive, rAlive] = aliveCounts();
              hud.textContent = `Mode: ${m} | Alive: L${lAlive} - R${rAlive}`;
              requestAnimationFrame(frame);
            }
            requestAnimationFrame(frame);
          </script>
        </body>
        </html>
        """
    )
    components.html(html, height=480, scrolling=False)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Acity Oracle – Academic City Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Main background - behind everything */
body {
    background: linear-gradient(135deg, #0a1a12 0%, #0d2818 50%, #06140c 100%);
    background-attachment: fixed;
}

/* Left pane / Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 26, 18, 0.95) 0%, rgba(13, 40, 24, 0.95) 100%);
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="%2300ff88" stop-opacity="0.1"/><stop offset="100%" stop-color="%23ffd700" stop-opacity="0.05"/></linearGradient></defs><circle cx="100" cy="100" r="80" fill="url(%23g)"/></svg>');
    background-repeat: no-repeat;
    background-position: left bottom;
    background-size: 80% auto;
}

/* Retrieved chunk cards */
.chunk-card {
    background: #f0f4ff;
    border-left: 4px solid #4a6cf7;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.84rem;
    line-height: 1.5;
}
.score-badge {
    display: inline-block;
    background: #4a6cf7;
    color: #fff;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
/* Prompt display */
.prompt-box {
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    font-family: "Courier New", monospace;
    font-size: 0.77rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 320px;
    overflow-y: auto;
}
/* Pipeline log */
.log-box {
    background: #0d1117;
    color: #7ee787;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-family: "Courier New", monospace;
    font-size: 0.73rem;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
}
/* Source tag */
.source-tag {
    font-size: 0.72rem;
    color: #555;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 The Acity Oracle")
    st.caption("Academic City University Knowledge Assistant")
    st.divider()

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-K documents to retrieve",
        min_value=1, max_value=10, value=5,
        help="Number of context chunks shown to the LLM.",
    )
    alpha = st.slider(
        "Semantic weight α",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="α=1.0 → pure semantic search | α=0.0 → pure keyword (TF-IDF)\n"
             "Default 0.7 balances both (hybrid search).",
    )
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=["fixed", "sentence"],
        index=0,
        help=(
            "fixed: sliding window (500 chars, 75 overlap)\n"
            "sentence: sentence-boundary grouping (~450 chars, ≤5 sentences)\n\n"
            "Changing strategy rebuilds the index on first use."
        ),
    )

    st.divider()

    st.subheader("🔍 Debug Options")
    show_prompt  = st.checkbox("Show prompt sent to LLM",   value=True)
    show_log     = st.checkbox("Show pipeline execution log", value=False)

    st.divider()

    st.subheader("📊 Feedback Stats")
    fb_store = FeedbackStore()
    stats    = fb_store.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("👍 Boosted",   stats["positive"])
    c2.metric("👎 Penalised", stats["negative"])
    if st.button("🔄 Reset feedback"):
        FeedbackStore().reset()
        st.success("Feedback cleared.")
        st.rerun()

    st.divider()
    st.caption(
        "Built by **Michael Nana Kwame Osei-Dei**  \n"
        "Index: **10022300168**  \n"
        "IT3241 · AI Project 2026"
    )


# ── Cached pipeline loader ────────────────────────────────────────────────────

@st.cache_resource(
    show_spinner="⏳ Building knowledge base — first run may take a few minutes…"
)
def get_pipeline(strategy: str) -> RAGPipeline:
    """
    Cache one RAGPipeline per chunking strategy.
    Streamlit reruns reuse the cached object; retrieval parameters (α, k)
    are passed at query time rather than baked into the cached object.
    """
    return RAGPipeline(strategy=strategy)


# ── Session state initialisation ──────────────────────────────────────────────
if "messages"      not in st.session_state:
    st.session_state.messages      = []   # [{role, content}, …]
if "last_result"   not in st.session_state:
    st.session_state.last_result   = None
if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🎓 The Acity Oracle – Academic City Knowledge Assistant")
st.markdown(
    "Ask me anything about **Ghana Election Results** "
    "or the **Ghana 2025 Budget Statement**."
)
render_arena_widget()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Query input ───────────────────────────────────────────────────────────────
query = st.chat_input("Type your question here…")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            pipeline = get_pipeline(chunking_strategy)
            result   = pipeline.run(query, alpha=alpha, top_k=top_k)

        st.session_state.last_result   = result
        st.session_state.feedback_done = False

        # ── Retrieved context panel ───────────────────────────────────────────
        n_retrieved = len(result["retrieved"])
        n_used      = len(result["used_chunks"])
        with st.expander(
            f"📚 Retrieved Context  ({n_used} used in prompt / {n_retrieved} retrieved)",
            expanded=True,
        ):
            if not result["retrieved"]:
                st.info("No relevant chunks found in the knowledge base.")
            else:
                for i, r in enumerate(result["retrieved"], start=1):
                    chunk  = r["chunk"]
                    text   = chunk.get("chunk_text", "")
                    source = chunk.get("source", "Unknown")
                    page   = chunk.get("page", "")
                    truncated = text[:350] + ("…" if len(text) > 350 else "")
                    in_prompt  = i <= n_used    # was this snippet sent to LLM?

                    border_col = "#4a6cf7" if in_prompt else "#aab4cc"
                    bg_col     = "#f0f4ff" if in_prompt else "#f8f9ff"
                    page_suffix = f" · page {page}" if page else ""

                    st.markdown(
                        f'<div class="chunk-card" style="border-color:{border_col};background:{bg_col};">'
                        f'<strong>#{i}</strong>&ensp;'
                        f'<span class="score-badge">Combined: {r["combined_score"]:.3f}</span>&ensp;'
                        f'<span style="font-size:0.75rem;">Sem: {r["semantic_score"]:.3f} &nbsp;|&nbsp; '
                        f'Kwd: {r["keyword_score"]:.3f}</span>'
                        f'{"&ensp;<em style=\"color:#4a6cf7;font-size:0.72rem;\">[in prompt]</em>" if in_prompt else ""}'
                        f'<br><span class="source-tag">📄 {source}{page_suffix}</span>'
                        f"<br><br>{truncated}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Prompt display ────────────────────────────────────────────────────
        if show_prompt:
            with st.expander("🔍 Exact Prompt Sent to LLM", expanded=False):
                safe_prompt = result["prompt"].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f'<div class="prompt-box">{safe_prompt}</div>',
                    unsafe_allow_html=True,
                )

        # ── Pipeline log ──────────────────────────────────────────────────────
        if show_log:
            with st.expander("🪵 Pipeline Execution Log", expanded=False):
                log_html = "\n".join(result["pipeline_log"])
                st.markdown(
                    f'<div class="log-box">{log_html}</div>',
                    unsafe_allow_html=True,
                )

        # ── LLM response ──────────────────────────────────────────────────────
        llm     = result["llm_result"]
        resp    = llm.get("response") or ""
        model   = llm.get("model", "unknown")
        err     = llm.get("error")

        if err and not resp:
            st.error(f"LLM Error: {err}")
            resp = (
                "⚠️ Could not reach the language model. "
                "Make sure **HF_TOKEN** is set in your environment or `.env` file."
            )

        st.markdown(resp)
        st.caption(f"_Model: {model}_")

        st.session_state.messages.append({"role": "assistant", "content": resp})


# ── Feedback panel (Part G) ───────────────────────────────────────────────────
if st.session_state.last_result and not st.session_state.feedback_done:
    st.divider()
    st.markdown("**Was this answer helpful?**")
    col_y, col_n, col_pad = st.columns([1, 1, 8])

    with col_y:
        if st.button("👍 Yes", key="btn_positive", use_container_width=True):
            cids = [r["chunk"].get("chunk_id", "") for r in st.session_state.last_result["retrieved"]]
            FeedbackStore().record(cids, positive=True)
            st.session_state.feedback_done = True
            st.success("Thank you! Retrieved chunks boosted for future queries.")
            st.rerun()

    with col_n:
        if st.button("👎 No", key="btn_negative", use_container_width=True):
            cids = [r["chunk"].get("chunk_id", "") for r in st.session_state.last_result["retrieved"]]
            FeedbackStore().record(cids, positive=False)
            st.session_state.feedback_done = True
            st.warning("Noted. These chunks will be ranked lower next time.")
            st.rerun()
