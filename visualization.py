from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence


def write_algorithm_3d_model(here: Path, results: Sequence[Dict[str, object]], portfolio: Dict[str, object]) -> Path:
    """Write an interactive 3D HTML model of the latest algorithm run."""

    ranked = sorted(results, key=lambda item: item["test"].summary.get("sharpe", -99.0), reverse=True)
    pairs = []
    for item in ranked[:40]:
        test = item["test"].summary
        train = item["train"].summary
        validation = item["validation"].summary
        quality = item["quality"]
        config = item["config"]
        pairs.append(
            {
                "pair": item["pair"],
                "selected": item in portfolio["selected"],
                "trainSharpe": train.get("sharpe", 0.0),
                "validationSharpe": validation.get("sharpe", 0.0),
                "testSharpe": test.get("sharpe", 0.0),
                "annualReturn": test.get("annual_return", 0.0),
                "drawdown": test.get("max_drawdown", 0.0),
                "quality": quality.get("quality_score", 0.0),
                "correlation": quality.get("return_correlation", 0.0),
                "halfLife": quality.get("half_life", 0.0),
                "residualVolatility": quality.get("residual_volatility", 0.0),
                "targetVol": config.target_vol_annual,
                "maxLeverage": config.max_leverage,
                "breakLimit": config.break_probability_limit,
                "direction": config.signal_direction,
            }
        )

    payload = {
        "summary": portfolio["summary"],
        "selectedPairs": [item["pair"] for item in portfolio["selected"]],
        "weights": {
            item["pair"]: float(weight) for item, weight in zip(portfolio["selected"], portfolio["weights"])
        },
        "pairs": pairs,
        "equity": [float(x) for x in portfolio["equity"]],
        "returns": [float(x) for x in portfolio["returns"]],
        "leverage": [float(x) for x in portfolio["leverage"]],
    }

    html = _html(payload)
    path = here / "algorithm_3d_model.html"
    path.write_text(html, encoding="utf-8")
    return path


def _html(payload: Dict[str, object]) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Regime-Switching Pairs Trading: 3D Algorithm Model</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #07090d;
      color: #f4f7fb;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #scene {{
      position: fixed;
      inset: 0;
    }}
    .panel {{
      position: fixed;
      left: 18px;
      top: 18px;
      width: min(420px, calc(100vw - 36px));
      max-height: calc(100vh - 36px);
      overflow: auto;
      box-sizing: border-box;
      padding: 16px;
      background: rgba(9, 14, 22, 0.78);
      border: 1px solid rgba(255,255,255,0.14);
      backdrop-filter: blur(14px);
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(0,0,0,0.35);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 19px;
      letter-spacing: 0;
      line-height: 1.2;
    }}
    .sub {{
      color: #b7c3d4;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 14px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .metric {{
      border: 1px solid rgba(255,255,255,0.11);
      border-radius: 8px;
      padding: 10px;
      background: rgba(255,255,255,0.055);
    }}
    .label {{
      color: #91a0b5;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
    }}
    .legend {{
      display: grid;
      gap: 7px;
      font-size: 12px;
      color: #d6deea;
    }}
    .legend-row {{
      display: grid;
      grid-template-columns: 14px 1fr;
      gap: 8px;
      align-items: center;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
    .hint {{
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: min(420px, calc(100vw - 36px));
      padding: 10px 12px;
      color: #c7d0dd;
      background: rgba(9, 14, 22, 0.72);
      border: 1px solid rgba(255,255,255,0.13);
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.4;
      backdrop-filter: blur(14px);
    }}
    @media (max-width: 720px) {{
      .panel {{
        top: 10px;
        left: 10px;
        width: calc(100vw - 20px);
        max-height: 42vh;
      }}
      .hint {{
        left: 10px;
        right: 10px;
        bottom: 10px;
      }}
    }}
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <section class="panel">
    <h1>3D Model of the Algorithm</h1>
    <div class="sub">Universe scan, regime engine, break detector, sizing layer, selected pair core, and equity path from the latest run.</div>
    <div class="metrics" id="metrics"></div>
    <div class="legend">
      <div class="legend-row"><span class="swatch" style="background:#4dd2ff"></span><span>Candidate pair towers: height = out-of-sample Sharpe, width glow = quality.</span></div>
      <div class="legend-row"><span class="swatch" style="background:#f7d35c"></span><span>Selected portfolio core: chosen pair(s) and allocation weights.</span></div>
      <div class="legend-row"><span class="swatch" style="background:#ff5d73"></span><span>Break-risk shield: red panels represent relationship-break defense.</span></div>
      <div class="legend-row"><span class="swatch" style="background:#8cff9a"></span><span>Equity ribbon: strategy growth through the test window.</span></div>
    </div>
  </section>
  <div class="hint">Drag to rotate. Scroll to zoom. The model animates slowly so regime layers, leverage coils, and pair towers stay visible.</div>

  <script type="module">
    import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

    const run = {data};
    const canvas = document.getElementById('scene');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: false }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090d);
    scene.fog = new THREE.FogExp2(0x07090d, 0.018);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 300);
    camera.position.set(28, 21, 36);

    const root = new THREE.Group();
    scene.add(root);

    const ambient = new THREE.AmbientLight(0x8fa8c8, 0.85);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 2.6);
    key.position.set(10, 20, 14);
    scene.add(key);
    const rim = new THREE.PointLight(0x4dd2ff, 2.2, 90);
    rim.position.set(-18, 10, -10);
    scene.add(rim);
    const gold = new THREE.PointLight(0xffce62, 1.6, 65);
    gold.position.set(5, 8, 15);
    scene.add(gold);

    const materials = {{
      cyan: new THREE.MeshStandardMaterial({{ color: 0x4dd2ff, emissive: 0x06435a, roughness: 0.34, metalness: 0.42 }}),
      blue: new THREE.MeshStandardMaterial({{ color: 0x5277ff, emissive: 0x111f5c, roughness: 0.38, metalness: 0.36 }}),
      gold: new THREE.MeshStandardMaterial({{ color: 0xf7d35c, emissive: 0x57420b, roughness: 0.28, metalness: 0.54 }}),
      red: new THREE.MeshStandardMaterial({{ color: 0xff5d73, emissive: 0x5a101a, roughness: 0.35, metalness: 0.34 }}),
      green: new THREE.MeshStandardMaterial({{ color: 0x8cff9a, emissive: 0x154d20, roughness: 0.35, metalness: 0.25 }}),
      purple: new THREE.MeshStandardMaterial({{ color: 0xb38cff, emissive: 0x2a1652, roughness: 0.34, metalness: 0.38 }}),
      dark: new THREE.MeshStandardMaterial({{ color: 0x111824, roughness: 0.55, metalness: 0.2 }}),
      glass: new THREE.MeshStandardMaterial({{ color: 0x9ec9ff, transparent: true, opacity: 0.18, roughness: 0.08, metalness: 0.1 }}),
      line: new THREE.LineBasicMaterial({{ color: 0x97a8bf, transparent: true, opacity: 0.36 }}),
      equity: new THREE.LineBasicMaterial({{ color: 0x8cff9a, transparent: true, opacity: 0.95 }}),
      leverage: new THREE.LineBasicMaterial({{ color: 0xf7d35c, transparent: true, opacity: 0.86 }}),
    }};

    function addTextSprite(text, position, scale = 1, color = '#eff6ff') {{
      const c = document.createElement('canvas');
      const ctx = c.getContext('2d');
      c.width = 512;
      c.height = 128;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.font = '700 34px Inter, Arial, sans-serif';
      ctx.fillStyle = color;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.shadowColor = 'rgba(0,0,0,0.8)';
      ctx.shadowBlur = 12;
      ctx.fillText(text, c.width / 2, c.height / 2);
      const texture = new THREE.CanvasTexture(c);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: texture, transparent: true }}));
      sprite.position.copy(position);
      sprite.scale.set(5.0 * scale, 1.25 * scale, 1);
      root.add(sprite);
      return sprite;
    }}

    function cylinderBetween(a, b, radius, material) {{
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      const direction = new THREE.Vector3().subVectors(b, a);
      const length = direction.length();
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 14), material);
      mesh.position.copy(mid);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      root.add(mesh);
      return mesh;
    }}

    function makeRing(radius, y, color, opacity = 0.55) {{
      const pts = [];
      for (let i = 0; i <= 220; i++) {{
        const a = i / 220 * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * radius, y, Math.sin(a) * radius));
      }}
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({{ color, transparent: true, opacity }})
      );
      root.add(line);
      return line;
    }}

    makeRing(18, 0, 0x2a3a52, 0.75);
    makeRing(12, 5.4, 0x31466d, 0.55);
    makeRing(6.2, 10.2, 0x614c20, 0.65);

    addTextSprite('Universe Scan', new THREE.Vector3(-20, 0.4, 0), 0.82, '#b9dfff');
    addTextSprite('Regime Engine', new THREE.Vector3(0, 6.7, -12.2), 0.82, '#c8b6ff');
    addTextSprite('Break-Risk Shield', new THREE.Vector3(12.5, 7.5, 2), 0.72, '#ffc0c8');
    addTextSprite('Selected Portfolio Core', new THREE.Vector3(0, 13.2, 0), 0.8, '#ffe49a');

    const pairGroup = new THREE.Group();
    root.add(pairGroup);
    const pairs = run.pairs || [];
    pairs.forEach((p, i) => {{
      const a = i / Math.max(1, pairs.length) * Math.PI * 2;
      const radius = 17.0 + (i % 3) * 0.9;
      const x = Math.cos(a) * radius;
      const z = Math.sin(a) * radius;
      const sharpe = Math.max(-2, Math.min(3, p.testSharpe || 0));
      const h = 0.7 + Math.max(0, sharpe + 0.25) * 1.45;
      const selected = !!p.selected;
      const mat = selected ? materials.gold : (p.testSharpe >= 0 ? materials.cyan : materials.red);
      const geo = new THREE.BoxGeometry(0.62 + Math.max(0, p.quality || 0) * 0.45, h, 0.62 + Math.max(0, p.quality || 0) * 0.45);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, h / 2, z);
      mesh.rotation.y = -a;
      mesh.userData = p;
      pairGroup.add(mesh);

      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.24 + Math.max(0, p.correlation || 0) * 0.18, 18, 12), selected ? materials.gold : materials.blue);
      cap.position.set(x, h + 0.34, z);
      pairGroup.add(cap);

      if (i < 18 || selected) {{
        addTextSprite(p.pair, new THREE.Vector3(x, h + 0.9, z), selected ? 0.46 : 0.32, selected ? '#ffe49a' : '#dce9ff');
      }}
      if (selected) {{
        cylinderBetween(new THREE.Vector3(x, h + 0.2, z), new THREE.Vector3(0, 11.2, 0), 0.035, materials.gold);
      }}
    }});

    const regimeGroup = new THREE.Group();
    root.add(regimeGroup);
    const regimeNames = ['Bull', 'Bear', 'Transition'];
    const regimeColors = [materials.green, materials.red, materials.purple];
    regimeNames.forEach((name, i) => {{
      const angle = i / 3 * Math.PI * 2 + Math.PI / 6;
      const pos = new THREE.Vector3(Math.cos(angle) * 8.2, 6.0, Math.sin(angle) * 8.2);
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(1.1, 36, 20), regimeColors[i]);
      sphere.position.copy(pos);
      regimeGroup.add(sphere);
      cylinderBetween(pos, new THREE.Vector3(0, 6.0, 0), 0.045, regimeColors[i]);
      addTextSprite(name, pos.clone().add(new THREE.Vector3(0, 1.65, 0)), 0.44, '#f4f7fb');
    }});
    const regimeHub = new THREE.Mesh(new THREE.IcosahedronGeometry(1.55, 2), materials.purple);
    regimeHub.position.set(0, 6.0, 0);
    root.add(regimeHub);

    const shieldGroup = new THREE.Group();
    root.add(shieldGroup);
    for (let i = 0; i < 9; i++) {{
      const angle = i / 9 * Math.PI * 2;
      const panel = new THREE.Mesh(new THREE.BoxGeometry(0.22, 3.2, 2.0), materials.red);
      panel.position.set(Math.cos(angle) * 5.2 + 8.5, 7.5, Math.sin(angle) * 5.2);
      panel.rotation.y = -angle;
      shieldGroup.add(panel);
    }}
    const glass = new THREE.Mesh(new THREE.TorusGeometry(5.3, 0.035, 10, 90), materials.glass);
    glass.position.set(8.5, 7.5, 0);
    glass.rotation.x = Math.PI / 2;
    shieldGroup.add(glass);

    const core = new THREE.Group();
    root.add(core);
    const coreMesh = new THREE.Mesh(new THREE.DodecahedronGeometry(2.1, 1), materials.gold);
    coreMesh.position.set(0, 11.0, 0);
    core.add(coreMesh);
    const selected = run.selectedPairs || [];
    selected.forEach((name, i) => {{
      const angle = i / Math.max(1, selected.length) * Math.PI * 2;
      const weight = (run.weights && run.weights[name]) || 0.25;
      const orb = new THREE.Mesh(new THREE.SphereGeometry(0.48 + weight * 0.9, 32, 18), materials.gold);
      orb.position.set(Math.cos(angle) * 4.2, 11.2 + i * 0.18, Math.sin(angle) * 4.2);
      core.add(orb);
      cylinderBetween(new THREE.Vector3(0, 11.0, 0), orb.position.clone(), 0.04, materials.gold);
      addTextSprite(name + ' ' + Math.round(weight * 100) + '%', orb.position.clone().add(new THREE.Vector3(0, 1.0, 0)), 0.42, '#ffe49a');
    }});

    const equity = run.equity || [];
    if (equity.length > 2) {{
      const minE = Math.min(...equity);
      const maxE = Math.max(...equity);
      const pts = equity.map((v, i) => {{
        const t = i / Math.max(1, equity.length - 1);
        const x = -16 + t * 32;
        const y = -5 + ((v - minE) / Math.max(1e-9, maxE - minE)) * 6.0;
        const z = -13 + Math.sin(t * Math.PI * 8) * 0.65;
        return new THREE.Vector3(x, y, z);
      }});
      const equityLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), materials.equity);
      root.add(equityLine);
      addTextSprite('Equity Path', new THREE.Vector3(-15, 2.2, -13), 0.55, '#bfffc8');
    }}

    const leverage = run.leverage || [];
    if (leverage.length > 2) {{
      const maxL = Math.max(...leverage, 1);
      const pts = leverage.map((v, i) => {{
        const t = i / Math.max(1, leverage.length - 1);
        const angle = t * Math.PI * 14;
        const r = 2.8 + (v / maxL) * 2.0;
        return new THREE.Vector3(Math.cos(angle) * r - 10, 7 + t * 6, Math.sin(angle) * r + 8);
      }});
      root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), materials.leverage));
      addTextSprite('Leverage Coil', new THREE.Vector3(-10, 14, 8), 0.48, '#ffe49a');
    }}

    const pipeline = [
      new THREE.Vector3(-18, 0, 0),
      new THREE.Vector3(-10, 2.5, 0),
      new THREE.Vector3(0, 6, 0),
      new THREE.Vector3(8.5, 7.5, 0),
      new THREE.Vector3(0, 11, 0),
      new THREE.Vector3(0, 15, 0)
    ];
    for (let i = 0; i < pipeline.length - 1; i++) {{
      cylinderBetween(pipeline[i], pipeline[i + 1], 0.075, i === 3 ? materials.gold : materials.blue);
    }}
    addTextSprite('Data → Cointegration → Regimes → Break Filter → Position Sizing → Portfolio', new THREE.Vector3(0, 16.7, 0), 0.62, '#dce9ff');

    const stars = new THREE.BufferGeometry();
    const starPts = [];
    for (let i = 0; i < 900; i++) {{
      starPts.push((Math.random() - 0.5) * 140, (Math.random() - 0.2) * 95, (Math.random() - 0.5) * 140);
    }}
    stars.setAttribute('position', new THREE.Float32BufferAttribute(starPts, 3));
    scene.add(new THREE.Points(stars, new THREE.PointsMaterial({{ color: 0x7f91aa, size: 0.055, transparent: true, opacity: 0.72 }})));

    const metrics = document.getElementById('metrics');
    const s = run.summary || {{}};
    const metricRows = [
      ['Annual Return', pct(s.annual_return)],
      ['Sharpe', num(s.sharpe)],
      ['Max Drawdown', pct(s.max_drawdown)],
      ['Final Equity', num(s.final_equity)],
      ['Avg Leverage', num(s.average_portfolio_leverage)],
      ['Selected Pairs', num(s.selected_pair_count)]
    ];
    metrics.innerHTML = metricRows.map(([label, value]) => `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`).join('');

    function pct(v) {{ return Number.isFinite(v) ? (v * 100).toFixed(2) + '%' : 'n/a'; }}
    function num(v) {{ return Number.isFinite(v) ? Number(v).toFixed(2) : 'n/a'; }}

    let dragging = false;
    let previous = {{ x: 0, y: 0 }};
    let targetRotY = -0.35;
    let targetRotX = 0.08;
    let zoom = 1;

    window.addEventListener('pointerdown', e => {{ dragging = true; previous = {{ x: e.clientX, y: e.clientY }}; }});
    window.addEventListener('pointerup', () => dragging = false);
    window.addEventListener('pointermove', e => {{
      if (!dragging) return;
      const dx = e.clientX - previous.x;
      const dy = e.clientY - previous.y;
      previous = {{ x: e.clientX, y: e.clientY }};
      targetRotY += dx * 0.006;
      targetRotX += dy * 0.004;
      targetRotX = Math.max(-0.75, Math.min(0.75, targetRotX));
    }});
    window.addEventListener('wheel', e => {{
      zoom += e.deltaY * 0.0008;
      zoom = Math.max(0.52, Math.min(1.7, zoom));
    }}, {{ passive: true }});
    window.addEventListener('resize', () => {{
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }});

    function animate(time) {{
      const t = time * 0.001;
      root.rotation.y += (targetRotY - root.rotation.y) * 0.06;
      root.rotation.x += (targetRotX - root.rotation.x) * 0.06;
      pairGroup.rotation.y = Math.sin(t * 0.12) * 0.05;
      regimeGroup.rotation.y = t * 0.18;
      shieldGroup.rotation.y = -t * 0.26;
      core.rotation.y = t * 0.34;
      coreMesh.rotation.x = t * 0.6;
      coreMesh.rotation.y = t * 0.42;
      camera.position.set(28 * zoom, 21 * zoom, 36 * zoom);
      camera.lookAt(0, 6, 0);
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate(0);
  </script>
</body>
</html>
"""

