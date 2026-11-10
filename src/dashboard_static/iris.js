// The patient's iris: the one moving thing on the page.
// Colour follows mood (blue calm or happy, grey neutral, red upset); the fibres ripple and
// the pupil widens with how loud and agitated the patient sounds right now.
import * as THREE from "./three.module.min.js";

const FIBRES = 1100;
const SEGMENTS = 7;

const fibreVertex = /* glsl */ `
  attribute float aAngle;
  attribute float aT;
  attribute float aSeed;
  uniform float uTime;
  uniform float uPupil;
  uniform float uAgit;
  uniform float uLevel;
  varying float vT;
  varying float vSeed;
  void main() {
    float r = mix(uPupil, 0.9, aT);
    float wave = sin(aAngle * 7.0 + uTime * 1.1 + aSeed * 6.283) * 0.5
               + sin(aAngle * 13.0 - uTime * 0.7 + aSeed * 2.0) * 0.5;
    float ang = aAngle + (wave * 0.02 + (aSeed - 0.5) * 0.028) * aT * (1.0 + uAgit * 2.2);
    r += sin(aT * 11.0 - uTime * 7.0 + aSeed * 3.0) * 0.016 * uLevel * aT;
    vT = aT;
    vSeed = aSeed;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(cos(ang) * r, sin(ang) * r, 0.0, 1.0);
  }
`;

const fibreFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uDeep;
  uniform float uGlow;
  varying float vT;
  varying float vSeed;
  void main() {
    float collarette = exp(-pow((vT - 0.27) / 0.07, 2.0)) * 0.9;
    vec3 c = mix(uColor * (1.15 + collarette), uDeep, smoothstep(0.45, 1.0, vT));
    float a = smoothstep(0.0, 0.07, vT) * (1.0 - smoothstep(0.84, 1.0, vT));
    a *= (0.28 + 0.72 * vSeed) * uGlow;
    gl_FragColor = vec4(c, a);
  }
`;

const discFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uDeep;
  uniform vec3 uRim;
  varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    vec3 c = mix(uColor * 0.55, uDeep * 0.6, smoothstep(0.3, 0.9, r));
    c = mix(c, uRim, smoothstep(0.86, 0.97, r));          // the dark limbal ring
    float a = 1.0 - smoothstep(0.975, 1.0, r);
    gl_FragColor = vec4(c, a);
  }
`;

const pupilFragment = /* glsl */ `
  uniform vec3 uColor;
  varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    gl_FragColor = vec4(uColor, 1.0 - smoothstep(0.93, 1.0, r));
  }
`;

const uvVertex = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;

function cssColor(name) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return new THREE.Color(v || "#888888");
}

export class Iris {
  constructor(canvas) {
    this.canvas = canvas;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.target = { score: null, arousal: 0 };
    this.level = 0; // live voice, 0..1
    this.levelTarget = 0;
    this.ok = false;
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    } catch (e) {
      canvas.replaceWith(Object.assign(document.createElement("p"), {
        className: "why", textContent: "This browser cannot draw the 3D iris (WebGL is off). Everything else works.",
      }));
      return;
    }
    this.ok = true;
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    this.camera.position.z = 2;
    this.colors = this.readColors();
    this.color = this.colors.mid.clone();
    this.build();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    document.addEventListener("visibilitychange", () => this.loop());
    this.clock = new THREE.Clock();
    this.t = 0;
    this.loop();
  }

  readColors() {
    return {
      pos: cssColor("--mood-pos"),
      mid: cssColor("--mood-mid"),
      neg: cssColor("--mood-neg"),
      pupil: cssColor("--pupil"),
      rim: cssColor("--pupil").lerp(new THREE.Color("#000000"), 0.3),
    };
  }

  themeChanged() {
    if (!this.ok) return;
    this.colors = this.readColors();
    this.pupil.material.uniforms.uColor.value.copy(this.colors.pupil);
    this.disc.material.uniforms.uRim.value.copy(this.colors.rim);
    this.draw();
  }

  build() {
    const n = FIBRES * SEGMENTS * 2;
    const angle = new Float32Array(n);
    const tt = new Float32Array(n);
    const seed = new Float32Array(n);
    const pos = new Float32Array(n * 3);
    let k = 0;
    for (let f = 0; f < FIBRES; f++) {
      const a = (f / FIBRES) * Math.PI * 2 + (Math.random() - 0.5) * 0.01;
      const s = Math.random();
      const reach = 0.75 + Math.random() * 0.25; // not every fibre runs the full width
      for (let g = 0; g < SEGMENTS; g++) {
        for (const e of [g, g + 1]) {
          angle[k] = a;
          tt[k] = (e / SEGMENTS) * reach;
          seed[k] = s;
          k++;
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aAngle", new THREE.BufferAttribute(angle, 1));
    geo.setAttribute("aT", new THREE.BufferAttribute(tt, 1));
    geo.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
    this.uniforms = {
      uTime: { value: 0 },
      uPupil: { value: 0.3 },
      uAgit: { value: 0 },
      uLevel: { value: 0 },
      uColor: { value: this.color.clone() },
      uDeep: { value: this.color.clone().multiplyScalar(0.35) },
      uGlow: { value: 0.9 },
    };
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(0.97, 96),
      new THREE.ShaderMaterial({
        vertexShader: uvVertex,
        fragmentShader: discFragment,
        transparent: true,
        uniforms: { uColor: this.uniforms.uColor, uDeep: this.uniforms.uDeep, uRim: { value: this.colors.rim.clone() } },
      }),
    );
    const fibres = new THREE.LineSegments(
      geo,
      new THREE.ShaderMaterial({
        vertexShader: fibreVertex,
        fragmentShader: fibreFragment,
        uniforms: this.uniforms,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    fibres.frustumCulled = false;
    const pupil = new THREE.Mesh(
      new THREE.CircleGeometry(1, 72),
      new THREE.ShaderMaterial({
        vertexShader: uvVertex,
        fragmentShader: pupilFragment,
        transparent: true,
        uniforms: { uColor: { value: this.colors.pupil.clone() } },
      }),
    );
    pupil.position.z = 0.1;
    // the light from the window: two reflections make it read as a living eye
    const shine = (x, y, r, a) => {
      const m = new THREE.Mesh(
        new THREE.CircleGeometry(r, 40),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: a }),
      );
      m.position.set(x, y, 0.2);
      return m;
    };
    this.scene.add(disc, fibres, pupil, shine(-0.2, 0.22, 0.065, 0.8), shine(0.17, -0.19, 0.022, 0.4));
    this.disc = disc;
    this.pupil = pupil;
  }

  resize() {
    if (!this.ok) return;
    const w = this.canvas.clientWidth || 300;
    const h = this.canvas.clientHeight || 300;
    this.renderer.setSize(w, h, false);
    const a = w / h;
    Object.assign(this.camera, { left: -a, right: a, top: 1, bottom: -1 });
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  /** score -2..2 (null: nobody on the line), arousal 0..1 */
  setMood(score, arousal) {
    this.target = { score, arousal: arousal || 0 };
    if (this.reduced && this.ok) this.settle(); // no WebGL: nothing to draw
  }

  /** live loudness from the bridge, dBFS */
  setVoice(db, speaking) {
    this.levelTarget = speaking ? Math.max(0, Math.min(1, (db + 50) / 38)) : 0;
  }

  goal() {
    const { score } = this.target;
    const c = this.colors;
    if (score == null) return c.mid.clone().lerp(new THREE.Color("#000000"), 0.15);
    const t = Math.max(-1, Math.min(1, score / 2));
    return t >= 0 ? c.mid.clone().lerp(c.pos, t) : c.mid.clone().lerp(c.neg, -t);
  }

  settle() {
    this.color.copy(this.goal());
    this.apply(0);
    this.draw();
  }

  apply(dt) {
    const u = this.uniforms;
    const k = dt ? 1 - Math.exp(-dt / 0.7) : 1;
    this.color.lerp(this.goal(), k);
    u.uColor.value.copy(this.color);
    u.uDeep.value.copy(this.color).multiplyScalar(0.32);
    const attack = this.levelTarget > this.level ? 0.08 : 0.45;
    this.level += (this.levelTarget - this.level) * (dt ? 1 - Math.exp(-dt / attack) : 1);
    u.uLevel.value = this.level;
    const ar = this.target.arousal;
    u.uAgit.value += (ar - u.uAgit.value) * (dt ? 1 - Math.exp(-dt / 0.8) : 1);
    const pupil = 0.27 + 0.08 * u.uAgit.value + 0.035 * this.level;
    u.uPupil.value += (pupil - u.uPupil.value) * (dt ? 1 - Math.exp(-dt / 0.25) : 1);
    this.pupil.scale.setScalar(u.uPupil.value);
  }

  draw() {
    if (this.ok) this.renderer.render(this.scene, this.camera);
  }

  loop() {
    if (!this.ok || this.reduced || this.running || document.hidden) return;
    this.running = true;
    const frame = () => {
      if (document.hidden) { this.running = false; return; }
      const dt = Math.min(0.1, this.clock.getDelta());
      const pace = this.target.score == null ? 0.35 : 1 + this.uniforms.uAgit.value * 1.5;
      this.t += dt * pace;
      this.uniforms.uTime.value = this.t;
      this.apply(dt);
      this.draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
}
