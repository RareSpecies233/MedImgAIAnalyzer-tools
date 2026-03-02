#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: rgba(148, 163, 184, 0.35);
      --text: #0f172a;
      --muted: #64748b;
      --shadow: 0 10px 30px rgba(2, 6, 23, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .title {{
      font-size: 22px;
      font-weight: 700;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .global-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .global-tip {{
      font-size: 12px;
      color: var(--muted);
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--card-bg);
      box-shadow: var(--shadow);
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 600;
    }}
    .meta {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
      white-space: nowrap;
    }}
    .canvas-wrap {{
      position: relative;
      border: 1px solid rgba(148, 163, 184, 0.4);
      border-radius: 10px;
      overflow: hidden;
      height: 320px;
      background: #f8fafc;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      user-select: none;
    }}
    canvas:active {{ cursor: grabbing; }}
    .overlay {{
      position: absolute;
      left: 10px;
      bottom: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 11px;
      color: #0f172a;
      background: rgba(255, 255, 255, 0.74);
      pointer-events: none;
    }}
    .controls {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .btn-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    button {{
      border: 1px solid rgba(148, 163, 184, 0.45);
      border-radius: 8px;
      background: #ffffff;
      color: #0f172a;
      padding: 5px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    button:hover {{ background: #f1f5f9; }}
    .slider {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #2563eb;
    }}
    .error {{
      color: #b91c1c;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 8px;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="title">GLB 模型浏览</div>
    <div class="subtitle">每行展示 2 个模型，标题为 GLB 文件名；支持旋转/平移/缩放、重置视角、自动旋转、线框、背景切换、导出模型。</div>
    <div class="global-controls">
      <button id="sync-rotate-btn">同步旋转</button>
      <button id="sync-zoom-btn">同步缩放</button>
      <span class="global-tip">开启后不会重置现有状态，仅在后续操作时同步</span>
    </div>
    <div id="boot-error" class="error" style="display:none"></div>
    <section id="grid" class="grid"></section>
  </main>

  <script type="application/json" id="model-data">{models_json}</script>
  <script type="application/json" id="three-modules">{three_modules_json}</script>

  <script type="module">
    const modelData = JSON.parse(document.getElementById('model-data').textContent || '[]');
    const threeModules = JSON.parse(document.getElementById('three-modules').textContent || '{{}}');
    const grid = document.getElementById('grid');
    const bootError = document.getElementById('boot-error');
    const syncRotateBtn = document.getElementById('sync-rotate-btn');
    const syncZoomBtn = document.getElementById('sync-zoom-btn');

    let syncRotateEnabled = false;
    let syncZoomEnabled = false;
    let syncLock = false;
    const viewerApis = [];

    function showBootError(message) {{
      if (!bootError) return;
      bootError.style.display = 'block';
      bootError.textContent = message;
    }}

    function decodeBase64Text(base64) {{
      const binary = atob(base64);
      const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
      return new TextDecoder().decode(bytes);
    }}

    function rewriteThreeImport(code, threeUrl) {{
      return code.replace(/from\\s+['\"]three['\"]\\s*;/g, `from '${{threeUrl}}';`);
    }}

    async function loadThreeDeps() {{
      const revokeList = [];
      const revokeUrls = () => revokeList.forEach((url) => URL.revokeObjectURL(url));
      const toBlobUrl = (code) => {{
        const url = URL.createObjectURL(new Blob([code], {{ type: 'text/javascript' }}));
        revokeList.push(url);
        return url;
      }};

      const hasEmbedded = Boolean(
        threeModules?.three &&
        threeModules?.orbitControls &&
        threeModules?.gltfLoader &&
        threeModules?.bufferGeometryUtils,
      );

      if (hasEmbedded) {{
        try {{
          const threeSource = decodeBase64Text(threeModules.three);
          const orbitSourceRaw = decodeBase64Text(threeModules.orbitControls);
          const gltfSourceRaw = decodeBase64Text(threeModules.gltfLoader);
          const bufferUtilsRaw = decodeBase64Text(threeModules.bufferGeometryUtils);

          const threeUrl = toBlobUrl(threeSource);
          const bufferUtilsUrl = toBlobUrl(rewriteThreeImport(bufferUtilsRaw, threeUrl));
          const orbitUrl = toBlobUrl(rewriteThreeImport(orbitSourceRaw, threeUrl));
          const gltfSource = rewriteThreeImport(gltfSourceRaw, threeUrl)
            .replace(/from\\s+['\"]\\.\\.\\/utils\\/BufferGeometryUtils\\.js['\"]\\s*;/g, `from '${{bufferUtilsUrl}}';`);
          const gltfUrl = toBlobUrl(gltfSource);

          const [THREE, orbitModule, gltfModule] = await Promise.all([
            import(threeUrl),
            import(orbitUrl),
            import(gltfUrl),
          ]);
          return {{ THREE, OrbitControls: orbitModule.OrbitControls, GLTFLoader: gltfModule.GLTFLoader, revokeUrls }};
        }} catch (error) {{
          console.error('内嵌 three 依赖加载失败，改用 CDN 兜底', error);
          revokeUrls();
        }}
      }}

      try {{
        const [THREE, orbitModule, gltfModule] = await Promise.all([
          import('https://unpkg.com/three@0.160.0/build/three.module.js'),
          import('https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js'),
          import('https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js'),
        ]);
        return {{ THREE, OrbitControls: orbitModule.OrbitControls, GLTFLoader: gltfModule.GLTFLoader, revokeUrls: () => {{}} }};
      }} catch (error) {{
        throw new Error('无法加载 three.js 依赖（离线依赖不可用且 CDN 访问失败）。请在项目目录执行脚本，或检查网络后重试。');
      }}
    }}

    let THREE;
    let OrbitControls;
    let GLTFLoader;
    let revokeThreeModuleUrls = () => {{}};

    try {{
      const deps = await loadThreeDeps();
      THREE = deps.THREE;
      OrbitControls = deps.OrbitControls;
      GLTFLoader = deps.GLTFLoader;
      revokeThreeModuleUrls = deps.revokeUrls;
    }} catch (error) {{
      console.error(error);
      showBootError(error instanceof Error ? error.message : 'three.js 依赖加载失败');
    }}

    if (!THREE || !OrbitControls || !GLTFLoader) {{
      throw new Error('three.js 初始化失败');
    }}

    if (!Array.isArray(modelData) || modelData.length === 0) {{
      showBootError('未发现可展示的 GLB 模型数据。');
    }}

    function updateSyncButtons() {{
      if (syncRotateBtn) syncRotateBtn.textContent = syncRotateEnabled ? '关闭同步旋转' : '同步旋转';
      if (syncZoomBtn) syncZoomBtn.textContent = syncZoomEnabled ? '关闭同步缩放' : '同步缩放';
    }}

    function syncFromSource(sourceApi, syncRotateValue, syncZoomValue) {{
      if (syncLock) return;
      if (!syncRotateValue && !syncZoomValue) return;
      syncLock = true;
      try {{
        const sourceSpherical = sourceApi.getSpherical();
        const sourceZoomFactor = sourceApi.getZoomFactor();
        viewerApis.forEach((targetApi) => {{
          if (targetApi === sourceApi) return;
          const nextSpherical = targetApi.getSpherical();
          if (syncRotateValue) {{
            nextSpherical.phi = sourceSpherical.phi;
            nextSpherical.theta = sourceSpherical.theta;
          }}
          if (syncZoomValue) {{
            const nextRadius = targetApi.baseDistance / clamp(sourceZoomFactor, 0.4, 3);
            nextSpherical.radius = clamp(nextRadius, targetApi.minDistance, targetApi.maxDistance);
          }}
          targetApi.applySpherical(nextSpherical);
        }});
      }} finally {{
        syncLock = false;
      }}
    }}

    if (syncRotateBtn) {{
      syncRotateBtn.addEventListener('click', () => {{
        syncRotateEnabled = !syncRotateEnabled;
        updateSyncButtons();
      }});
    }}
    if (syncZoomBtn) {{
      syncZoomBtn.addEventListener('click', () => {{
        syncZoomEnabled = !syncZoomEnabled;
        updateSyncButtons();
      }});
    }}
    updateSyncButtons();

    const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

    function formatZoom(v) {{ return `${{v.toFixed(2)}}x`; }}
    function formatRotation(phi, theta) {{
      const x = Math.round((phi * 180) / Math.PI);
      const y = Math.round((theta * 180) / Math.PI);
      return `${{x}}° / ${{y}}°`;
    }}

    function decodeBase64ToArrayBuffer(base64) {{
      const binary = atob(base64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }}

    function setWireframe(object, enabled) {{
      object.traverse((child) => {{
        if (!child.isMesh) return;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((m) => {{ if ('wireframe' in m) m.wireframe = enabled; }});
      }});
    }}

    function createCard(model) {{
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `
        <div class="head">
          <span>${{model.name}}</span>
          <span class="meta">${{(model.size / 1024).toFixed(1)}} KB</span>
        </div>
        <div class="canvas-wrap">
          <canvas></canvas>
          <div class="overlay">
            <span data-role="zoom">缩放 1.00x</span>
            <span data-role="rotation">旋转 0° / 0°</span>
          </div>
        </div>
        <div class="controls">
          <div class="btn-row">
            <button data-act="reset">重置视角</button>
            <button data-act="auto">自动旋转</button>
            <button data-act="wire">显示线框</button>
            <button data-act="bg">切换背景</button>
            <button data-act="export">导出模型</button>
          </div>
          <label class="slider">
            <span>缩放</span>
            <input data-role="zoom-slider" type="range" min="0.4" max="3" step="0.05" value="1" />
          </label>
          <div data-role="error" class="error" style="display:none"></div>
        </div>
      `;

      const canvas = card.querySelector('canvas');
      const zoomText = card.querySelector('[data-role="zoom"]');
      const rotText = card.querySelector('[data-role="rotation"]');
      const zoomSlider = card.querySelector('[data-role="zoom-slider"]');
      const errorBox = card.querySelector('[data-role="error"]');

      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setClearColor(0xffffff, 0);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
      camera.position.set(0, 0, 4);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.autoRotate = false;
      controls.autoRotateSpeed = 1;

      const hemi = new THREE.HemisphereLight(0xffffff, 0x94a3b8, 0.9);
      scene.add(hemi);
      const dir = new THREE.DirectionalLight(0xffffff, 0.8);
      dir.position.set(4, 6, 4);
      scene.add(dir);

      const modelGroup = new THREE.Group();
      scene.add(modelGroup);

      let baseDistance = 4;
      let center = new THREE.Vector3(0, 0, 0);
      let wireframe = false;
      let autoRotate = false;
      let background = 'light';
      let modelRoot = null;
      let lastSpherical = new THREE.Spherical(4, Math.PI / 2, 0);

      function updateBackground() {{
        const color = background === 'dark' ? 0x0f172a : background === 'blue' ? 0x0b2545 : 0xf8fafc;
        scene.background = new THREE.Color(color);
      }}

      function updateOverlay() {{
        const offset = camera.position.clone().sub(controls.target);
        const sp = new THREE.Spherical().setFromVector3(offset);
        lastSpherical = new THREE.Spherical(sp.radius, sp.phi, sp.theta);
        const zoom = clamp(baseDistance / sp.radius, 0.4, 3);
        zoomText.textContent = `缩放 ${{formatZoom(zoom)}}`;
        rotText.textContent = `旋转 ${{formatRotation(sp.phi, sp.theta)}}`;
      }}

      function getSpherical() {{
        const offset = camera.position.clone().sub(controls.target);
        return new THREE.Spherical().setFromVector3(offset);
      }}

      function applySpherical(nextSpherical) {{
        const nextOffset = new THREE.Vector3().setFromSpherical(nextSpherical);
        camera.position.copy(controls.target.clone().add(nextOffset));
        controls.update();
        updateOverlay();
        const zoomNow = clamp(baseDistance / nextSpherical.radius, 0.4, 3);
        zoomSlider.value = zoomNow.toFixed(2);
      }}

      function getZoomFactor() {{
        const sp = getSpherical();
        return clamp(baseDistance / sp.radius, 0.4, 3);
      }}

      function resize() {{
        const width = Math.max(1, canvas.clientWidth);
        const height = Math.max(1, canvas.clientHeight);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }}

      function setZoomBySlider() {{
        const nextZoom = clamp(parseFloat(zoomSlider.value), 0.4, 3);
        const distance = baseDistance / nextZoom;
        const sp = getSpherical();
        sp.radius = distance;
        applySpherical(sp);
      }}

      function fitCamera(obj) {{
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = THREE.MathUtils.degToRad(camera.fov);
        const distance = (maxDim / 2) / Math.tan(fov / 2);
        baseDistance = Math.max(1, distance * 1.4);

        controls.target.copy(center);
        const direction = new THREE.Vector3(0.2, 0.2, 1).normalize();
        camera.position.copy(center.clone().add(direction.multiplyScalar(baseDistance)));
        controls.minDistance = baseDistance * 0.4;
        controls.maxDistance = baseDistance * 3;
        controls.update();
        zoomSlider.value = '1';
        updateOverlay();
      }}

      function resetView() {{
        controls.target.copy(center);
        const direction = new THREE.Vector3(0.2, 0.2, 1).normalize();
        camera.position.copy(center.clone().add(direction.multiplyScalar(baseDistance)));
        controls.update();
        zoomSlider.value = '1';
        updateOverlay();
      }}

      const viewerApi = {{
        baseDistance: 4,
        minDistance: 0.1,
        maxDistance: 2000,
        getSpherical,
        getZoomFactor,
        applySpherical,
      }};
      viewerApis.push(viewerApi);

      function downloadRawGlb() {{
        const bytes = decodeBase64ToArrayBuffer(model.base64);
        const blob = new Blob([bytes], {{ type: 'model/gltf-binary' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = model.name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      card.querySelector('[data-act="reset"]').addEventListener('click', resetView);
      card.querySelector('[data-act="auto"]').addEventListener('click', (e) => {{
        autoRotate = !autoRotate;
        controls.autoRotate = autoRotate;
        e.currentTarget.textContent = autoRotate ? '停止自动旋转' : '自动旋转';
      }});
      card.querySelector('[data-act="wire"]').addEventListener('click', (e) => {{
        wireframe = !wireframe;
        if (modelRoot) setWireframe(modelRoot, wireframe);
        e.currentTarget.textContent = wireframe ? '隐藏线框' : '显示线框';
      }});
      card.querySelector('[data-act="bg"]').addEventListener('click', () => {{
        background = background === 'light' ? 'dark' : background === 'dark' ? 'blue' : 'light';
        updateBackground();
      }});
      card.querySelector('[data-act="export"]').addEventListener('click', downloadRawGlb);

      zoomSlider.addEventListener('input', setZoomBySlider);

      controls.addEventListener('change', () => {{
        const prev = new THREE.Spherical(lastSpherical.radius, lastSpherical.phi, lastSpherical.theta);
        updateOverlay();
        const sp = getSpherical();
        zoomSlider.value = clamp(baseDistance / sp.radius, 0.4, 3).toFixed(2);
        const zoomChanged = Math.abs(sp.radius - prev.radius) > 0.001;
        const rotateChanged =
          Math.abs(sp.phi - prev.phi) > 0.001 ||
          Math.abs(sp.theta - prev.theta) > 0.001;
        if (!syncLock && ((syncRotateEnabled && rotateChanged) || (syncZoomEnabled && zoomChanged))) {{
          syncFromSource(viewerApi, syncRotateEnabled && rotateChanged, syncZoomEnabled && zoomChanged);
        }}
      }});

      const resizeObserver = new ResizeObserver(() => resize());
      resizeObserver.observe(canvas);
      resize();
      updateBackground();

      const loader = new GLTFLoader();
      try {{
        const buf = decodeBase64ToArrayBuffer(model.base64);
        loader.parse(buf, '', (gltf) => {{
          modelRoot = gltf.scene;
          modelGroup.add(modelRoot);
          fitCamera(modelRoot);
          viewerApi.baseDistance = baseDistance;
          viewerApi.minDistance = controls.minDistance;
          viewerApi.maxDistance = controls.maxDistance;
        }}, (error) => {{
          console.error(error);
          errorBox.style.display = 'block';
          errorBox.textContent = '模型解析失败';
        }});
      }} catch (error) {{
        console.error(error);
        errorBox.style.display = 'block';
        errorBox.textContent = '模型解码失败';
      }}

      let rafId = null;
      const renderLoop = () => {{
        controls.update();
        renderer.render(scene, camera);
        rafId = requestAnimationFrame(renderLoop);
      }};
      rafId = requestAnimationFrame(renderLoop);

      card.__dispose = () => {{
        if (rafId !== null) cancelAnimationFrame(rafId);
        resizeObserver.disconnect();
        controls.dispose();
        renderer.dispose();
        const idx = viewerApis.indexOf(viewerApi);
        if (idx >= 0) viewerApis.splice(idx, 1);
      }};

      return card;
    }}

    modelData.forEach((model) => grid.appendChild(createCard(model)));

    window.addEventListener('beforeunload', () => {{
      [...grid.children].forEach((card) => {{
        if (typeof card.__dispose === 'function') card.__dispose();
      }});
      revokeThreeModuleUrls();
    }});
  </script>
</body>
</html>
"""


def collect_glb_files(inputs: list[str], recursive: bool) -> list[Path]:
    results: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == '.glb':
            results.append(path)
            continue
        if path.is_dir():
            pattern = '**/*.glb' if recursive else '*.glb'
            results.extend(sorted(path.glob(pattern)))
            continue
        if any(ch in item for ch in '*?[]'):
            base = Path('.').resolve()
            results.extend(sorted(base.glob(item)))
    uniq = []
    seen = set()
    for file in results:
        key = str(file.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(file)
    return uniq


def to_model_payload(file: Path) -> dict[str, object]:
    raw = file.read_bytes()
    b64 = base64.b64encode(raw).decode('ascii')
    return {
        'name': file.name,
        'size': len(raw),
        'base64': b64,
    }


def load_embedded_three_modules() -> dict[str, str]:
    candidates = []
    script_path = Path(__file__).resolve()
    candidates.append(script_path.parent.parent)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    def find_three_root() -> Path | None:
        seen: set[str] = set()
        for base in candidates:
            key = str(base)
            if key in seen:
                continue
            seen.add(key)
            root = base / 'node_modules' / 'three'
            if root.is_dir():
                return root
        return None

    three_root = find_three_root()
    if not three_root:
        return {}

    paths = {
        'three': three_root / 'build' / 'three.module.js',
        'orbitControls': three_root / 'examples' / 'jsm' / 'controls' / 'OrbitControls.js',
        'gltfLoader': three_root / 'examples' / 'jsm' / 'loaders' / 'GLTFLoader.js',
        'bufferGeometryUtils': three_root / 'examples' / 'jsm' / 'utils' / 'BufferGeometryUtils.js',
    }
    if not all(path.is_file() for path in paths.values()):
        return {}

    payload: dict[str, str] = {}
    for key, path in paths.items():
        payload[key] = base64.b64encode(path.read_bytes()).decode('ascii')
    return payload


def build_html(models: list[dict[str, object]], title: str, three_modules: dict[str, str]) -> str:
    return HTML_TEMPLATE.format(
        title=title,
        models_json=json.dumps(models, ensure_ascii=False),
    three_modules_json=json.dumps(three_modules, ensure_ascii=False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='将多个 GLB 文件打包为单个 HTML（每行 2 个模型，保留三维查看交互功能）。',
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='输入路径：可传单个 glb 文件、目录，或通配符（如 ./models/*.glb）。',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='packed_glb_viewer.html',
        help='输出 HTML 文件路径（默认：packed_glb_viewer.html）。',
    )
    parser.add_argument(
        '--title',
        default='GLB 模型浏览',
        help='输出 HTML 页面标题。',
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='当输入为目录时，递归查找子目录中的 .glb 文件。',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    glb_files = collect_glb_files(args.inputs, recursive=args.recursive)
    glb_files = [f for f in glb_files if f.is_file() and f.suffix.lower() == '.glb']
    if not glb_files:
        print('未找到任何 .glb 文件，请检查输入路径。')
        return 1

    models = [to_model_payload(file) for file in glb_files]
    three_modules = load_embedded_three_modules()
    html = build_html(models, title=args.title, three_modules=three_modules)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')

    total_size_mb = sum(m['size'] for m in models) / (1024 * 1024)
    print(f'已打包 {len(models)} 个 GLB 文件到: {output}')
    print(f'模型总大小: {total_size_mb:.2f} MB')
    if three_modules:
        print('three.js 依赖来源: 已内嵌本地 node_modules（离线可用）')
    else:
        print('three.js 依赖来源: CDN（请保证可访问 unpkg）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
