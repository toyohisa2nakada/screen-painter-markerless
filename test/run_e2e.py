"""End-to-end check in headless Chromium: PC page + phone page in one browser, the phone camera
replaced by a pre-rendered (perspective-warped) view of the frozen demo screen, real PeerJS/WebRTC
between the two tabs. Verifies tracking and that a touch lands where it should on the screen."""
import asyncio, base64, json, os, subprocess, sys, tempfile
from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
GPU = os.environ.get("GPU", "0") == "1"
PHONE_W, PHONE_H = 640, 480
# same "far" ground truth as synth.html: screen (1280x720) -> phone quad
GT_SRC = [0, 0, 1280, 0, 1280, 720, 0, 720]
GT_DST = [150, 110, 520, 130, 500, 370, 130, 340]

RENDER_JS = """
async ([src, dst, W, H]) => {
  const game = document.createElement('canvas'); game.width = 1280; game.height = 720;
  const g = new DemoGame(game, { texture: true, frozen: true }); g.draw(performance.now());
  const Hgt = Homography.fromPoints(new Float32Array(src), new Float32Array(dst));
  let seed = 7; const rnd = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const cv = warpImage(game, Hgt, W, H, rnd);
  return cv.toDataURL('image/png');
}
"""

async def main():
    async with async_playwright() as p:
        tmp = tempfile.mkdtemp()
        # 1) render the fake phone view using the synth page's helpers
        b0 = await p.chromium.launch()
        pg = await b0.new_page()
        await pg.goto(f"{BASE}/test/synth.html?levels=320")  # loads helpers; its own run is irrelevant
        data_url = await pg.evaluate(RENDER_JS, [GT_SRC, GT_DST, PHONE_W, PHONE_H])
        await b0.close()
        png = os.path.join(tmp, "phone.png")
        with open(png, "wb") as f:
            f.write(base64.b64decode(data_url.split(",", 1)[1]))
        y4m = os.path.join(tmp, "phone.y4m")
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", png, "-t", "4", "-r", "15",
                               "-pix_fmt", "yuv420p", y4m])

        # 2) launch with the fake camera
        args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={y4m}", "--autoplay-policy=no-user-gesture-required"]
        if GPU:
            args += ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan"]
        browser = await p.chromium.launch(args=args)
        ctx = await browser.new_context(viewport={"width": 1400, "height": 800})
        pc = await ctx.new_page()
        pc.on("console", lambda m: print("[pc]", m.text[:200]) if m.type in ("warning", "error") else None)
        await pc.goto(f"{BASE}/pc.html?frozen=1" + ("" if GPU else "&gpu=0"))
        if not GPU:
            pass
        await pc.wait_for_function("window.__sp && window.__sp.refs.length > 0", timeout=120000)
        print("pc: model ready, refs:", await pc.evaluate("window.__sp.refs.length"))

        phone = await ctx.new_page()
        phone.on("console", lambda m: print("[phone]", m.text[:200]) if m.type in ("warning", "error") else None)
        await phone.set_viewport_size({"width": 400, "height": 700})
        await phone.goto(f"{BASE}/phone.html")
        await phone.click("#start")
        await phone.wait_for_function("document.getElementById('status').textContent.includes('接続済み')", timeout=60000)
        print("phone:", await phone.evaluate("document.getElementById('status').textContent"))

        # 3) wait for tracking
        await pc.wait_for_function("window.__sp.Hcur !== null && window.__sp.stats.inliers >= 20", timeout=120000)
        st = await pc.evaluate("({inliers: __sp.stats.inliers, matches: __sp.stats.matches, ms: __sp.stats.msExtract + __sp.stats.msMatch, rtt: __sp.clock.rtt, quad: __sp.quad})")
        print("pc tracking:", json.dumps(st))

        # 4) touch the phone at the screen centre as the phone sees it, drag a short line
        H = await pc.evaluate("Array.from(__sp.Hcur)")
        cx = sum(GT_DST[0::2]) / 4; cy = sum(GT_DST[1::2]) / 4  # ~ phone px of the screen centre
        # map phone px -> client px on the phone page (object-fit: contain)
        rect = await phone.evaluate("(() => { const r = document.getElementById('video').getBoundingClientRect(); const v = document.getElementById('video'); return [r.left, r.top, r.width, r.height, v.videoWidth, v.videoHeight]; })()")
        L, T, RW, RH, VW, VH = rect
        s = min(RW / VW, RH / VH); ox = L + (RW - VW * s) / 2; oy = T + (RH - VH * s) / 2
        def to_client(px, py): return (ox + px * s, oy + py * s)
        await pc.evaluate("document.getElementById('clear').click()")
        x0, y0 = to_client(cx - 40, cy); x1, y1 = to_client(cx + 40, cy)
        await phone.mouse.move(x0, y0); await phone.mouse.down()
        for i in range(1, 11):
            await phone.mouse.move(x0 + (x1 - x0) * i / 10, y0 + (y1 - y0) * i / 10); await asyncio.sleep(0.03)
        await phone.mouse.up()
        await asyncio.sleep(0.8)

        # 5) where did the stroke land on the screen?
        bbox = await pc.evaluate("""() => {
          const c = __sp.paintCanvas, d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let minx = 1e9, miny = 1e9, maxx = -1, maxy = -1, n = 0;
          for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) if (d[(y * c.width + x) * 4 + 3] > 0) { n++; minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
          return { n, minx, miny, maxx, maxy };
        }""")
        print("stroke bbox on screen:", bbox)
        await pc.screenshot(path="/tmp/e2e_pc.png")
        await browser.close()
        # expected: a horizontal-ish line through roughly the screen centre (640,360); GT_DST is a tilted
        # quad so its centroid isn't exactly the centre -> allow a generous window.
        ok = bbox["n"] > 50 and 350 < (bbox["minx"] + bbox["maxx"]) / 2 < 930 and 200 < (bbox["miny"] + bbox["maxy"]) / 2 < 520 \
             and (bbox["maxx"] - bbox["minx"]) > 100 and (bbox["maxy"] - bbox["miny"]) < 120
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

asyncio.run(main())
