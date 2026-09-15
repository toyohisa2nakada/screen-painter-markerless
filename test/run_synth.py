"""Headless check: opens public/test/synth.html and prints the recovered-homography errors."""
import json, sys, asyncio
from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/test/synth.html"

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan"])
        page = await browser.new_page()
        page.on("console", lambda m: print("[console]", m.text) if m.type in ("warning", "error") else None)
        await page.goto(URL)
        await page.wait_for_function("window.testResults !== null", timeout=300000)
        res = await page.evaluate("window.testResults")
        print(json.dumps(res, indent=1))
        await page.screenshot(path="/tmp/synth.png", full_page=True)
        await browser.close()
        ok = "error" not in res and all(v["maxErrPx"] is not None and v["maxErrPx"] < 12 and v["sane"] for v in res["results"].values())
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

asyncio.run(main())
