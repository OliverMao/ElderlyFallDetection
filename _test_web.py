"""Self-contained smoke test: boots the web server in-process, then exercises it."""
import asyncio
import json
import sys
import threading
import time

import httpx
import uvicorn
import websockets


def p(*a):
    print(*a, flush=True)


async def main():
    import server as srv

    port = 8780
    config = uvicorn.Config(srv.app, host="127.0.0.1", port=port, log_level="error")
    uvr = uvicorn.Server(config)
    threading.Thread(target=uvr.run, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    # wait for readiness
    async with httpx.AsyncClient(base_url=base, timeout=5) as c:
        for _ in range(30):
            try:
                await c.get("/api/state")
                break
            except Exception:
                await asyncio.sleep(0.5)
        else:
            p("server did not come up"); sys.exit(1)
        p("server up")

        s = (await c.get("/api/state")).json()
        p("state:", s)
        srcs = (await c.get("/api/sources")).json()
        p("sources:", srcs)

        r = (await c.post("/api/start", json={
            "source": "sample_videos/fall-0001_seed-1203328030.mp4",
            "imgsz": 480, "stride": 3
        })).json()
        p("start:", r.get("ok"), r.get("error", ""))

        t0 = time.time()
        n_frames = 0
        first_alert = None
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            while time.time() - t0 < 50:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg["type"] == "frame":
                    n_frames += 1
                    if n_frames == 1:
                        p("first frame:", len(msg["jpeg"]), "b64 chars, tracks:",
                          [(t["id"], t["state"]) for t in msg["meta"]["tracks"]])
                    for a in msg["meta"]["alerts"]:
                        if first_alert is None:
                            first_alert = (msg["meta"]["frame"], a["state"], a["track_id"])
                            p("first alert:", first_alert)
                    if n_frames % 15 == 0:
                        p(f"...{n_frames} frames @ {msg['meta']['fps']} fps, {time.time()-t0:.0f}s")
                    if first_alert is not None and n_frames > 10:
                        break
        p(f"received {n_frames} frames in {time.time()-t0:.1f}s; first alert at: {first_alert}")

        s = (await c.get("/api/state")).json()
        p("state after:", s["state"], s["frames"], "frames @", s["fps"], "fps")
        await c.post("/api/stop")
        s = (await c.get("/api/state")).json()
        p("state after stop:", s["state"])

    ok = n_frames > 5 and first_alert is not None
    p("SMOKE TEST", "PASSED" if ok else "FAILED")
    uvr.should_exit = True
    sys.exit(0 if ok else 1)


asyncio.run(main())
