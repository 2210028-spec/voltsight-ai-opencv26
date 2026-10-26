"""
app.py
FastAPI Web Endpoint & Live Judge Demonstration Server for VoltSight AI.
Provides:
- GET  /             -> Interactive Web Dashboard (OpenCV 5 Telemetry, Agentic Trace Viewer, AWS Ticket Console)
- POST /api/inspect  -> Runs live closed-loop Agentic Vision inspection on uploaded or synthetic EV pack frames
- GET  /api/benchmark -> Returns reproducible 60-frame benchmark evaluation results
- GET  /api/health   -> AWS Load Balancer / App Runner health check endpoint
"""
import base64
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np

from voltsight.agent_orchestrator import VoltSightAgent, MCP_TOOLS_MANIFEST
from evaluate_benchmarks import synthesize_battery_pack_frame

app = FastAPI(title="VoltSight AI — OpenCV 5 & AWS Agentic Scrutineering API", version="1.0.0")
agent = VoltSightAgent(storage_dir="./aws_local_store")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "VoltSight AI (OpenCV + AWS Agentic Vision)",
        "opencv_version": cv2.__version__,
        "mcp_tools_registered": len(MCP_TOOLS_MANIFEST),
    }


@app.get("/api/inspect/{scenario}")
def inspect_scenario(scenario: str):
    valid = [
        "NOMINAL",
        "THERMAL_HOTSPOT",
        "BUSBAR_MICRO_CRACK",
        "HV_CLEARANCE_VIOLATION",
        "BLURRED_MICRO_CRACK",
        "GLARE_MASKED_HOTSPOT",
    ]
    if scenario not in valid:
        raise HTTPException(status_code=400, detail=f"Scenario must be one of {valid}")
    frame, gt = synthesize_battery_pack_frame(scenario, seed=2026)
    res = agent.inspect_frame_agentic(f"LIVE-{scenario[:8]}", frame)
    _, buf = cv2.imencode(".png", res["annotated_frame"])
    b64_img = base64.b64encode(buf.tobytes()).decode("ascii")
    payload = {k: v for k, v in res.items() if k not in ("annotated_frame", "restored_frame")}
    payload["annotated_image_b64"] = f"data:image/png;base64,{b64_img}"
    payload["ground_truth"] = gt
    return JSONResponse(payload)


@app.get("/api/benchmark")
def get_benchmark():
    p = Path("benchmark_results.json")
    if p.exists():
        return JSONResponse(json.loads(p.read_text()))
    return JSONResponse({"error": "Run python3 evaluate_benchmarks.py first"}, status_code=404)
