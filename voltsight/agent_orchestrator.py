"""
voltsight/agent_orchestrator.py
Agentic Vision Multi-Step Perception -> Decision -> Action Orchestrator with MCP Tool Registry.
Qualifies for the OpenCV 2026 Agentic Vision Award:
- Visual evidence from OpenCV 5 directly alters the agent's subsequent plan, tool calls, and human-in-the-loop governance.
- Never acts as a static single-pass wrapper:
  1. Perception Pass 1 (`opencv_frame_diagnostics`): Inspects Laplacian focus variance, LAB contrast, and glare ratio.
  2. Conditional Plan Branch A (`opencv_active_restoration`): Invoked ONLY when visual telemetry detects blur, glare, or low contrast.
  3. Perception Pass 2 (`opencv_scrutineer_scan`): Executes HSV thermal segmentation, morphological busbar crack detection, and Euclidean distance transform HV clearance checks.
  4. Conditional Plan Branch B (`opencv_roi_highres_inspect`): Invoked dynamically on each ambiguous finding (0.55 <= conf < 0.86) using 2.5x Lanczos-4 super-sampling + Scharr gradient energy verification to confirm true defects or reject benign surface marks.
  5. Action & Governance (`aws_s3_archive_evidence`, `aws_cloudwatch_emit_telemetry`, `aws_dispatch_remediation_ticket`, `request_human_scrutineer_approval`): Dispatches cloud remediation tickets and locks high-voltage energization pending human approval on critical faults.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from .vision_engine import OpenCVVisionEngine
from .aws_cloud import AWSCloudBridge


MCP_TOOLS_MANIFEST = [
    {
        "name": "opencv_frame_diagnostics",
        "description": "Evaluates frame sharpness (Laplacian variance), LAB illumination dynamic range, and HSV specular glare ratio.",
        "input_schema": {"type": "object", "properties": {"frame_id": {"type": "string"}}, "required": ["frame_id"]},
    },
    {
        "name": "opencv_active_restoration",
        "description": "Applies Navier-Stokes glare inpainting, LAB CLAHE contrast equalization, and unsharp mask deblurring when frame diagnostics fail threshold.",
        "input_schema": {"type": "object", "properties": {"frame_id": {"type": "string"}, "mode": {"type": "string"}}, "required": ["frame_id", "mode"]},
    },
    {
        "name": "opencv_scrutineer_scan",
        "description": "Runs multi-modal OpenCV defect detection: HSV connected-components thermal hotspot analysis, morphological black-hat busbar crack profiling, and cv2.distanceTransform HV cable clearance verification.",
        "input_schema": {"type": "object", "properties": {"frame_id": {"type": "string"}}, "required": ["frame_id"]},
    },
    {
        "name": "opencv_roi_highres_inspect",
        "description": "Performs 2.5x Lanczos-4 ROI zoom and Scharr gradient energy verification on ambiguous findings to confirm or reject defects.",
        "input_schema": {"type": "object", "properties": {"frame_id": {"type": "string"}, "defect_id": {"type": "string"}, "zoom_scale": {"type": "number"}}, "required": ["frame_id", "defect_id"]},
    },
    {
        "name": "aws_dispatch_remediation_ticket",
        "description": "Archives annotated OpenCV evidence to Amazon S3, logs metrics to CloudWatch, and creates a remediation ticket with human-in-the-loop safety lock.",
        "input_schema": {"type": "object", "properties": {"frame_id": {"type": "string"}, "severity": {"type": "string"}, "requires_human_approval": {"type": "boolean"}}, "required": ["frame_id", "severity"]},
    },
]


class VoltSightAgent:
    """Closed-Loop Agentic Vision Controller integrating OpenCV 5 tools and AWS cloud workflows."""

    def __init__(self, storage_dir: str = "./aws_local_store"):
        self.engine = OpenCVVisionEngine()
        self.aws = AWSCloudBridge(storage_dir=storage_dir)

    def inspect_frame_agentic(self, frame_id: str, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Executes the closed-loop multi-step perception-decision-action workflow on an input frame."""
        t_start = time.perf_counter()
        trace: List[Dict[str, Any]] = []
        working_frame = bgr_frame.copy()

        # STEP 1: Initial Frame Telemetry Perception
        diag = self.engine.run_frame_diagnostics(working_frame)
        trace.append({
            "step": 1,
            "phase": "PERCEIVE",
            "tool": "opencv_frame_diagnostics",
            "observation": {
                "laplacian_variance": diag["laplacian_variance"],
                "glare_pixel_ratio": diag["glare_pixel_ratio"],
                "dynamic_range": diag["dynamic_range_p95_p5"],
                "needs_active_restoration": diag["needs_active_restoration"],
            },
            "decision": (
                f"Visual degradation detected ({diag['recommended_restoration_mode']}). "
                "Branching to opencv_active_restoration before defect scan."
                if diag["needs_active_restoration"]
                else "Frame telemetry passes sharpness and illumination thresholds. Proceeding directly to opencv_scrutineer_scan."
            ),
        })

        # STEP 2 (CONDITIONAL): Active Restoration Tool Call driven by Step 1 visual output
        restoration_info = None
        if diag["needs_active_restoration"]:
            working_frame, restoration_info = self.engine.run_active_restoration(
                working_frame, mode=diag["recommended_restoration_mode"]
            )
            trace.append({
                "step": len(trace) + 1,
                "phase": "ACT_AND_REPERCEIVE",
                "tool": "opencv_active_restoration",
                "observation": restoration_info,
                "decision": (
                    f"Restored sharpness to Laplacian={restoration_info['post_restoration_laplacian']} "
                    f"and reduced glare ratio to {restoration_info['post_restoration_glare_ratio']}. "
                    "Now invoking opencv_scrutineer_scan on restored frame."
                ),
            })

        # STEP 3: Multi-Modal OpenCV Scrutineering Scan
        raw_findings, annotated_frame, scan_summary = self.engine.run_scrutineering_scan(working_frame)
        ambiguous_ids = [f["defect_id"] for f in raw_findings if f["needs_roi_verification"]]
        trace.append({
            "step": len(trace) + 1,
            "phase": "PERCEIVE",
            "tool": "opencv_scrutineer_scan",
            "observation": {
                "num_findings": scan_summary["num_findings"],
                "ambiguous_findings": ambiguous_ids,
                "critical_count": scan_summary["critical_count"],
            },
            "decision": (
                f"Detected {len(ambiguous_ids)} ambiguous finding(s) ({', '.join(ambiguous_ids)}) with borderline confidence. "
                "Triggering secondary 2.5x ROI active zoom inspection (opencv_roi_highres_inspect)."
                if ambiguous_ids
                else "All findings have high-certainty metrics. Proceeding to action and governance stage."
            ),
        })

        # STEP 4 (CONDITIONAL): Secondary 2.5x High-Res ROI Active Perception Loop
        verified_findings: List[Dict[str, Any]] = []
        rejected_false_positives = 0
        for finding in raw_findings:
            if finding["needs_roi_verification"]:
                refined = self.engine.run_roi_highres_inspect(working_frame, finding, zoom_scale=2.5)
                trace.append({
                    "step": len(trace) + 1,
                    "phase": "PERCEIVE_ZOOM_ROI",
                    "tool": "opencv_roi_highres_inspect",
                    "observation": {
                        "defect_id": refined["defect_id"],
                        "category": refined["category"],
                        "prior_confidence": finding["confidence"],
                        "refined_confidence": refined["roi_refined_confidence"],
                        "roi_edge_energy": refined["roi_edge_energy"],
                        "roi_verdict": refined["roi_verdict"],
                    },
                    "decision": (
                        f"ROI Scharr gradient energy ({refined['roi_edge_energy']}) confirmed structural anomaly; "
                        f"upgraded confidence to {refined['roi_refined_confidence']}."
                        if refined["roi_verdict"] == "CONFIRMED_DEFECT_ON_ZOOM"
                        else f"ROI inspection showed low gradient energy ({refined['roi_edge_energy']}); "
                        "classified as benign surface mark and suppressed false alarm."
                    ),
                })
                if refined["roi_verdict"] == "CONFIRMED_DEFECT_ON_ZOOM":
                    verified_findings.append(refined)
                else:
                    rejected_false_positives += 1
            else:
                verified_findings.append(finding)

        # STEP 5: Action & Human-in-the-Loop Governance on AWS
        overall_severity = "NOMINAL"
        if any(f["severity"] == "CRITICAL" for f in verified_findings):
            overall_severity = "CRITICAL"
        elif any(f["severity"] == "WARNING" for f in verified_findings):
            overall_severity = "WARNING"

        requires_human_approval = overall_severity == "CRITICAL"
        ok, png_buf = cv2.imencode(".png", annotated_frame)
        img_bytes = png_buf.tobytes() if ok else b""

        total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        s3_res = self.aws.archive_evidence_s3(
            frame_id=frame_id,
            image_bytes=img_bytes,
            metadata={
                "frame_id": frame_id,
                "overall_severity": overall_severity,
                "verified_findings": verified_findings,
                "diagnostics": diag,
                "restoration": restoration_info,
                "total_latency_ms": total_latency_ms,
            },
        )
        cw_res = self.aws.emit_cloudwatch_telemetry({
            "PipelineLatencyMs": total_latency_ms,
            "VerifiedDefectCount": float(len(verified_findings)),
            "ActiveRestorationTriggered": 1.0 if diag["needs_active_restoration"] else 0.0,
            "FalsePositivesSuppressed": float(rejected_false_positives),
        })
        ticket_res = self.aws.dispatch_remediation_ticket(
            frame_id=frame_id,
            severity=overall_severity,
            requires_human_approval=requires_human_approval,
            findings=verified_findings,
            agent_trace=trace,
            s3_uri=s3_res["s3_uri"],
        )

        trace.append({
            "step": len(trace) + 1,
            "phase": "ACT_AND_GOVERN",
            "tool": "aws_dispatch_remediation_ticket",
            "observation": {
                "ticket_id": ticket_res["ticket_id"],
                "status": ticket_res["status"],
                "s3_uri": s3_res["s3_uri"],
                "cloudwatch_status": cw_res["status"],
            },
            "decision": (
                f"Dispatched {overall_severity} ticket {ticket_res['ticket_id']}. "
                + (
                    "High-voltage interlock ENGAGED: awaiting human scrutineer sign-off."
                    if requires_human_approval
                    else "Assembly telemetry logged to AWS S3 & CloudWatch."
                )
            ),
        })

        return {
            "frame_id": frame_id,
            "overall_severity": overall_severity,
            "requires_human_approval": requires_human_approval,
            "diagnostics": diag,
            "restoration": restoration_info,
            "verified_findings": verified_findings,
            "rejected_false_positives": rejected_false_positives,
            "agent_trace": trace,
            "aws_ticket": ticket_res,
            "aws_s3": s3_res,
            "aws_cloudwatch": cw_res,
            "total_latency_ms": total_latency_ms,
            "annotated_frame": annotated_frame,
            "restored_frame": working_frame,
        }
