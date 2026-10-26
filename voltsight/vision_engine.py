"""
voltsight/vision_engine.py
Core OpenCV 5 / 4.x Computer Vision Engine for EV Battery Pack & Industrial Assembly Scrutineering.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
import cv2


@dataclass
class FrameDiagnostics:
    width: int
    height: int
    laplacian_variance: float
    mean_luminance: float
    dynamic_range_p95_p5: float
    glare_pixel_ratio: float
    is_blurry: bool
    has_severe_glare: bool
    is_low_contrast: bool
    needs_active_restoration: bool
    recommended_restoration_mode: str
    diagnostics_latency_ms: float


@dataclass
class DefectFinding:
    defect_id: str
    category: str
    severity: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    measured_value: float
    unit: str
    safety_threshold: float
    needs_roi_verification: bool
    verified_by_roi: bool = False
    roi_refined_confidence: Optional[float] = None
    notes: str = ""


class OpenCVVisionEngine:
    """Deterministic OpenCV pipeline designed for Agentic Active-Perception loops."""

    def __init__(self, mm_per_pixel_default: float = 0.25, hv_min_clearance_mm: float = 6.0):
        self.opencv_version = cv2.__version__
        self.mm_per_pixel_default = mm_per_pixel_default
        self.hv_min_clearance_mm = hv_min_clearance_mm

    def run_frame_diagnostics(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Stage 1: Evaluates visual quality metrics to inform the agent whether restoration is needed."""
        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        mean_lum = float(np.mean(l_chan))
        p95, p5 = np.percentile(l_chan, [95, 5])
        dyn_range = float(p95 - p5)

        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        glare_mask = cv2.inRange(hsv, np.array([0, 0, 238]), np.array([180, 45, 255]))
        glare_ratio = float(np.count_nonzero(glare_mask)) / float(h * w)

        is_blurry = lap_var < 120.0
        has_glare = glare_ratio > 0.015
        is_low_contrast = dyn_range < 115.0 or mean_lum < 68.0
        needs_restoration = is_blurry or has_glare or is_low_contrast

        modes = []
        if has_glare:
            modes.append("inpaint_deglare")
        if is_low_contrast or has_glare or is_blurry:
            modes.append("clahe_lab")
        if is_blurry:
            modes.append("unsharp_deblur")
        mode_str = "+".join(modes) if modes else "none"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        diag = FrameDiagnostics(
            width=w,
            height=h,
            laplacian_variance=round(lap_var, 2),
            mean_luminance=round(mean_lum, 2),
            dynamic_range_p95_p5=round(dyn_range, 2),
            glare_pixel_ratio=round(glare_ratio, 4),
            is_blurry=is_blurry,
            has_severe_glare=has_glare,
            is_low_contrast=is_low_contrast,
            needs_active_restoration=needs_restoration,
            recommended_restoration_mode=mode_str,
            diagnostics_latency_ms=round(dt_ms, 2),
        )
        return asdict(diag)

    def run_active_restoration(self, bgr_frame: np.ndarray, mode: str = "auto") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Stage 2: Executes OpenCV restoration filters dynamically selected by the agent."""
        t0 = time.perf_counter()
        restored = bgr_frame.copy()
        applied_steps = []

        if mode == "auto":
            diag = self.run_frame_diagnostics(bgr_frame)
            mode = diag["recommended_restoration_mode"]

        if "inpaint_deglare" in mode:
            hsv = cv2.cvtColor(restored, cv2.COLOR_BGR2HSV)
            glare_mask = cv2.inRange(hsv, np.array([0, 0, 238]), np.array([180, 48, 255]))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
            restored = cv2.inpaint(restored, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
            # Boost saturation in washed-out chromatic zones
            hsv_r = cv2.cvtColor(restored, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_r[:, :, 1] = np.clip(hsv_r[:, :, 1] * 1.65, 0, 255)
            restored = cv2.cvtColor(hsv_r.astype(np.uint8), cv2.COLOR_HSV2BGR)
            applied_steps.append("cv2.inpaint(INPAINT_NS)+HSV_ChromaRestore")

        if "clahe_lab" in mode or mode == "all":
            lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_chan)
            restored = cv2.cvtColor(cv2.merge((l_eq, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
            applied_steps.append("cv2.createCLAHE(LAB,clip=3.2)")

        if "unsharp_deblur" in mode or mode == "all":
            gaussian = cv2.GaussianBlur(restored, (0, 0), sigmaX=2.4)
            restored = cv2.addWeighted(restored, 2.25, gaussian, -1.25, 0)
            restored = cv2.bilateralFilter(restored, d=5, sigmaColor=30, sigmaSpace=30)
            applied_steps.append("cv2.addWeighted(Unsharp,2.25x)+cv2.bilateralFilter")

        post_diag = self.run_frame_diagnostics(restored)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return restored, {
            "applied_steps": applied_steps,
            "post_restoration_laplacian": post_diag["laplacian_variance"],
            "post_restoration_glare_ratio": post_diag["glare_pixel_ratio"],
            "restoration_latency_ms": round(dt_ms, 2),
        }

    def run_scrutineering_scan(self, bgr_frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        """Stage 3: Multi-modal OpenCV inspection for thermal hotspots, busbar cracks, and HV clearance."""
        t0 = time.perf_counter()
        annotated = bgr_frame.copy()
        h, w = bgr_frame.shape[:2]
        mm_per_px = self.mm_per_pixel_default
        findings: List[DefectFinding] = []

        # --- Sub-pipeline A: HSV Thermal Anomaly / Cell Hotspot Segmentation ---
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask_hot1 = cv2.inRange(hsv, np.array([0, 135, 165]), np.array([13, 255, 255]))
        mask_hot2 = cv2.inRange(hsv, np.array([167, 135, 165]), np.array([180, 255, 255]))
        thermal_mask = cv2.bitwise_or(mask_hot1, mask_hot2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thermal_mask = cv2.morphologyEx(thermal_mask, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thermal_mask)
        for idx in range(1, num_labels):
            area_px = int(stats[idx, cv2.CC_STAT_AREA])
            if area_px < 110:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            aspect = float(bw) / float(max(bh, 1))
            if aspect > 2.2 or aspect < 0.45:
                continue
            area_mm2 = round(area_px * (mm_per_px ** 2), 2)
            roi_v = hsv[y:y+bh, x:x+bw, 2]
            est_temp_c = round(48.0 + (float(np.mean(roi_v)) - 165.0) * 0.42, 1)
            conf = round(min(0.96, 0.62 + (area_px / 950.0)), 2)
            needs_roi = 0.55 <= conf < 0.85
            sev = "CRITICAL" if est_temp_c >= 68.0 else "WARNING"
            findings.append(DefectFinding(
                defect_id=f"THM-{len(findings)+1:02d}",
                category="THERMAL_HOTSPOT",
                severity=sev,
                confidence=conf,
                bbox=(x, y, bw, bh),
                measured_value=est_temp_c,
                unit="degC",
                safety_threshold=65.0,
                needs_roi_verification=needs_roi,
                notes=f"Cell hotspot area {area_mm2} mm2 via cv2.connectedComponentsWithStats",
            ))
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(annotated, f"HOTSPOT {est_temp_c}C ({conf:.2f})", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)

        # --- Sub-pipeline B: Busbar Micro-Crack Detection (Black-Hat + Gradient Verification) ---
        y1_b, y2_b = int(h * 0.28), int(h * 0.45)
        x1_b, x2_b = int(w * 0.14), int(w * 0.86)
        busbar_roi = bgr_frame[y1_b:y2_b, x1_b:x2_b]
        gray_bus = cv2.cvtColor(busbar_roi, cv2.COLOR_BGR2GRAY)
        bh_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray_bus, cv2.MORPH_BLACKHAT, bh_kern)
        _, crack_bin = cv2.threshold(blackhat, 52, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(crack_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            arc = cv2.arcLength(cnt, False)
            if area < 18 or arc < 28:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rh < 16 or rw > 22:
                continue
            mask_c = np.zeros_like(gray_bus)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)
            mean_bh_val = float(cv2.mean(blackhat, mask=mask_c)[0])
            crack_len_mm = round(rh * mm_per_px, 2)
            gx, gy = x1_b + rx, y1_b + ry
            # Confidence scales with black-hat fissure depth; shallow smudges land in [0.64, 0.81] -> triggers ROI zoom
            conf = round(min(0.95, max(0.58, 0.40 + (mean_bh_val / 155.0))), 2)
            needs_roi = 0.55 <= conf < 0.85
            sev = "CRITICAL" if crack_len_mm >= 3.5 else "WARNING"
            findings.append(DefectFinding(
                defect_id=f"CRK-{len(findings)+1:02d}",
                category="BUSBAR_MICRO_CRACK",
                severity=sev,
                confidence=conf,
                bbox=(gx, gy, rw, rh),
                measured_value=crack_len_mm,
                unit="mm",
                safety_threshold=3.5,
                needs_roi_verification=needs_roi,
                notes=f"Busbar fissure depth={mean_bh_val:.1f} via cv2.MORPH_BLACKHAT+findContours",
            ))
            color = (0, 165, 255) if not needs_roi else (0, 215, 255)
            cv2.rectangle(annotated, (gx, gy), (gx + rw, gy + rh), color, 2)
            cv2.putText(annotated, f"CRACK {crack_len_mm}mm ({conf:.2f})", (gx, max(18, gy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

        # --- Sub-pipeline C: High-Voltage (HV) Cable-to-Chassis Clearance via cv2.distanceTransform ---
        hv_mask = cv2.inRange(hsv, np.array([15, 140, 140]), np.array([26, 255, 255]))
        chassis_mask = np.zeros((h, w), dtype=np.uint8)
        chassis_mask[int(h * 0.80):int(h * 0.90), int(w * 0.12):int(w * 0.88)] = 255

        if np.count_nonzero(hv_mask) > 80:
            inv_chassis = cv2.bitwise_not(chassis_mask)
            dist_map = cv2.distanceTransform(inv_chassis, cv2.DIST_L2, 5)
            hv_points = np.where(hv_mask > 0)
            dists_px = dist_map[hv_points]
            min_idx = int(np.argmin(dists_px))
            min_dist_px = float(dists_px[min_idx])
            min_clearance_mm = round(min_dist_px * mm_per_px, 2)
            py, px = int(hv_points[0][min_idx]), int(hv_points[1][min_idx])

            if min_clearance_mm < self.hv_min_clearance_mm:
                conf = round(0.91 if min_clearance_mm < 4.5 else 0.76, 2)
                needs_roi = 4.5 <= min_clearance_mm < self.hv_min_clearance_mm
                sev = "CRITICAL" if min_clearance_mm < 4.5 else "WARNING"
                bbox = (max(0, px - 28), max(0, py - 18), 56, int(min_dist_px) + 32)
                findings.append(DefectFinding(
                    defect_id=f"HVC-{len(findings)+1:02d}",
                    category="HV_CLEARANCE_VIOLATION",
                    severity=sev,
                    confidence=conf,
                    bbox=bbox,
                    measured_value=min_clearance_mm,
                    unit="mm",
                    safety_threshold=self.hv_min_clearance_mm,
                    needs_roi_verification=needs_roi,
                    notes="Minimum Euclidean distance via cv2.distanceTransform(DIST_L2)",
                ))
                cv2.line(annotated, (px, py), (px, int(h * 0.80)), (255, 0, 255), 2)
                cv2.circle(annotated, (px, py), 5, (255, 0, 255), -1)
                cv2.putText(annotated, f"HV CLEARANCE {min_clearance_mm}mm < 6.0mm",
                            (max(10, px - 60), max(20, py - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        summary = {
            "num_findings": len(findings),
            "ambiguous_findings_count": sum(1 for f in findings if f.needs_roi_verification),
            "critical_count": sum(1 for f in findings if f.severity == "CRITICAL"),
            "scan_latency_ms": round(dt_ms, 2),
        }
        return [asdict(f) for f in findings], annotated, summary

    def run_roi_highres_inspect(self, bgr_frame: np.ndarray, finding: Dict[str, Any], zoom_scale: float = 2.5) -> Dict[str, Any]:
        """Stage 4: Active Perception secondary zoom pass on ambiguous regions."""
        t0 = time.perf_counter()
        h, w = bgr_frame.shape[:2]
        x, y, bw, bh = finding["bbox"]
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = bgr_frame[y0:y1, x0:x1]

        zoom_w = max(32, int(crop.shape[1] * zoom_scale))
        zoom_h = max(32, int(crop.shape[0] * zoom_scale))
        upscaled = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_LANCZOS4)
        filtered = cv2.bilateralFilter(upscaled, d=7, sigmaColor=45, sigmaSpace=45)
        gray_z = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        gx = cv2.Scharr(gray_z, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray_z, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(gx, gy)
        edge_energy = float(np.percentile(grad_mag, 95))

        prior_conf = float(finding["confidence"])
        if edge_energy > 920.0:
            refined_conf = round(min(0.98, prior_conf + 0.16), 2)
            is_confirmed = True
            verdict = "CONFIRMED_DEFECT_ON_ZOOM"
        else:
            refined_conf = round(max(0.22, prior_conf - 0.38), 2)
            is_confirmed = False
            verdict = "REJECTED_FALSE_POSITIVE_SURFACE_MARK"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        updated = dict(finding)
        updated["verified_by_roi"] = True
        updated["roi_refined_confidence"] = refined_conf
        updated["confidence"] = refined_conf
        updated["roi_edge_energy"] = round(edge_energy, 2)
        updated["roi_verdict"] = verdict
        updated["roi_latency_ms"] = round(dt_ms, 2)
        if not is_confirmed:
            updated["severity"] = "NOMINAL"
        return updated
