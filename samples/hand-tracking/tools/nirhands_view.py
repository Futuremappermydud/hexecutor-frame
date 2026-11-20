"""Drawing for the NIR hand pipeline viewer: both slam views with skeletons, plus a top-down depth panel.

Keys (handled by ``Viewer.key``): q/Esc quit, space pause, d detector circles, r raw vs smoothed joints,
n step one frame while paused.
"""
from __future__ import annotations

import cv2
import numpy as np

from nirhands_pipeline import BONES

RIGHT_BGR = (255, 140, 30)   # right hands blue, left orange, as in the OpenXR viewer
LEFT_BGR = (20, 140, 255)


def head_axes(cameras) -> np.ndarray:
    """Rows: head right, up, forward, in the rig world frame (the slam pair is mounted rotated)."""
    centres = [-c.R_cw.T @ c.t_cw for c in cameras]
    right = centres[1] - centres[0]
    right /= np.linalg.norm(right)
    forward = cameras[0].R_cw[2] + cameras[1].R_cw[2]
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    forward = np.cross(up, right)
    return np.stack([right, up, forward])


def _safe_project(camera, points):
    try:
        pix = camera.project(points)
    except ValueError:
        return None
    return pix if np.isfinite(pix).all() else None


class Viewer:
    def __init__(self, cameras, title="Steam Frame NIR hands", scale=0.55, top_px=420):
        self.cameras = cameras
        self.title = title
        self.scale = scale
        self.top_px = top_px
        self.axes = head_axes(cameras)
        self.origin = np.mean([-c.R_cw.T @ c.t_cw for c in cameras], axis=0)
        self.show_detections = True
        self.show_raw = False
        self.paused = False
        self.step_once = False

    def render(self, images, result, stats, info: str = "") -> np.ndarray:
        panes = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in images]
        if self.show_detections and result.detections is not None:
            for pane, dets in zip(panes, result.detections):
                for d in dets:
                    c = RIGHT_BGR if d.is_right else LEFT_BGR
                    cv2.circle(pane, (int(d.u), int(d.v)), int(d.radius_px), c, 1, cv2.LINE_AA)
                    cv2.putText(pane, f"{d.score:.2f} {d.distance_m:.2f}m", (int(d.u - d.radius_px), int(d.v - d.radius_px) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
        for hand in result.hands:
            joints = hand.raw if self.show_raw else hand.joints
            color = RIGHT_BGR if hand.is_right else LEFT_BGR
            for pane, cam in zip(panes, self.cameras):
                pix = _safe_project(cam, joints)
                if pix is None:
                    continue
                h, w = pane.shape[:2]
                if not ((pix > -w) & (pix < 2 * w)).all():
                    continue
                for a, b in BONES:
                    cv2.line(pane, tuple(np.rint(pix[a]).astype(int)), tuple(np.rint(pix[b]).astype(int)), color, 3, cv2.LINE_AA)
                for j, p in enumerate(pix):
                    vis = hand.visibility[j] > 0.5
                    cv2.circle(pane, tuple(np.rint(p).astype(int)), 5 if vis else 3, color if vis else (90, 90, 90), -1, cv2.LINE_AA)
                anchor = tuple(np.rint(pix[0] + (0, 30)).astype(int))
                cv2.putText(pane, f"{'R' if hand.is_right else 'L'}{hand.id} {hand.branch}", anchor,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        view = np.hstack(panes)
        view = cv2.resize(view, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        view = np.hstack([view, self._top(result, view.shape[0])])
        self._hud(view, result, stats, info)
        return view

    def _top(self, result, height) -> np.ndarray:
        """Top-down panel: head right (x) against forward depth, 0-0.9 m, grid every 10 cm."""
        w = self.top_px
        panel = np.full((height, w, 3), 24, np.uint8)
        ppm = (height - 40) / 0.9

        def to_px(p):
            q = (np.asarray(p) - self.origin) @ self.axes.T
            return np.stack([w / 2 + q[..., 0] * ppm, height - 20 - q[..., 2] * ppm], -1)

        for d in np.arange(0.1, 0.91, 0.1):
            y = int(height - 20 - d * ppm)
            cv2.line(panel, (0, y), (w, y), (55, 55, 55), 1)
            cv2.putText(panel, f"{d:.1f}m", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.line(panel, (w // 2, 0), (w // 2, height), (55, 55, 55), 1)
        cv2.circle(panel, (w // 2, height - 20), 6, (200, 200, 200), -1)
        if result.seeds:
            for s in result.seeds:
                p = to_px(s.center_world)
                cv2.drawMarker(panel, tuple(np.rint(p).astype(int)), (160, 160, 160), cv2.MARKER_TILTED_CROSS, 12, 1)
        for hand in result.hands:
            joints = hand.raw if self.show_raw else hand.joints
            pix = to_px(joints)
            color = RIGHT_BGR if hand.is_right else LEFT_BGR
            for a, b in BONES:
                cv2.line(panel, tuple(np.rint(pix[a]).astype(int)), tuple(np.rint(pix[b]).astype(int)), color, 2, cv2.LINE_AA)
            depth = float((self.axes @ (hand.joints[9] - self.origin))[2])
            cv2.putText(panel, f"{depth:.2f}m", tuple(np.rint(pix[9] + (8, 0)).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(panel, "top view (head frame)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return panel

    def _hud(self, view, result, stats, info) -> None:
        t = result.timings_ms
        lines = [
            f"{stats.fps():4.1f} fps  hands {len(result.hands)}  "
            f"det {t.get('detect_ms', 0):4.1f} ms  kp {t.get('keypoints_ms', 0):4.1f} ms  total {t.get('total_ms', 0):4.1f} ms",
            f"{'RAW' if self.show_raw else 'smoothed'}  [d]etections {'on' if self.show_detections else 'off'}  "
            f"[space] {'paused' if self.paused else 'pause'}  [q]uit  {info}",
        ]
        for i, line in enumerate(lines):
            y = 26 + 26 * i
            cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    def show(self, frame) -> bool:
        """Display a rendered frame; returns False when the user quits. Blocks while paused."""
        cv2.imshow(self.title, frame)
        while True:
            key = cv2.waitKey(30 if self.paused else 1) & 0xFF
            if key in (27, ord("q")):
                return False
            if key == ord(" "):
                self.paused = not self.paused
            elif key == ord("d"):
                self.show_detections = not self.show_detections
            elif key == ord("r"):
                self.show_raw = not self.show_raw
            elif key == ord("n") and self.paused:
                return True
            if not self.paused:
                return True
