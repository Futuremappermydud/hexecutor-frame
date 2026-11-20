"""Live viewer for the end-to-end stereo NIR hand pipeline on the Frame's slam camera pair.

The device binary stream_nir_pairs (same directory) streams the read-only XRService tracking pair over SSH; it
must run with permission to duplicate XRService's root-owned FDs. Inference runs on this host: our detector,
stereo seeding, the keypoint model, tracking and smoothing (nirhands_pipeline.py). ``--input`` replays a stream
recorded with ``--record`` (or the device's raw dump) through the same receive path. No device model is needed.
"""
from __future__ import annotations

import argparse
import getpass
import queue
import struct
import subprocess
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from nirhands_replay import NirHandsBackend, _draw

HEADER = struct.Struct("<8sqqII")
SHAPE = (1024, 1056)
PIXELS = SHAPE[0] * SHAPE[1]


def read_exact(stream, size, allow_eof=False):
    data = bytearray(size)
    view = memoryview(data)
    while view:
        count = stream.readinto(view)
        if not count:
            if allow_eof and len(view) == size:
                return None
            raise EOFError("Frame stream ended")
        view = view[count:]
    return bytes(data)


def pairs(stream, truncated_ok=False):
    """Decoded pairs from a stream. ``truncated_ok``: a recording cut mid-pair (process stopped) just ends."""
    while True:
        try:
            pair = _read_pair(stream)
        except EOFError:
            if truncated_ok:
                return
            raise
        if pair is None:
            return
        yield pair


def _read_pair(stream):
    """One pair, or None at a clean end of stream."""
    header = read_exact(stream, HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, left_ns, right_ns, left_seq, right_seq = HEADER.unpack(header)
    if magic not in (b"FRNIR001", b"FRNIR002") or abs(left_ns - right_ns) > 1_000_000:
        raise ValueError("invalid or unsynchronized Frame pair")
    if magic == b"FRNIR002":
        lengths = struct.unpack("<II", read_exact(stream, 8))
        if any(n < 1 or n > PIXELS * 2 for n in lengths):
            raise ValueError("invalid JPEG length")
        left, right = [cv2.imdecode(np.frombuffer(read_exact(stream,n),np.uint8),cv2.IMREAD_GRAYSCALE)
                       for n in lengths]
        if any(image is None or image.shape != SHAPE for image in (left,right)):
            raise ValueError("invalid JPEG image")
    else:
        left = np.frombuffer(read_exact(stream, PIXELS), np.uint8).reshape(SHAPE)
        right = np.frombuffer(read_exact(stream, PIXELS), np.uint8).reshape(SHAPE)
    return np.stack((left, right)), (left_ns, right_ns), (left_seq, right_seq)


def newest_pairs(stream):
    """Drain transport independently of inference; retain at most one decoded pair."""
    pending = queue.Queue(maxsize=1)
    finished = threading.Event()
    errors = []
    def receive():
        try:
            for pair in pairs(stream):
                try:
                    pending.get_nowait()
                except queue.Empty:
                    pass
                pending.put((pair, time.monotonic()))
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()
    threading.Thread(target=receive, daemon=True).start()
    while not finished.is_set() or not pending.empty():
        try:
            yield pending.get(timeout=.1)
        except queue.Empty:
            continue
    if errors:
        raise errors[0]


class HandSeeder:
    """Frame pair in, per-view (pixel centre, right, score) detections out."""
    def __init__(self, name, cameras, model_dir):
        from nirhands.frametop.teacher import PalmDetector, MediaPipeLandmarks, MercuryTeacher
        from nirhands.training.real_eval import _headset_up
        self.name = name
        self.cameras = cameras
        self.up = _headset_up(*cameras)
        if name == "mercury":
            self.detector = MercuryTeacher(model_dir, threads=2)
        elif name == "mediapipe":
            self.palm = PalmDetector(model_dir, threads=2)
            self.mp = MediaPipeLandmarks(model_dir, threads=2, accelerate=False)
        else:
            raise ValueError(f"unknown seed backend: {name}")

    def detect(self, images):
        found = [[], []]
        for view, (image, camera) in enumerate(zip(images, self.cameras)):
            if self.name == "mercury":
                # MercuryTeacher.detect uses the original upright fisheye crop,
                # 160x160 resize and mean/std normalization from mercury_train.
                for side, score, center, size in self.detector.detect(image, camera, self.up):
                    # The detector's side convention is reversed relative to
                    # the Frametop label convention on these raw Frame views.
                    found[view].append((center, not bool(side), score))
            else:
                for det in self.palm.detect(image, camera, self.up):
                    kp, presence, handed, _ = self.mp.track(image, self.palm.seed_keypoints(det), iterations=2)
                    if presence >= .5 and np.isfinite(kp).all():
                        found[view].append((kp[9], handed >= .5, presence))
        return found


class SeedTracker:
    def __init__(self, backend, cameras, model_dir, seed="mercury", depth_guess=0.45, reseed_every=10):
        self.backend = backend
        self.cameras = cameras
        self.seeder = HandSeeder(seed, cameras, model_dir)
        self.depth_guess = depth_guess
        self.reseed_every = reseed_every
        self.frame = 0
        self.tracks = []

    def _seeds(self, images):
        return self.seeder.detect(images)

    def _mono_center(self, view, pixel, depth):
        from nirhands.frametop.session import unproject_kb
        cam = self.cameras[view]
        ray = unproject_kb(cam, np.asarray(pixel))[()] * depth
        return cam.R_cw.T @ (ray - cam.t_cw)

    def step(self, images):
        from nirhands.frametop.labeller import triangulate
        reseeding = not self.tracks or self.frame % self.reseed_every == 0
        seeds = self._seeds(images) if reseeding else [[], []]
        self.frame += 1
        candidates = []
        for track in self.tracks:
            center, right, missed = track
            matched = any(d[1] == right and np.linalg.norm(d[0] - cam.project(center)) < 100
                          for cam, detections in zip(self.cameras, seeds) for d in detections)
            missed = 0 if matched else missed + int(reseeding)
            if missed >= 2:
                continue
            seen = tuple(0 <= cam.project(center)[0] < SHAPE[1] and
                         0 <= cam.project(center)[1] < SHAPE[0] for cam in self.cameras)
            if any(seen):
                candidates.append((center, right, seen, missed))
        used = [set(), set()]
        for ia, a in enumerate(seeds[0]):
            for ib, b in enumerate(seeds[1]):
                if ia in used[0] or ib in used[1]:
                    continue
                center, error = triangulate(list(self.cameras), np.array([[a[0]], [b[0]]]))
                center = center[0]
                if max(error.ravel()) < 12 and 0.1 < self.cameras[0].world_to_camera(center)[2] < 1.5:
                    if any(np.linalg.norm(center - t[0]) < .12 for t in self.tracks):
                        continue
                    candidates.append((center, a[1], (True, True), 0))
                    used[0].add(ia); used[1].add(ib)
        for view in (0, 1):
            for index, (pixel, right, _) in enumerate(seeds[view]):
                if index not in used[view]:
                    mono = self._mono_center(view, pixel, self.depth_guess)
                    if not any(np.linalg.norm(mono - t[0]) < .12 for t in self.tracks):
                        candidates.append((mono, right, (view == 0, view == 1), 0))
        hands, tracks = [], []
        for center, right, seen, missed in candidates[:2]:
            result = self.backend.predict(images, self.cameras, center, right, seen)
            if result is None:
                continue
            joints, visibility, branch = result
            if float(np.mean(visibility)) < .2:
                continue
            hands.append((joints, visibility, right))
            tracks.append((joints[9], right, missed))
        self.tracks = tracks
        return hands


class TeeReader:
    """Binary reader that copies everything it reads to ``sink`` (records the live stream for replay)."""

    def __init__(self, stream, sink):
        self.stream, self.sink = stream, sink

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        if count:
            self.sink.write(memoryview(buffer)[:count])
        return count


def open_input(path: Path):
    if path.suffix == ".zst":
        try:
            from compression import zstd
        except ImportError:
            import zstandard
            return zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
        return zstd.open(path, "rb")
    return path.open("rb")


def load_cameras(args):
    if args.session_calibration:
        from nirhands.frametop.session import Session
        session = Session(args.session_calibration)
        return session.cameras["slam_left"], session.cameras["slam_right"]
    from nirhands.camera.calibration import RigCalibration
    rig = RigCalibration.load(args.rig)
    return rig.left, rig.right


def main():
    from nirhands_take import build_pipeline, pipeline_args
    p = argparse.ArgumentParser(description=__doc__)
    pipeline_args(p)
    p.add_argument("--seed", choices=("nirhands", "mercury", "mediapipe"), default="nirhands",
                   help="nirhands: our detector + tracker (default); mercury/mediapipe: earlier seeders")
    p.add_argument("--model-dir", type=Path, help="Mercury and MediaPipe ONNX assets (--seed mercury/mediapipe)")
    calib = p.add_mutually_exclusive_group(required=True)
    calib.add_argument("--rig", type=Path, help="generic rig YAML, e.g. <project>/config/steam_frame_tracking.yaml")
    calib.add_argument("--session-calibration", type=Path,
                       help="Frametop-format session directory whose calibration.json matches this headset")
    p.add_argument("--host", default="frame")
    p.add_argument("--sudo", action="store_true", help="prompt for Frame sudo password and run transiently")
    p.add_argument("--remote-binary", default="/tmp/nirhands-live/stream_nir_pairs")
    p.add_argument("--input", type=Path, help="replay a recorded stream (.raw or .raw.zst) instead of SSH")
    p.add_argument("--realtime", action="store_true", help="with --input: pace replay at the capture rate")
    p.add_argument("--record", type=Path, help="also save the received stream to this file for later --input")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--raw", action="store_true", help="use original lossless raw protocol")
    p.add_argument("--no-display", action="store_true", help="process pairs without opening a window")
    p.add_argument("--json", type=Path, help="write pipeline statistics here on exit")
    args = p.parse_args()
    from nirhands_pipeline import add_project
    add_project(args.project)
    cameras = load_cameras(args)
    if args.seed == "nirhands":
        pipe = build_pipeline(args, cameras)
        legacy = None
    else:
        if args.model_dir is None:
            p.error("--model-dir is required for --seed mercury/mediapipe")
        checkpoint = args.checkpoint or args.project.resolve().parent / "data/runs/cotrain_v5_labels_fah/best.pt"
        legacy = SeedTracker(NirHandsBackend(checkpoint, args.project, args.device or "cpu"), cameras,
                             args.model_dir, seed=args.seed)
        pipe = None

    process = None
    if args.input:
        stream = open_input(args.input)
    else:
        command = ["ssh", args.host]
        if args.sudo:
            command += ["sudo", "-S", "--prompt="]
        command.append(args.remote_binary)
        command += ["--jpeg-quality", str(args.jpeg_quality)]
        if args.raw:
            command.append("--raw")
        if args.max_pairs:
            command += ["--max-pairs", str(args.max_pairs)]
        password = getpass.getpass("Frame sudo password: ") if args.sudo else None
        process = subprocess.Popen(command, stdin=subprocess.PIPE if args.sudo else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE)
        if args.sudo:
            process.stdin.write((password + "\n").encode())
            process.stdin.flush()
            process.stdin.close()
            del password
        stream = process.stdout
    record = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record = args.record.open("wb")
        stream = TeeReader(stream, record)

    viewer = None
    if not args.no_display:
        from nirhands_view import Viewer
        viewer = Viewer(cameras)
    # A replayed file is read in order (every pair); a live stream keeps only the newest pair.
    source = (((pair, time.monotonic()) for pair in pairs(stream, truncated_ok=True)) if args.input else newest_pairs(stream))
    try:
        first_received = wall0 = t_first = None
        last_seq = None
        skipped = 0
        for index, ((images, times, sequences), received) in enumerate(source):
            if first_received is None:
                first_received = received
            t = times[0] / 1e9
            if args.input and args.realtime:
                if wall0 is None:
                    wall0, t_first = time.monotonic(), t
                wait = (t - t_first) - (time.monotonic() - wall0)
                if wait > 0:
                    time.sleep(wait)
            if last_seq is not None:
                skipped += max(0, (sequences[0] - last_seq) // 2 - 1)  # slam cameras step by 2 per pair
            last_seq = sequences[0]
            start = time.monotonic()
            if pipe is not None:
                result = pipe.step(images, t)
                hands = result.hands
            else:
                hands = legacy.step(images)
            processing_ms = (time.monotonic() - start) * 1000
            rate = index / max(received - first_received, .001)
            if index % 30 == 0 or args.no_display:
                print(f"pair={index} seq={sequences} hands={len(hands)} processing_ms={processing_ms:.1f} "
                      f"delivered_fps={rate:.2f} skipped={skipped} "
                      f"receive_to_result_ms={(time.monotonic()-received)*1000:.1f}", flush=True)
            if viewer is not None:
                if pipe is not None:
                    frame = viewer.render(images, result, pipe.stats, f"seq {sequences[0]} skipped {skipped}")
                    keep = viewer.show(frame)
                else:
                    cv2.imshow("Steam Frame NIR hands: left | right", _draw(images, cameras, hands))
                    keep = cv2.waitKey(1) not in (27, ord("q"))
                if not keep:
                    break
            if args.max_pairs and index + 1 >= args.max_pairs:
                break
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=3)
        if record is not None:
            record.close()
            print(f"recorded stream: {args.record}")
        cv2.destroyAllWindows()
        if pipe is not None:
            import json
            summary = pipe.stats.summary()
            print(json.dumps(summary, indent=2))
            if args.json:
                args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
