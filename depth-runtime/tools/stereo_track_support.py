"""Offline monotone training-support pruning; no calibration residual selection.

Keep a connected frame/track core satisfying the original support and left-ray
parallax requirements. Whole tracks and frames are removed, never coordinates
modified. Held-out observations and tags are deliberately not inputs.
"""
import numpy as np
from stereo_direct_tracks import filter_left_parallax


def prune_training_support(rows, rotations, intrinsic, minimum_frame_tracks=12,
                           minimum_track_frames=3, minimum_frames=3,
                           minimum_tracks=30, minimum_observations=90,
                           minimum_parallax_degrees=.5):
    settings = (minimum_frame_tracks, minimum_track_frames, minimum_frames, minimum_tracks, minimum_observations)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in settings) or minimum_track_frames < 3:
        raise ValueError('positive integer support settings, at least three track frames required')
    current, _ = filter_left_parallax(rows, rotations, intrinsic, minimum_parallax_degrees, minimum_track_frames)
    original = np.asarray(rows, float)
    original_frames = sorted(set(map(int, original[:, 0])))
    original_tracks = sorted(set(map(int, original[:, 1])))
    rounds = []
    while len(current):
        before = len(current)
        # Re-evaluate parallax after every frame deletion, retaining raw frame IDs.
        current, _ = filter_left_parallax(current, rotations, intrinsic,
                                          minimum_parallax_degrees, minimum_track_frames)
        frame_ids, counts = np.unique(current[:, 0], return_counts=True)
        removed_frames = frame_ids[counts < minimum_frame_tracks]
        current = current[~np.isin(current[:, 0], removed_frames)]
        # A bridge can disappear. Use the largest connected component by training
        # observation count, then frame count, then earliest original frame ID.
        graph = {int(f): set() for f in np.unique(current[:, 0])}
        for track in np.unique(current[:, 1]):
            frames = current[current[:, 1] == track, 0].astype(int)
            for a, b in zip(frames, frames[1:]):
                graph[a].add(b); graph[b].add(a)
        components, unseen = [], set(graph)
        while unseen:
            seed = min(unseen); stack, component = [seed], set()
            while stack:
                frame = stack.pop()
                if frame not in component:
                    component.add(frame); stack.extend(graph[frame]-component)
            unseen -= component; components.append(component)
        if components:
            chosen = max(components, key=lambda c: (int(np.isin(current[:, 0], list(c)).sum()), len(c), -min(c)))
            current = current[np.isin(current[:, 0], list(chosen))]
        rounds.append({'input_observations': before, 'output_observations': len(current),
                       'removed_under_supported_frames': removed_frames.astype(int).tolist(),
                       'connected_components_before_selection': len(components)})
        if len(current) == before:
            break
    frames = sorted(set(map(int, current[:, 0])))
    tracks = sorted(set(map(int, current[:, 1])))
    reasons = []
    if len(frames) < minimum_frames: reasons.append('insufficient_frames')
    if len(tracks) < minimum_tracks: reasons.append('insufficient_tracks')
    if len(current) < minimum_observations: reasons.append('insufficient_observations')
    return current, frames, {'support_pruning': {
        'eligible': not reasons, 'reasons': reasons,
        'input_observations': len(original), 'retained_observations': len(current),
        'retained_original_frame_indices': frames,
        'removed_original_frame_indices': sorted(set(original_frames)-set(frames)),
        'retained_track_count': len(tracks), 'removed_track_ids': sorted(set(original_tracks)-set(tracks)),
        'minimum_frame_tracks': minimum_frame_tracks, 'minimum_track_frames': minimum_track_frames,
        'minimum_frames': minimum_frames, 'minimum_tracks': minimum_tracks,
        'minimum_observations': minimum_observations, 'minimum_parallax_degrees': minimum_parallax_degrees,
        'rounds': rounds, 'scope': 'Training observations and independent rotations only; unchanged support minima, no fitting residuals or held-out selection.'}}
