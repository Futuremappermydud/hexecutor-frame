# Rendering and occlusion

## Diagnostic OpenVR monitor

The diagnostic application shows left color, depth, and right color. It is a head-locked monitor for inspecting acquisition and depth, not a substitute for a stereo world-space scene. Its monitor mode keeps side-by-side stereo disabled so both eyes see the same card.

Large or frequent `SetOverlayRaw` uploads have failed while an older image remained visible. A moving application log therefore does not prove the headset texture is updating. The small raw-upload mode and Vulkan texture path have different limits. Check presentation freshness when changing image size or upload rate.

## OpenXR cloud and mesh

The OpenXR application renders world-space geometry through per-eye view/projection matrices and depth testing. The background worker publishes geometry; the render thread continues pose location and frame submission independently of inference. The code must explicitly transform the camera convention (x right, y down, z forward) into the presentation convention.

Keep capture-time pose separate from display-time head pose. Capture pose places geometry in the world; the current display pose determines how each eye views that geometry. Using display-time pose to integrate an old camera frame causes apparent sliding during motion.

## Optional API-layer occlusion

The Vulkan OpenXR layer intercepts the supported rendering/submission path and uses a depth surface to hide virtual fragments behind observed real geometry. It is opt-in and constrained by application compatibility. Applications must already enable passthrough and submit the required stereo depth information; the layer does not manufacture missing runtime support.

Occlusion preparation may complete bounded holes and form triangles from compatible neighboring samples. It must stop triangles across depth discontinuities and handle unknown/stale depth explicitly. A dense mask that leaks over a foreground boundary can hide virtual objects incorrectly even when average depth error is small.

Keep the worker's published surface, camera pose, frame freshness, reprojection, and submission ownership consistent. Layer staging, completion, shader execution, and application submission are distinct sources of latency. The scene viewer's appearance alone does not validate the interception path.

## Compatibility and diagnostics

The default, guarded, and learned depth paths have different validation boundaries. In the current cloud worker, learned depth is not selected for the occlusion-only worker. Do not assume a viewer flag promotes that model into the API layer.

Investigate visual faults in this order: stale publication, incorrect pose/frame transform, depth convention, discontinuity handling, then GPU submission. Retain render-loop cadence and GPU contention measurements when changing the occlusion stage; the compositor shares the GPU with depth compute.

Source: [OpenXR viewer](../../runtime/src/openxr_point_cloud.cpp), [overlay](../../runtime/src/overlay_renderer.cpp), [API layer](../../runtime/src/openxr_occlusion_layer.cpp), and [occlusion renderer](../../runtime/src/occlusion_renderer.cpp).
