import struct
import math

def load_ppm(path):
    with open(path, "rb") as f:
        f.readline(); f.readline(); f.readline()
        raw = f.read()
    W, H = 960, 1080
    return raw, W, H

rawL, W, H = load_ppm("/home/steamos/test_openvr/left.ppm")
rawR, W, H = load_ppm("/home/steamos/test_openvr/right.ppm")

# Downsample to 240x270 for fast feature extraction
sW, sH = 240, 270
scale = 4
def to_grey(raw):
    g = bytearray(sW * sH)
    for y in range(sH):
        for x in range(sW):
            idx = ((y * scale) * W + (x * scale)) * 3
            g[y * sW + x] = int(0.299 * raw[idx] + 0.587 * raw[idx+1] + 0.114 * raw[idx+2])
    return g

gL = to_grey(rawL)
gR = to_grey(rawR)

corners = []
for y in range(10, sH - 10, 4):
    for x in range(10, sW - 10, 4):
        gx = (gL[y * sW + (x+1)] - gL[y * sW + (x-1)])
        gy = (gL[(y+1) * sW + x] - gL[(y-1) * sW + x])
        grad_mag = gx*gx + gy*gy
        if grad_mag > 800:
            corners.append((x, y, grad_mag))

corners.sort(key=lambda c: c[2], reverse=True)
selected = []
for c in corners:
    if all((c[0]-s[0])**2 + (c[1]-s[1])**2 > 100 for s in selected):
        selected.append(c)
        if len(selected) >= 50:
            break

print(f"Found {len(selected)} strong corners in Left image.")

matches = []
patch_r = 5
for x, y, _ in selected:
    best_ncc = -1.0
    best_dx, best_dy = 0, 0
    pL = []
    for dy in range(-patch_r, patch_r+1):
        for dx in range(-patch_r, patch_r+1):
            pL.append(gL[(y+dy)*sW + (x+dx)])
    meanL = sum(pL) / len(pL)
    stdL = math.sqrt(sum((v - meanL)**2 for v in pL) + 1e-4)

    for dy in range(5, 45):
        for dx in range(30, 110):
            rx = x + dx
            ry = y + dy
            if patch_r <= rx < sW - patch_r and patch_r <= ry < sH - patch_r:
                pR = []
                for pdy in range(-patch_r, patch_r+1):
                    for pdx in range(-patch_r, patch_r+1):
                        pR.append(gR[(ry+pdy)*sW + (rx+pdx)])
                meanR = sum(pR) / len(pR)
                stdR = math.sqrt(sum((v - meanR)**2 for v in pR) + 1e-4)
                cov = sum((pL[i]-meanL)*(pR[i]-meanR) for i in range(len(pL)))
                ncc = cov / (stdL * stdR)
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_dx = dx
                    best_dy = dy

    if best_ncc > 0.85:
        matches.append((x * scale, y * scale, (x + best_dx) * scale, (y + best_dy) * scale, best_ncc))

print(f"High-confidence stereo matches found: {len(matches)}")
for idx, (lx, ly, rx, ry, ncc) in enumerate(matches[:10]):
    print(f"  Match {idx}: L=({lx}, {ly}) -> R=({rx}, {ry}) | dx={rx-lx}, dy={ry-ly} (ncc={ncc:.3f})")

dxs = [m[2] - m[0] for m in matches]
dys = [m[3] - m[1] for m in matches]
dxs.sort()
dys.sort()
median_dx = dxs[len(dxs)//2]
median_dy = dys[len(dys)//2]
print(f"\nRobust Median Stereo Shift: dx = {median_dx} px, dy = {median_dy} px")

# Save vertically rectified left and right PPMs
header = ("P6\n" + str(W) + " " + str(H) + "\n255\n").encode()
with open("/home/steamos/test_openvr/rectified_left.ppm", "wb") as f:
    f.write(header)
    f.write(rawL)

with open("/home/steamos/test_openvr/rectified_right.ppm", "wb") as f:
    f.write(header)
    shifted_right = bytearray(W * H * 3)
    for y in range(H):
        src_y = y + median_dy
        if 0 <= src_y < H:
            shifted_right[y * W * 3 : (y+1) * W * 3] = rawR[src_y * W * 3 : (src_y+1) * W * 3]
    f.write(shifted_right)

print("Saved /home/steamos/test_openvr/rectified_left.ppm and rectified_right.ppm!")
