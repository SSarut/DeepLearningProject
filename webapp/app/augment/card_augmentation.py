"""
4 augmentation functions for playing-card PNG images.

  1. augment_rotate_scale_noise   - Rotate -> Scale -> Background -> Noise
  2. augment_3d_warp_noise        - 3-D Tilt -> Perspective Warp -> Background -> Noise
  3. augment_rotate_partial       - Rotate -> Background -> Partial Visibility
  4. augment_3d_partial           - 3-D Tilt -> Background -> Partial Visibility

Requirements:  pip install opencv-python numpy pillow
Usage:
    from card_augmentation import process_dataset
    process_dataset("dataset/", "dataset_augmented/")

    # CLI - single card preview
    python card_augmentation.py card.png
    # CLI - full dataset
    python card_augmentation.py input_dir/ output_dir/
"""

import cv2
import math
import random
from pathlib import Path

import numpy as np

try:
    from PIL import Image, PngImagePlugin
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS  -  all random values are passed in explicitly
# ══════════════════════════════════════════════════════════════════════════════

def _load_card(path: str) -> np.ndarray:
    """Load any PNG as BGRA (adds alpha channel when missing)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2: # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3: # BGR, no alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img  # always BGRA


def _make_background(w: int, h: int,
                     kind: str,
                     color1: list,
                     color2: list) -> np.ndarray:
    """
    Return an (H, W, 3) uint8 BGR background.

    kind   : 'solid' | 'gradient' | 'fabric'
    color1 : [B, G, R]  base colour for all three modes
    color2 : [B, G, R]  second colour (gradient end / unused for solid)
    """
    if kind == "solid":
        return np.full((h, w, 3), color1, dtype=np.uint8)

    if kind == "gradient":
        c1 = np.array(color1, dtype=np.float32)
        c2 = np.array(color2, dtype=np.float32)
        t  = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
        bg = (c1 * (1 - t) + c2 * t).astype(np.uint8) # (H, 1, 3)
        return np.broadcast_to(bg, (h, w, 3)).copy()

    # fabric: low-frequency coloured noise (felt / cloth look)
    base  = np.array(color1, dtype=np.int16)
    noise = np.random.randint(-25, 26, (h, w, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _composite(card_bgra: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """
    Alpha-blend card_bgra (H_c, W_c, 4) centred on bg (H, W, 3).
    Shrinks the card when it is larger than the background.
    Returns a BGR (H, W, 3) image.
    """
    bh, bw = bg.shape[:2]
    ch, cw = card_bgra.shape[:2]

    # Scale down card to fit inside background (with a small margin)
    if ch > bh or cw > bw:
        scale    = min(bh / ch, bw / cw) * 0.92
        card_bgra = cv2.resize(card_bgra,
                               (max(1, int(cw * scale)), max(1, int(ch * scale))))
        ch, cw   = card_bgra.shape[:2]

    y0 = (bh - ch) // 2
    x0 = (bw - cw) // 2

    result   = bg.copy().astype(np.float32)
    alpha    = card_bgra[:, :, 3:4].astype(np.float32) / 255.0   # (H_c, W_c, 1)
    card_rgb = card_bgra[:, :, :3].astype(np.float32)

    roi = result[y0:y0 + ch, x0:x0 + cw]
    result[y0:y0 + ch, x0:x0 + cw] = alpha * card_rgb + (1.0 - alpha) * roi

    return result.astype(np.uint8)


def _rotate_card(card_bgra: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate card_bgra by `angle` degrees with an expanded canvas so no
    corner is clipped.  Returns BGRA.
    """
    h, w  = card_bgra.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(card_bgra, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def _perspective_3d(card_bgra: np.ndarray,
                    tilt: float,
                    axis: str,
                    side: int) -> np.ndarray:
    """
    Simulate rotation around a random X or Y axis by remapping the four
    corners to a foreshortened trapezoid.  Returns BGRA, same canvas size.

    tilt : degrees of rotation (20-50)
    axis : 'x' (top/bottom recedes) | 'y' (left/right recedes)
    side : 1 or -1 (which side recedes)

    Returns BGRA, same canvas size.
    """
    h, w  = card_bgra.shape[:2]
    cos_a = math.cos(math.radians(tilt))
    src   = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if axis == "y":
        margin = (h * (1.0 - cos_a)) / 2.0
        if side == 1: # right edge recedes -> right becomes shorter
            dst = np.float32([[0, 0], 
                              [w, margin], 
                              [w, h - margin], 
                              [0, h]])
        else: # left edge recedes
            dst = np.float32([[0, margin], 
                              [w, 0], 
                              [w, h], 
                              [0, h - margin]])
    else:  # axis == "x"
        margin = (w * (1.0 - cos_a)) / 2.0
        if side == 1: # bottom edge recedes -> bottom becomes narrower
            dst = np.float32([[0, 0], 
                              [w, 0], 
                              [w - margin, h], 
                              [margin, h]])
        else: # top edge recedes
            dst = np.float32([[margin, 0], 
                              [w - margin, 0], 
                              [w, h], 
                              [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(card_bgra, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def _subtle_warp(card_bgra: np.ndarray,
                 jitter: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """
    Apply a gentle random perspective warp to simulate a slightly off-axis camera angle.
    `rng` is a seeded numpy RandomState for full reproducibility via warp_seed.

    Returns BGRA, same canvas size.
    """
    h, w   = card_bgra.shape[:2]
    offset = jitter * min(w, h)
    src    = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst    = (src + rng.uniform(-offset, offset, src.shape)).astype(np.float32)
    M      = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(card_bgra, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


def _add_noise(image: np.ndarray,
               sigma: float,
               sp: float) -> np.ndarray:
    """
    Add Gaussian noise + sparse salt-and-pepper noise.
    Simulates camera sensor noise / image compression artefacts.

    sigma : Gaussian standard deviation  (3.0-15.0)
    sp    : salt-and-pepper pixel fraction  (0.001-0.008)
    """
    # Gaussian
    gauss = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out   = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    # Salt-and-pepper
    n_pix = int(out.size * sp)
    ys = np.random.randint(0, out.shape[0], n_pix)
    xs = np.random.randint(0, out.shape[1], n_pix)
    out[ys, xs] = 255
    ys = np.random.randint(0, out.shape[0], n_pix)
    xs = np.random.randint(0, out.shape[1], n_pix)
    out[ys, xs] = 0

    return out


def _partial_visibility(image: np.ndarray,
                        canvas_w: int,
                        canvas_h: int,
                        visibility: float,
                        use_occlusion: bool,
                        use_corner: bool,
                        edge_or_corner: str) -> np.ndarray:
    """
    Hide part of the card image.

    Occlusion   : zero out alpha in the hidden region -> background shows through.
    Out-of-frame: place the card on a transparent canvas shifted so part falls outside the frame.

    visibility      : controls what fraction of the image stays visible (0.65-0.80)
    use_occlusion   : True  -> paint a rectangle over part of card
                    : False -> shift card so part is off-frame
    use_corner      : True  -> occlude a corner rather than a full edge
    edge_or_corner  : which edge ('top'|'bottom'|'left'|'right') or
                      corner ('tl'|'tr'|'bl'|'br')

    Returns a BGRA ndarray at new (canvas_h, canvas_w).
    """
    h, w   = image.shape[:2]
    cover  = 1.0 - visibility

    if use_occlusion:
        out = image.copy()
        if use_corner: # Corner occlusion
            cx = int(w * cover * 1.5)
            cy = int(h * cover * 1.5)
            if edge_or_corner == "tl":
                out[:cy, :cx] = 0
            elif edge_or_corner == "tr":
                out[:cy, w - cx:] = 0
            elif edge_or_corner == "bl":
                out[h - cy:, :cx] = 0
            else: # "br"
                out[h - cy:, w - cx:] = 0
        else: # Full-edge occlusion
            if edge_or_corner == "top":
                out[:int(h * cover), :] = 0
            elif edge_or_corner == "bottom":
                out[h - int(h * cover):, :] = 0
            elif edge_or_corner == "left":
                out[:, :int(w * cover)] = 0
            else: # "right"
                out[:, w - int(w * cover):]  = 0
        return out

    else: # Out-of-frame shift the card slides off one edge.
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        if edge_or_corner == "top":
            shift  = int(h * cover)
            y0, x0 = shift, (canvas_w - w) // 2
        elif edge_or_corner == "bottom":
            shift  = int(h * cover)
            y0, x0 = -shift, (canvas_w - w) // 2
        elif edge_or_corner == "left":
            shift  = int(w * cover)
            y0, x0 = (canvas_h - h) // 2, shift
        else: # "right"
            shift  = int(w * cover)
            y0, x0 = (canvas_h - h) // 2, -shift

        # Clip to canvas bounds and copy only the visible portion
        src_y0 = max(0, -y0);  dst_y0 = max(0, y0)
        src_x0 = max(0, -x0);  dst_x0 = max(0, x0)
        src_y1 = h - max(0, (y0 + h) - canvas_h)
        src_x1 = w - max(0, (x0 + w) - canvas_w)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

        return canvas

# ══════════════════════════════════════════════════════════════════════════════
# FILENAME & METADATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_rand(val) -> str:
    """
    Encode a random value as a compact, filename-safe string.

    Float  ->  3 decimal places with dot = "d".
              45.234 -> "45d234",  1.153 -> "1d153",  0.720 -> "0d720"
    Int    ->  as-is.  42 -> "42"
    """
    if isinstance(val, int):
        return str(val)
    return f"{val:.3f}".replace(".", "d")


def _save_image(image_bgr: np.ndarray,
                path,
                metadata: dict = None) -> None:
    """
    Save BGR image as PNG.
    When Pillow is available, embed `metadata` as PNG tEXt chunks.
    Falls back to cv2.imwrite when Pillow is absent or metadata is None.
    """
    if HAS_PILLOW and metadata:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(image_rgb)
        pnginfo   = PngImagePlugin.PngInfo()
        for k, v in metadata.items():
            pnginfo.add_text(str(k), str(v))
        pil_img.save(str(path), "PNG", pnginfo=pnginfo)
    else:
        cv2.imwrite(str(path), image_bgr)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC AUGMENTATION FUNCTIONS
#
# Every function returns a 3-tuple:
#   (image_bgr, rand_list, metadata_dict)
#
#   image_bgr     - BGR ndarray of shape (H, W, 3) ready to save
#   rand_list     - one primary float/int per named transform step;
#                   used to build the output filename (background excluded)
#   metadata_dict - full parameter record for PNG tEXt embedding
# ══════════════════════════════════════════════════════════════════════════════

def augment_rotate_scale_noise(card_path: str,
                                out_size: tuple = None):
    """
    Template 1  -  Rotate -> Scale -> Background -> Noise

    - Full 0-360° rotation (expanded canvas, no corner clipping).
    - Random scale x0.70-1.20; card is scaled back down if it exceeds
      the target canvas.
    - Random background composited behind the card.
    - Gaussian + salt-and-pepper noise applied last.

    rand_list : [angle, scale, sigma]
    """
    card   = _load_card(card_path)
    h, w   = card.shape[:2]
    target = out_size or (w, h)

    # Generate all randoms
    angle   = random.uniform(0, 360) # Rotate
    scale   = random.uniform(0.70, 1.20) # Scale
    sigma   = random.uniform(3.0, 15.0) # Gaussian
    sp      = random.uniform(0.001, 0.008) # Salt-and-Pepper
    bg_kind = random.choice(["solid", "gradient", "fabric"])
    bg_c1   = [random.randint(20, 235) for _ in range(3)]
    bg_c2   = [random.randint(20, 235) for _ in range(3)]

    # Transforms
    rotated = _rotate_card(card, angle)

    sw     = max(1, int(rotated.shape[1] * scale))
    sh     = max(1, int(rotated.shape[0] * scale))
    scaled = cv2.resize(rotated, (sw, sh))

    fit  = min(target[0] / scaled.shape[1], target[1] / scaled.shape[0], 1.0)
    if fit < 1.0: # Shrink to fit target if needed
        scaled = cv2.resize(scaled,
                            (max(1, int(scaled.shape[1] * fit)),
                             max(1, int(scaled.shape[0] * fit))))
    
    bg = _make_background(target[0], target[1], bg_kind, bg_c1, bg_c2)
    composited = _composite(scaled, bg)

    result = _add_noise(composited, sigma, sp)
    
    # Output data
    rand_list = [angle, scale, sigma]
    metadata  = {
        "aug_template":  1,
        "source":        card_path,
        "rotate_angle":  round(angle, 6),
        "scale_factor":  round(scale, 6),
        "noise_sigma":   round(sigma, 6),
        "noise_sp":      round(sp, 8),
        "bg_kind":       bg_kind,
        "bg_color1_bgr": str(bg_c1),
        "bg_color2_bgr": str(bg_c2),
    }
    return result, rand_list, metadata


def augment_3d_warp_noise(card_path: str,
                           out_size: tuple = None):
    """
    Template 2  -  3-D Tilt -> Perspective Warp -> Background -> Noise

    - Simulates rotation around a random X or Y axis (foreshortening).
    - A secondary gentle perspective warp adds a camera-angle feel.
    - Random background composited behind the card.
    - Gaussian + salt-and-pepper noise applied last.

    rand_list : [tilt, warp_seed, sigma]
    """
    card   = _load_card(card_path)
    h, w   = card.shape[:2]
    target = out_size or (w, h)

    # Generate all randoms
    tilt      = random.uniform(20, 50) # 3-D rotation angle
    axis      = random.choice(["x", "y"]) # 3-D recedes axis
    side      = random.choice([-1, 1]) # 3-D recedes side
    warp_seed = random.randint(0, 99999) # Perspective warp
    sigma     = random.uniform(3.0, 15.0) # Gaussian
    sp        = random.uniform(0.001, 0.008) # Salt-and-Pepper
    bg_kind   = random.choice(["solid", "gradient", "fabric"])
    bg_c1     = [random.randint(20, 235) for _ in range(3)]
    bg_c2     = [random.randint(20, 235) for _ in range(3)]

    # Transforms 
    tilted = _perspective_3d(card, tilt, axis, side)

    rng    = np.random.RandomState(warp_seed) # seeded for reproducibility
    warped = _subtle_warp(tilted, jitter=0.06, rng=rng)

    fit = min(target[0] / warped.shape[1], target[1] / warped.shape[0], 1.0)
    if fit < 1.0: # Shrink to fit target if needed
        warped = cv2.resize(warped,
                            (max(1, int(warped.shape[1] * fit)),
                             max(1, int(warped.shape[0] * fit))))

    bg = _make_background(target[0], target[1], bg_kind, bg_c1, bg_c2)
    composited = _composite(warped, bg)

    result = _add_noise(composited, sigma, sp)

    # Output data
    rand_list = [tilt, warp_seed, sigma]
    metadata  = {
        "aug_template":  2,
        "source":        card_path,
        "3d_tilt_angle": round(tilt, 6),
        "3d_tilt_axis":  axis,
        "3d_tilt_side":  side,
        "warp_seed":     warp_seed,
        "warp_jitter":   0.06,
        "noise_sigma":   round(sigma, 6),
        "noise_sp":      round(sp, 8),
        "bg_kind":       bg_kind,
        "bg_color1_bgr": str(bg_c1),
        "bg_color2_bgr": str(bg_c2),
    }
    return result, rand_list, metadata


def augment_rotate_partial(card_path: str,
                            out_size: tuple = None):
    """
    Template 3  -  Rotate -> Background -> Partial Visibility

    - Full 0-360° rotation.
    - Random background composited behind the card.
    - Partial visibility applied last (65-80 % visible,
      80 % occlusion / 20 % out-of-frame).

    rand_list : [angle, visibility]
    """
    card   = _load_card(card_path)
    h, w   = card.shape[:2]
    target = out_size or (w, h)

    # Generate all randoms
    angle         = random.uniform(0, 360) # Rotation
    visibility    = random.uniform(0.65, 0.80) # Card Visibility

    use_occlusion = random.random() < 0.80
    if use_occlusion:
        use_corner     = random.random() < 0.25
        edge_or_corner = random.choice(
            ["tl", "tr", "bl", "br"] if use_corner
            else ["top", "bottom", "left", "right"]
        )
    else:
        use_corner     = False
        edge_or_corner = random.choice(["top", "bottom", "left", "right"])

    bg_kind = random.choice(["solid", "gradient", "fabric"])
    bg_c1   = [random.randint(20, 235) for _ in range(3)]
    bg_c2   = [random.randint(20, 235) for _ in range(3)]

    # Transforms
    rotated = _rotate_card(card, angle)

    fit = min(target[0] / rotated.shape[1], target[1] / rotated.shape[0], 1.0)
    if fit < 1.0: # Shrink to fit target if needed
        rotated = cv2.resize(rotated,
                             (max(1, int(rotated.shape[1] * fit)),
                              max(1, int(rotated.shape[0] * fit))))

    partial = _partial_visibility(rotated, target[0], target[1],
                                  visibility, use_occlusion, use_corner, edge_or_corner)

    bg     = _make_background(target[0], target[1], bg_kind, bg_c1, bg_c2)
    result = _composite(partial, bg)

    # Output data
    rand_list = [angle, visibility]
    metadata  = {
        "aug_template":   3,
        "source":         card_path,
        "rotate_angle":   round(angle, 6),
        "visibility":     round(visibility, 6),
        "partial_method": ("corner_occlusion" if use_corner
                           else "edge_occlusion" if use_occlusion
                           else "out_of_frame"),
        "edge_corner":    edge_or_corner,
        "bg_kind":        bg_kind,
        "bg_color1_bgr":  str(bg_c1),
        "bg_color2_bgr":  str(bg_c2),
    }
    return result, rand_list, metadata


def augment_3d_partial(card_path: str,
                        out_size: tuple = None):
    """
    Template 4  -  3-D Tilt -> Background -> Partial Visibility

    - Simulates rotation around a random X or Y axis.
    - Random background composited behind the card.
    - Partial visibility applied last (65-80 % visible,
      80 % occlusion / 20 % out-of-frame).

    rand_list : [tilt, visibility]
      tilt       : 3-D rotation angle  (20-50°)
      visibility : fraction of card left visible  (0.65-0.80)
    """
    card   = _load_card(card_path)
    h, w   = card.shape[:2]
    target = out_size or (w, h)

    # ── Generate ALL randoms upfront ─────────────────────────────────────────
    tilt          = random.uniform(20, 50) # 3-D rotation angle
    axis          = random.choice(["x", "y"]) # 3-D recedes axis
    side          = random.choice([-1, 1]) # 3-D recedes side
    visibility    = random.uniform(0.65, 0.80) # Card Visibility

    use_occlusion = random.random() < 0.80
    if use_occlusion:
        use_corner     = random.random() < 0.25
        edge_or_corner = random.choice(
            ["tl", "tr", "bl", "br"] if use_corner
            else ["top", "bottom", "left", "right"]
        )
    else:
        use_corner     = False
        edge_or_corner = random.choice(["top", "bottom", "left", "right"])

    bg_kind = random.choice(["solid", "gradient", "fabric"])
    bg_c1   = [random.randint(20, 235) for _ in range(3)]
    bg_c2   = [random.randint(20, 235) for _ in range(3)]

    # Transforms
    tilted = _perspective_3d(card, tilt, axis, side)

    fit    = min(target[0] / tilted.shape[1], target[1] / tilted.shape[0], 1.0)
    if fit < 1.0: # Shrink to fit target if needed
        tilted = cv2.resize(tilted,
                            (max(1, int(tilted.shape[1] * fit)),
                             max(1, int(tilted.shape[0] * fit))))

    partial = _partial_visibility(tilted, target[0], target[1],
                                  visibility, use_occlusion, use_corner, edge_or_corner)

    bg     = _make_background(target[0], target[1], bg_kind, bg_c1, bg_c2)
    result = _composite(partial, bg)

    rand_list = [tilt, visibility]
    metadata  = {
        "aug_template":   4,
        "source":         card_path,
        "3d_tilt_angle":  round(tilt, 6),
        "3d_tilt_axis":   axis,
        "3d_tilt_side":   side,
        "visibility":     round(visibility, 6),
        "partial_method": ("corner_occlusion" if use_corner
                           else "edge_occlusion" if use_occlusion
                           else "out_of_frame"),
        "edge_corner":    edge_or_corner,
        "bg_kind":        bg_kind,
        "bg_color1_bgr":  str(bg_c1),
        "bg_color2_bgr":  str(bg_c2),
    }
    return result, rand_list, metadata


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_AUGMENT_FNS = [
    augment_rotate_scale_noise,   # aug1
    augment_3d_warp_noise,        # aug2
    augment_rotate_partial,       # aug3
    augment_3d_partial,           # aug4
]


def process_dataset(input_dir: str,
                    output_dir: str,
                    out_size: tuple = None) -> None:
    """
    Walk `input_dir`, find every class sub-folder containing PNG/JPG files,
    and apply all 4 augmentations to each image -> 4x images per class.

    Directory layout expected:
        input_dir/
            class_A/   ← N PNGs
            class_B/   ← N PNGs

    Output filename anatomy:
        {folder_name}_aug{1-4}_{count:03d}_{rand_per_transform...}.png

    Examples:
        ace_spades_aug1_007_45d234_1d153_8d341.png
        ace_spades_aug2_007_35d120_42d317_9d821.png
        ace_spades_aug3_007_45d234_0d720.png
        ace_spades_aug4_007_35d120_0d720.png

    PNG text metadata (requires Pillow):
        aug_template, source_file, exact transform floats,
        axis/side choices, bg_kind, bg colors, partial_method, etc.

        Read with:
          python -c "from PIL import Image; print(Image.open('f.png').info)"
          exiftool f.png
    """
    in_root  = Path(input_dir)
    out_root = Path(output_dir)

    class_dirs = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No sub-folders found in: {input_dir}")

    if not HAS_PILLOW:
        print("  Note: Pillow not installed - PNG metadata will not be embedded.")
        print("        Install with: pip install Pillow\n")

    for class_dir in class_dirs:
        imgs = sorted(class_dir.glob("*.png"))
        imgs += sorted(class_dir.glob("*.jpg"))
        out_cls = out_root / class_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)

        print(f"[{class_dir.name}]  {len(imgs)} source images x 4 augmentations …")

        for count, img_path in enumerate(imgs):
            for fn_idx, fn in enumerate(_AUGMENT_FNS, start=1):
                augmented, rand_list, metadata = fn(str(img_path), out_size)

                # Build filename
                rand_parts = [_fmt_rand(v) for v in rand_list]
                filename   = (
                    f"{class_dir.name}"
                    f"_aug{fn_idx}"
                    f"_{count:03d}"
                    f"_{'_'.join(rand_parts)}"
                    f".png"
                )

                metadata["source_file"] = img_path.name
                _save_image(augmented, out_cls / filename, metadata)

        print(f"  -> {len(imgs) * 4} images written to {out_cls}")

    print("\nAll classes done.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        process_dataset(sys.argv[1], sys.argv[2])

    elif len(sys.argv) == 2:
        # Single-card preview - saves 4 augmented images next to the source.
        src     = sys.argv[1]
        stem    = Path(src).stem
        out_dir = Path(src).parent

        print(f"Preview mode - augmenting: {src}\n")
        for idx, fn in enumerate(_AUGMENT_FNS, start=1):
            img, rand_list, meta = fn(src)
            rand_parts = [_fmt_rand(v) for v in rand_list]
            fname = (
                f"{stem}_aug{idx}_000"
                f"_{'_'.join(rand_parts)}"
                f"_preview.png"
            )
            path = out_dir / fname
            _save_image(img, str(path), meta)
            print(f"  Saved: {path}")
            print(f"  Rands: {rand_list}\n")

    else:
        print("Usage:")
        print("  Single card preview : python card_augmentation.py card.png")
        print("  Full dataset        : python card_augmentation.py input_dir/ output_dir/")