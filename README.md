# Other Resource
Such as weights, presentation slide. dataset can be found here:
https://drive.google.com/drive/folders/17b0KGZ6-bqh9Hh-biAbiDLO3msndEnfE?usp=sharing

Student ID:
- 65010700
- 65010966
- 65011019

# Augmentation Algorithm

## Function 1 — augment_rotate_scale_noise
Character: clean geometry variation with colour correction and sensor noise

### Card BGRA
  * White Balance Shift     (simulate camera colour temperature)
  * Colour Jitter           (hue ±30°, sat 0.7–1.3, channel ±30)
  * Brightness/Contrast     (alpha 0.7–1.3, beta ±40)
  * Square Canvas
  * Rotate (0–360°)
  * Scale (0.70–1.20×)
  * Background
  * Blur (sigma 0.3–1.5)
  * Noise (gaussian 3–8, sp 0.001–0.003)
  ( JPEG Artifact (60–95)

rand_list : [angle, scale, blur_sigma, sigma]


---

## Function 2 — augment_3d_warp_noise
Character: 3D camera angle with physical card imperfection and glare

### Card BGRA
  * White Balance Shift
  * Brightness/Contrast     (alpha 0.6–1.4, beta ±50)
  * Card Warp               (amplitude 3–12, frequency 0.5–1.5, sine bend)
  * Square Canvas
  * 3D Tilt                 (20–50°, random axis/side)
  * Perspective Warp        (seeded jitter 0.06)
  * Background
  * Glare                   (soft ellipse, intensity 0.4–0.85)
  * Blur (sigma 0.3–1.5)
  * Noise (gaussian 3–8, sp 0.001–0.003)
  * JPEG Artifact (60–95)

rand_list : [tilt, warp_seed, blur_sigma, sigma]


---

## Function 3 — augment_rotate_partial
Character: occlusion/framing with strongest colour stress

### Card BGRA
  * White Balance Shift
  * Colour Jitter           (hue ±60°, sat 0.4–1.6, channel ±60) ← strongest
  * Brightness/Contrast     (alpha 0.7–1.3, beta ±40)
  * Card Warp               (amplitude 3–12, frequency 0.5–1.5)
  * Square Canvas
  * Rotate (0–360°)
  * Partial Visibility      (65–80%, 80% occlusion / 20% out-of-frame)
  * Background
  * JPEG Artifact (60–95)

rand_list : [angle, visibility]


---
## Function 4 — augment_3d_partial
Character: 3D angle with occlusion and dramatic lighting

### Card BGRA
  * White Balance Shift
  * Brightness/Contrast     (alpha 0.5–1.5, beta ±60) ← strongest
  * Square Canvas
  * 3D Tilt                 (20–50°, random axis/side)
  * Partial Visibility      (65–80%, 80% occlusion / 20% out-of-frame)
  * Background
  * Glare                   (soft ellipse, intensity 0.4–0.85)
  * JPEG Artifact (60–95)

rand_list : [tilt, visibility]


---

## Function 5 — augment_colour_stress
Character: extreme colour destruction, forces model to learn frame shape not colour

### Card BGRA
  * White Balance Shift     (±0.4, widest range)
  * Colour Jitter           (hue ±60°, sat 0.3–1.7, channel ±60)
  * Brightness/Contrast     (alpha 0.5–1.5, beta ±60)
  * Greyscale               (15% chance — forces structure learning)
  * Square Canvas
  * Background
  * JPEG Artifact (50–95)   ← lowest floor, most compression artefacts

rand_list : [wb_temp, hue_shift, bc_alpha]


---
