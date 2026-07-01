import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


def _load_gray_and_rgb(image_path, size=768, crop_frac=0.45):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    crop_w, crop_h = int(w * crop_frac), int(h * crop_frac)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((size, size), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.float64)
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    return gray, rgb


def _fft_features(gray):
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f) + 1.0)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_norm = r / (h / 2)

    low = mag[(r_norm > 0.02) & (r_norm <= 0.15)]
    mid = mag[(r_norm > 0.15) & (r_norm <= 0.45)]
    high = mag[(r_norm > 0.45) & (r_norm <= 0.95)]

    low_e, mid_e, high_e = low.mean(), mid.mean(), high.mean()

    # Peakiness: ratio of max to mean energy in the mid band. Regular screen
    # pixel grids create sharp narrow spikes; natural textures are smoother.
    mid_peak_ratio = mid.max() / (mid.mean() + 1e-6)

    # Peak count: number of mid-band bins that are statistical outliers
    # relative to the local mid-band distribution. A periodic moire/pixel
    # grid produces a handful of sharp narrow spikes; natural texture
    # produces a smooth, noise-like spectrum with few or no such outliers.
    mid_thresh = mid.mean() + 3 * mid.std()
    peak_count = int((mid > mid_thresh).sum())

    return {
        "fft_low_energy": low_e,
        "fft_mid_energy": mid_e,
        "fft_high_energy": high_e,
        "fft_mid_high_ratio": mid_e / (high_e + 1e-6),
        "fft_mid_peak_ratio": mid_peak_ratio,
        "fft_peak_count": peak_count,
    }


def _color_features(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Color temperature proxy: blue/red channel balance.
    blue_red_ratio = (b.mean() + 1e-6) / (r.mean() + 1e-6)

    # Gamut coverage proxy: std across channel means (screens often have a
    # narrower / shifted gamut than natural scenes).
    channel_std = np.std([r.mean(), g.mean(), b.mean()])

    # Saturation stats (HSV)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    sat_mean = sat.mean()

    # Highlight clipping: fraction of near-white pixels (specular glare off
    # a screen surface tends to clip harder / more uniformly than real glare).
    clipped_frac = np.mean(mx > 250)

    return {
        "color_blue_red_ratio": blue_red_ratio,
        "color_channel_std": channel_std,
        "color_sat_mean": sat_mean,
        "color_clipped_frac": clipped_frac,
    }


def _texture_features(gray):
    # Laplacian variance: overall sharpness/detail.
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    lap_var = lap.var()

    # Local contrast uniformity: std of local 8x8 block stds. Screens lit by
    # uniform backlight often have more uniform local contrast than varied
    # real-world lighting/texture.
    h, w = gray.shape
    bs = 32
    block_stds = []
    for i in range(0, h - bs, bs):
        for j in range(0, w - bs, bs):
            block_stds.append(gray[i : i + bs, j : j + bs].std())
    block_stds = np.array(block_stds)

    return {
        "tex_laplacian_var": lap_var,
        "tex_block_std_mean": block_stds.mean(),
        "tex_block_std_var": block_stds.var(),
        "tex_global_std": gray.std(),
    }


def _blockiness_features(gray):
    h, w = gray.shape
    # horizontal gradient strength at 8-aligned columns vs non-aligned
    gx = np.abs(np.diff(gray, axis=1))
    cols = np.arange(gx.shape[1])
    aligned = gx[:, (cols % 8 == 7)]
    nonaligned = gx[:, (cols % 8 != 7)]
    h_blockiness = aligned.mean() / (nonaligned.mean() + 1e-6)

    gy = np.abs(np.diff(gray, axis=0))
    rows = np.arange(gy.shape[0])
    aligned_r = gy[(rows % 8 == 7), :]
    nonaligned_r = gy[(rows % 8 != 7), :]
    v_blockiness = aligned_r.mean() / (nonaligned_r.mean() + 1e-6)

    return {
        "block_h_ratio": h_blockiness,
        "block_v_ratio": v_blockiness,
    }


def _edge_orientation_features(gray):
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)  # -pi..pi

    # only consider edges with non-trivial magnitude
    mask = mag > np.percentile(mag, 80)
    ang_strong = ang[mask]

    # fold angle into 0..90 degrees (axis-aligned vs diagonal)
    ang_deg = np.degrees(ang_strong) % 90
    # distance to nearest axis (0 or 90)
    dist_to_axis = np.minimum(ang_deg, 90 - ang_deg)
    axis_bias = 1.0 - (dist_to_axis.mean() / 45.0)  # 1 = all axis-aligned, 0 = uniform/diagonal

    return {
        "edge_axis_bias": axis_bias,
    }


def _noise_residual_feature(gray):
    h, w = gray.shape
    cs = 256
    cy, cx = h // 2, w // 2
    crop = gray[cy - cs // 2 : cy + cs // 2, cx - cs // 2 : cx + cs // 2]
    med = median_filter(crop, size=3)
    residual = crop - med
    return {"noise_residual_std": residual.std()}


def _color_fringe_features(rgb):
    def highpass(ch):
        return ch - np.roll(ch, 1, axis=0) / 2 - np.roll(ch, 1, axis=1) / 2

    r, g, b = highpass(rgb[..., 0]), highpass(rgb[..., 1]), highpass(rgb[..., 2])
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]
    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]

    return {
        "color_fringe_corr": (corr_rg + corr_gb + corr_rb) / 3,
    }


FEATURE_NAMES = [
    "fft_low_energy",
    "fft_mid_energy",
    "fft_high_energy",
    "fft_mid_high_ratio",
    "fft_mid_peak_ratio",
    "fft_peak_count",
    "color_blue_red_ratio",
    "color_channel_std",
    "color_sat_mean",
    "color_clipped_frac",
    "tex_laplacian_var",
    "tex_block_std_mean",
    "tex_block_std_var",
    "tex_global_std",
    "block_h_ratio",
    "block_v_ratio",
    "edge_axis_bias",
    "color_fringe_corr",
    "noise_residual_std",
]


def extract_features(image_path, crop_frac=0.45):
    gray, rgb = _load_gray_and_rgb(image_path, crop_frac=crop_frac)
    feats = {}
    feats.update(_fft_features(gray))
    feats.update(_color_features(rgb))
    feats.update(_texture_features(gray))
    feats.update(_blockiness_features(gray))
    feats.update(_edge_orientation_features(gray))
    feats.update(_color_fringe_features(rgb))
    feats.update(_noise_residual_feature(gray))
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
