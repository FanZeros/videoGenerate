#!/usr/bin/env python3
"""Compose Gate of Finality 30s 16:9 intro from storyboard stills + two AI clips."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = "/workspace/videoGenerate/gate-of-finality-intro"
ASSETS = "/workspace/assets"
SHOTS = os.path.join(ASSETS, "image/shots")
WORK = os.path.join(ROOT, "work")
FRAMES = os.path.join(WORK, "frames")
AUDIO_DIR = os.path.join(WORK, "audio")
OUT_DIR = os.path.join(ROOT, "output")
PREVIEW = os.path.join(ASSETS, "image/preview")

W, H, FPS = 1920, 1080, 24
DURATION = 30.0
NFRAMES = int(DURATION * FPS)  # 720
SR = 48000

FONT_REG = os.path.join(WORK, "fonts/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf")
FONT_BD = os.path.join(WORK, "fonts/OTF/SimplifiedChinese/NotoSerifCJKsc-SemiBold.otf")

CLIP_A = os.path.join(ASSETS, "video/cgt-20260915211652-htxz9_video.mp4")
CLIP_B = os.path.join(ASSETS, "video/cgt-20260915212032-2j7tk_video.mp4")

VO_V1 = os.path.join(ASSETS, "audio/voice")
BGM = os.path.join(ASSETS, "audio/music_1789478399967.ogg")
SFX = os.path.join(ASSETS, "audio/sfx")

LETTER_LINES = [
    "致我从未谋面的孩子。",
    "当你读到这封信时，我大概已经不在了。",
    "很多年前，我带着同伴走向那扇能解答一切的门。",
    "门开了。里面不是答案，是吞噬一切的光。",
    "我把这顶帽子留给你。它不是荣耀，是提醒。",
    "如果有一天，你也听到那扇门的呼唤——",
    "不要一个人去。",
]
LETTER_SIGN = "——父亲"


def ensure_dirs() -> None:
    for d in (FRAMES, AUDIO_DIR, OUT_DIR, PREVIEW, os.path.join(WORK, "video")):
        os.makedirs(d, exist_ok=True)


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def load_bgr(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("failed to read " + path)
    if im.shape[1] != W or im.shape[0] != H:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)
    return im


def kenburns(
    img: np.ndarray,
    p: float,
    z0: float,
    z1: float,
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
) -> np.ndarray:
    """p in 0..1, zoom >=1, anchor is crop-center in 0..1 image coords."""
    e = ease_in_out(p)
    z = z0 + (z1 - z0) * e
    ax = ax0 + (ax1 - ax0) * e
    ay = ay0 + (ay1 - ay0) * e
    h0, w0 = img.shape[:2]
    cw = w0 / z
    ch = h0 / z
    cx = ax * w0
    cy = ay * h0
    x0 = cx - cw / 2.0
    y0 = cy - ch / 2.0
    x0 = max(0.0, min(x0, w0 - cw))
    y0 = max(0.0, min(y0, h0 - ch))
    x1 = x0 + cw
    y1 = y0 + ch
    M = np.array(
        [
            [W / cw, 0.0, -x0 * W / cw],
            [0.0, H / ch, -y0 * H / ch],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crossfade(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    k = max(0.0, min(1.0, k))
    return cv2.addWeighted(a, 1.0 - k, b, k, 0)


def detect_paper_bbox(img_rgb: Image.Image) -> Tuple[int, int, int, int]:
    arr = np.array(img_rgb)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    warm = (r.astype(np.int16) - b.astype(np.int16)) > 18
    mask = (lum > 95) & (lum < 210) & warm
    # keep largest bright blob near center
    ys, xs = np.where(mask)
    if xs.size < 4000:
        return 360, 140, 1560, 940
    x0, x1 = int(np.percentile(xs, 4)), int(np.percentile(xs, 96))
    y0, y1 = int(np.percentile(ys, 6)), int(np.percentile(ys, 94))
    pad = 18
    x0 = max(80, x0 + pad)
    y0 = max(60, y0 + pad)
    x1 = min(W - 80, x1 - pad)
    y1 = min(H - 40, y1 - pad)
    if x1 - x0 < 900 or y1 - y0 < 520:
        return 360, 140, 1560, 940
    return x0, y0, x1, y1


def render_letter() -> str:
    parchment = Image.open(os.path.join(SHOTS, "shot_14.png")).convert("RGB")
    parchment = ImageEnhance.Contrast(parchment).enhance(1.06)
    parchment = ImageEnhance.Color(parchment).enhance(1.04)
    # shot_14 paper sits in the center of the desk; keep a conservative inset
    x0, y0, x1, y1 = 430, 180, 1490, 900
    print("paper bbox", x0, y0, x1, y1)

    img = parchment.copy()
    draw = ImageDraw.Draw(img)
    box_w = x1 - x0
    box_h = y1 - y0

    font_size = 46
    font = ImageFont.truetype(FONT_REG, font_size)
    sign_font = ImageFont.truetype(FONT_BD, 44)

    def fits(sz: int) -> bool:
        f = ImageFont.truetype(FONT_REG, sz)
        line_h = int(sz * 1.62)
        total = line_h * (len(LETTER_LINES) + 1) + 20
        if total > box_h - 30:
            return False
        for line in LETTER_LINES:
            bb = draw.textbbox((0, 0), line, font=f)
            if bb[2] - bb[0] > box_w - 48:
                return False
        return True

    while font_size >= 32 and not fits(font_size):
        font_size -= 2
        font = ImageFont.truetype(FONT_REG, font_size)
    line_h = int(font_size * 1.62)
    total_h = line_h * len(LETTER_LINES) + 56
    ty = y0 + max(12, (box_h - total_h) // 2)
    tx = x0 + 36
    ink = (52, 32, 18)

    def draw_ink(pos: Tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont) -> None:
        x, y = pos
        draw.text((x + 1, y + 1), text, font=fnt, fill=(24, 14, 8))
        draw.text((x, y), text, font=fnt, fill=ink)

    for i, line in enumerate(LETTER_LINES):
        draw_ink((tx, ty + i * line_h), line, font)

    sb = draw.textbbox((0, 0), LETTER_SIGN, font=sign_font)
    sw = sb[2] - sb[0]
    draw_ink((x1 - 48 - sw, ty + len(LETTER_LINES) * line_h + 8), LETTER_SIGN, sign_font)

    # gentle vignette
    vig = Image.new("L", img.size, 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse((-120, -80, W + 80, H + 40), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(90))
    dark = Image.new("RGB", img.size, (14, 8, 6))
    img = Image.composite(img, Image.blend(img, dark, 0.28), vig)

    path15 = os.path.join(FRAMES, "letter_full.png")
    img.save(path15)
    print("wrote", path15, "font", font_size)
    return path15


def load_clip_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError("empty clip " + path)
    print("clip", path, "frames", len(frames))
    return frames


class Shot:
    def __init__(
        self,
        start: float,
        end: float,
        img: np.ndarray,
        z0: float,
        z1: float,
        a0: Tuple[float, float],
        a1: Tuple[float, float],
    ) -> None:
        self.start = start
        self.end = end
        self.img = img
        self.z0 = z0
        self.z1 = z1
        self.a0 = a0
        self.a1 = a1

    def at(self, t: float) -> np.ndarray:
        dur = max(1e-4, self.end - self.start)
        p = (t - self.start) / dur
        return kenburns(self.img, p, self.z0, self.z1, self.a0[0], self.a0[1], self.a1[0], self.a1[1])


def build_video(letter15: np.ndarray) -> str:
    clip_a = load_clip_frames(CLIP_A)
    clip_b = load_clip_frames(CLIP_B)
    shots_img = {i: load_bgr(os.path.join(SHOTS, f"shot_{i:02d}.png")) for i in range(1, 17)}
    shots_img[15] = letter15
    # keep original 16 (letter + hat hold)

    stills = [
        Shot(8.00, 10.00, shots_img[5], 1.04, 1.12, (0.50, 0.52), (0.50, 0.48)),
        Shot(10.00, 12.00, shots_img[6], 1.02, 1.14, (0.50, 0.55), (0.50, 0.46)),
        Shot(12.00, 14.00, shots_img[7], 1.06, 1.16, (0.46, 0.52), (0.58, 0.46)),
        Shot(14.00, 16.00, shots_img[8], 1.02, 1.10, (0.50, 0.50), (0.50, 0.46)),
        Shot(16.00, 17.55, shots_img[9], 1.04, 1.14, (0.48, 0.52), (0.42, 0.40)),
        Shot(17.55, 19.00, shots_img[10], 1.08, 1.20, (0.50, 0.50), (0.52, 0.46)),
        Shot(19.00, 20.40, shots_img[11], 1.10, 1.22, (0.50, 0.50), (0.50, 0.48)),
        Shot(20.40, 22.00, shots_img[12], 1.04, 1.12, (0.55, 0.52), (0.62, 0.46)),
        Shot(22.00, 24.00, shots_img[13], 1.06, 1.16, (0.50, 0.48), (0.50, 0.42)),
        Shot(24.00, 25.80, shots_img[14], 1.02, 1.08, (0.50, 0.52), (0.50, 0.48)),
        Shot(25.80, 28.40, shots_img[15], 1.00, 1.06, (0.50, 0.50), (0.50, 0.48)),
        Shot(28.40, 30.00, shots_img[16], 1.02, 1.08, (0.52, 0.50), (0.48, 0.48)),
    ]

    fade = 0.12  # seconds
    raw_path = os.path.join(WORK, "video/intro_video_only.mp4")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        raw_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    def clip_frame(frames: List[np.ndarray], local_t: float, clip_dur: float) -> np.ndarray:
        if clip_dur <= 0:
            return frames[-1]
        p = max(0.0, min(0.999, local_t / clip_dur))
        idx = int(p * (len(frames) - 1))
        return frames[idx]

    preview_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 26.5, 29.4]
    preview_saved = set()

    for i in range(NFRAMES):
        t = i / float(FPS)
        if t < 8.0 - fade:
            if t < 4.0:
                frame = clip_frame(clip_a, t, 4.0)
            else:
                frame = clip_frame(clip_b, t - 4.0, 4.0)
        else:
            active: List[Tuple[Shot, float]] = []
            for s in stills:
                if t < s.start - fade or t >= s.end:
                    continue
                if t < s.start:
                    wgt = ease_in_out(1.0 - (s.start - t) / fade)
                else:
                    remaining = s.end - t
                    if remaining < fade:
                        wgt = ease_in_out(remaining / fade)
                    else:
                        wgt = 1.0
                if wgt > 0.001:
                    active.append((s, wgt))
            if not active:
                if t < 8.0:
                    frame = clip_frame(clip_b, t - 4.0, 4.0)
                else:
                    frame = stills[-1].at(min(t, stills[-1].end - 1e-4))
            elif len(active) == 1:
                frame = active[0][0].at(max(t, active[0][0].start))
            else:
                frame = active[0][0].at(max(t, active[0][0].start))
                acc = active[0][1]
                for s, wgt in active[1:]:
                    nxt = s.at(max(t, s.start))
                    k = wgt / max(acc + wgt, 1e-6)
                    frame = crossfade(frame, nxt, k)
                    acc += wgt
            if t < 8.0:
                clip = clip_frame(clip_b, t - 4.0, 4.0)
                k = ease_in_out((t - (8.0 - fade)) / fade)
                frame = crossfade(clip, frame, k)

        proc.stdin.write(frame.tobytes())

        for pt in preview_times:
            key = int(pt * 10)
            if key in preview_saved:
                continue
            if abs(t - pt) < (0.5 / FPS):
                tag = ("%04.1f" % pt).replace(".", "p")
                outp = os.path.join(PREVIEW, "compose_" + tag + "s.png")
                cv2.imwrite(outp, frame)
                preview_saved.add(key)

        if i % 120 == 0:
            print(f"video {i}/{NFRAMES} t={t:.2f}")

    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg video encode failed " + str(rc))
    print("wrote", raw_path)
    return raw_path


def load_wav(path: str, sr: int = SR) -> np.ndarray:
    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-ac", "2", "-ar", str(sr), "-f", "f32le", "pipe:1",
        ]
    )
    x = np.frombuffer(raw, dtype=np.float32)
    if x.size % 2 == 1:
        x = x[:-1]
    return x.reshape(-1, 2).copy()


def atempo_file(src: str, dst: str, tempo: float) -> None:
    tempo = max(0.5, min(2.0, tempo))
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-v", "error", "-i", src,
            "-filter:a", f"atempo={tempo:.4f}",
            "-ar", str(SR), "-ac", "2", dst,
        ]
    )


def place(mix: np.ndarray, src: np.ndarray, t0: float, gain: float) -> None:
    start = int(t0 * SR)
    if start >= mix.shape[0] or start < 0:
        return
    n = min(src.shape[0], mix.shape[0] - start)
    if n <= 0:
        return
    mix[start:start + n] += src[:n] * gain


def fade_audio(x: np.ndarray, fade_in: float, fade_out: float, sr: int = SR) -> np.ndarray:
    n = x.shape[0]
    env = np.ones((n, 1), dtype=np.float32)
    fi = int(fade_in * sr)
    fo = int(fade_out * sr)
    if fi > 0:
        env[:fi, 0] = np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        env[n - fo:, 0] = np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return x * env


def loop_to(x: np.ndarray, seconds: float) -> np.ndarray:
    need = int(seconds * SR)
    if x.shape[0] >= need:
        return x[:need]
    reps = int(math.ceil(need / max(1, x.shape[0])))
    y = np.tile(x, (reps, 1))
    return y[:need]


def build_audio() -> str:
    n = int(DURATION * SR)
    mix = np.zeros((n, 2), dtype=np.float32)

    vo_plan = [
        # file, start, atempo, gain
        ("vo_gate_lines_1_3d67aea450b6f695.ogg", 0.20, 1.74, 1.12),
        ("vo_gate_lines_2_3d67aea450b6f695.ogg", 2.55, 1.64, 1.12),
        ("vo_gate_lines_3_3d67aea450b6f695.ogg", 4.80, 1.75, 1.12),
        ("vo_gate_lines_4_3d67aea450b6f695.ogg", 7.10, 1.78, 1.10),
        ("vo_gate_lines_5_3d67aea450b6f695.ogg", 9.95, 1.49, 1.10),
        ("vo_gate_lines_6_3d67aea450b6f695.ogg", 12.15, 1.23, 1.08),
        ("vo_gate_lines_7_3d67aea450b6f695.ogg", 18.80, 1.00, 1.10),
        ("vo_gate_lines_8_3d67aea450b6f695.ogg", 24.10, 1.00, 1.10),
    ]
    for name, t0, tempo, gain in vo_plan:
        src = os.path.join(VO_V1, name)
        dst = os.path.join(AUDIO_DIR, name.replace(".ogg", f"_t{tempo:.2f}.wav"))
        atempo_file(src, dst, tempo)
        wav = load_wav(dst)
        # tiny fade to avoid clicks
        wav = fade_audio(wav, 0.02, 0.06)
        place(mix, wav, t0, gain)
        print("VO", name, "t0", t0, "dur", wav.shape[0] / SR, "end", t0 + wav.shape[0] / SR)

    bgm = load_wav(BGM)
    bgm = bgm[:n]
    bgm = fade_audio(bgm, 0.6, 2.2)
    place(mix, bgm, 0.0, 0.16)

    sfx_plan = [
        ("sfx_sky_tear.mp3", 2.05, 0.55),
        ("sfx_door_open.mp3", 10.15, 0.50),
        ("sfx_dragon_distant.mp3", 12.25, 0.48),
        ("sfx_candle_room.mp3", 16.05, 0.28),
        ("sfx_wax_break.mp3", 22.15, 0.42),
        ("sfx_paper_unfold.mp3", 24.05, 0.40),
    ]
    for name, t0, gain in sfx_plan:
        wav = load_wav(os.path.join(SFX, name))
        wav = fade_audio(wav, 0.02, 0.12)
        if "candle" in name:
            wav = loop_to(wav, 8.0)
            wav = fade_audio(wav, 0.4, 1.2)
        place(mix, wav, t0, gain)

    # very low wind bed 0-16s from filtered noise
    rng = np.random.RandomState(7)
    wind_n = int(16.5 * SR)
    wind = rng.randn(wind_n, 2).astype(np.float32) * 0.04
    # cheap one-pole lowpass
    a = 0.0015
    acc = np.zeros(2, dtype=np.float32)
    for i in range(wind_n):
        acc = acc + a * (wind[i] - acc)
        wind[i] = acc
    wind = fade_audio(wind, 1.0, 2.0)
    place(mix, wind, 0.0, 1.0)

    peak = float(np.max(np.abs(mix))) + 1e-8
    if peak > 0.95:
        mix *= 0.95 / peak
    print("audio peak", peak, "after", float(np.max(np.abs(mix))))

    wav_path = os.path.join(AUDIO_DIR, "mix_30s.wav")
    pcm = np.clip(mix, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    raw_pcm = os.path.join(AUDIO_DIR, "mix_30s.s16le")
    pcm16.tofile(raw_pcm)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", raw_pcm,
            wav_path,
        ]
    )
    return wav_path


def mux(video_path: str, audio_path: str, out_path: str) -> None:
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_path, "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            "-t", "30",
            out_path,
        ]
    )


def main() -> int:
    ensure_dirs()
    letter_path = render_letter()
    letter15 = load_bgr(letter_path)
    video_path = build_video(letter15)
    audio_path = build_audio()
    out_path = os.path.join(OUT_DIR, "intro_30s.mp4")
    mux(video_path, audio_path, out_path)
    game_copy = os.path.join(ASSETS, "video/intro_30s.mp4")
    subprocess.check_call(["cp", "-f", out_path, game_copy])
    print("DONE", out_path)
    subprocess.check_call(["ffprobe", "-hide_banner", out_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
