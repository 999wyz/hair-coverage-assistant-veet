"""毛发覆盖率量化 AI v2：独立于现有原型的前后照片比较基线。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "work" / "mplcache"))

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from photo_input import prepare_photo, read_uploaded_photo


# V2 has independent source code. It temporarily reuses the bundled person
# segmentation asset only to remove obvious background; it is not a hair model.
PERSON_MODEL_PATH = PROJECT_DIR / "models" / "deeplab_v3.tflite"

st.set_page_config(page_title="Hair Coverage Assistant", page_icon="📷", layout="wide")


@st.cache_resource(show_spinner=False)
def load_person_segmenter():
    if not PERSON_MODEL_PATH.exists():
        raise FileNotFoundError("未找到人体区域模型，请保留上级 models/deeplab_v3.tflite 文件。")
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(PERSON_MODEL_PATH)),
        output_category_mask=True,
        output_confidence_masks=False,
    )
    return mp.tasks.vision.ImageSegmenter.create_from_options(options)


def photo_quality(rgb: np.ndarray) -> dict[str, float | str]:
    """Return a conservative, non-blocking image-quality assessment.

    A global Laplacian score is strongly affected by smooth skin, so it is a
    weak proxy for whether dark, visible hairs can be analysed.  It should
    warn users about confidence rather than reject ordinary phone photos.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    severe_reasons: list[str] = []
    caution_reasons: list[str] = []

    # Only near-black / near-white images are rejected.  Normal indoor phone
    # photos with a soft skin background fall into "partly analysable" instead.
    if brightness < 32:
        severe_reasons.append("画面严重偏暗")
    elif brightness < 50:
        caution_reasons.append("画面偏暗")
    elif brightness > 238:
        severe_reasons.append("画面严重过亮或过曝")
    elif brightness > 215:
        caution_reasons.append("画面偏亮")

    # The previous 45 threshold rejected many photos where individual dark
    # hairs were still visible.  Below 12 is reserved for genuinely defocused
    # images; 12-45 merely lowers confidence.
    if sharpness < 12:
        severe_reasons.append("毛发边缘明显失焦")
    elif sharpness < 45:
        caution_reasons.append("毛发边缘偏软，细小毛发可能漏检")

    if severe_reasons:
        status = "建议重拍"
        action = "；".join(severe_reasons) + "。建议靠近皮肤局部并重新对焦；仍可继续分析，但不建议据此做前后对比。"
    elif caution_reasons:
        status = "部分可分析"
        action = "；".join(caution_reasons) + "。可继续分析，结果作为初步参考；若要做前后对比，建议在稳定光线下复拍。"
    else:
        status = "可分析"
        action = "照片质量满足当前演示版的基础分析条件。"
    return {
        "状态": status,
        "提示": action,
        "亮度": round(brightness, 1),
        "清晰度": round(sharpness, 1),
    }


def person_mask(bgr: np.ndarray) -> np.ndarray:
    """PASCAL VOC label 15 is person. Fall back safely if the model is unavailable."""
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = load_person_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        categories = np.squeeze(result.category_mask.numpy_view())
        result_mask = np.where(categories == 15, 255, 0).astype(np.uint8)
        if 0.03 <= float((result_mask > 0).mean()) <= 0.97:
            return cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    except Exception:
        pass
    return np.full(bgr.shape[:2], 255, dtype=np.uint8)


def central_foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """Extract the central photographed limb/body from its surrounding scene.

    GrabCut is used only as a foreground prior.  It is especially useful when
    the semantic person model cannot distinguish a close-up limb from a warm
    floor; the component containing the image centre is retained.
    """
    h, w = bgr.shape[:2]
    if h < 40 or w < 40:
        return np.zeros((h, w), dtype=np.uint8)
    grabcut_mask = np.zeros((h, w), dtype=np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    rect = (max(1, int(w * 0.04)), max(1, int(h * 0.01)), max(2, int(w * 0.92)), max(2, int(h * 0.98)))
    try:
        cv2.grabCut(bgr, grabcut_mask, rect, background_model, foreground_model, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)
    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    centre_label = int(labels[h // 2, w // 2])
    if centre_label == 0 or count <= 1:
        return np.zeros((h, w), dtype=np.uint8)
    region = np.where(labels == centre_label, 255, 0).astype(np.uint8)
    fraction = float(stats[centre_label, cv2.CC_STAT_AREA]) / float(h * w)
    return region if 0.015 <= fraction <= 0.78 else np.zeros((h, w), dtype=np.uint8)


def central_skin_region(bgr: np.ndarray, human: np.ndarray | None = None) -> np.ndarray:
    """Select only the central body-skin component, never the photo background.

    A person segmentation model can occasionally fall back on a scene that is
    dominated by a limb.  We therefore use it only as a weak prior, then grow
    the region from the image centre using its local skin colour.  This keeps a
    wood floor or nearby object with a similar warm colour out of the region.
    """
    h, w = bgr.shape[:2]
    person = human if human is not None else person_mask(bgr)
    foreground = central_foreground_mask(bgr)
    if not np.any(foreground):
        return np.zeros((h, w), dtype=np.uint8)
    # Keep useful semantic-person boundaries when available, but never use the
    # all-white fallback in place of the foreground extraction.
    person_fraction = float(np.count_nonzero(person)) / float(person.size)
    if 0.03 <= person_fraction <= 0.90:
        overlap = cv2.bitwise_and(foreground, person)
        if np.count_nonzero(overlap) >= 0.55 * np.count_nonzero(foreground):
            foreground = overlap
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # A broad first pass only identifies plausible warm skin for selecting the
    # centre seed.  It is intentionally not used as the final body boundary.
    plausible = cv2.inRange(ycrcb, (35, 120, 65), (255, 205, 170))
    plausible &= cv2.inRange(hsv, (0, 8, 25), (35, 245, 255))
    plausible = cv2.bitwise_and(plausible, foreground)

    y0, y1 = int(h * 0.38), int(h * 0.62)
    x0, x1 = int(w * 0.38), int(w * 0.62)
    seed = plausible[y0:y1, x0:x1] > 0
    seed_values = ycrcb[y0:y1, x0:x1][seed]
    if seed_values.size < 100:
        # Do not silently expand to the entire image when no body can be
        # located. An empty result is handled as "please retake" downstream.
        return np.zeros((h, w), dtype=np.uint8)

    centre_colour = np.median(seed_values, axis=0)
    seed_chroma_distance = np.linalg.norm(seed_values[:, 1:3] - centre_colour[1:3], axis=1)
    chroma_tolerance = float(np.clip(np.percentile(seed_chroma_distance, 85) + 5, 8, 20))
    chroma_distance = np.linalg.norm(ycrcb[:, :, 1:3].astype(np.float32) - centre_colour[1:3], axis=2)
    luminance_gap = np.abs(ycrcb[:, :, 0].astype(np.float32) - float(centre_colour[0]))

    # Chroma locks onto the body seeded in the middle; the luminance guard
    # prevents a bright wooden floor from merging into the limb.
    colour = np.where((chroma_distance <= chroma_tolerance) & (luminance_gap <= 38), 255, 0).astype(np.uint8)
    colour = cv2.bitwise_and(colour, plausible)
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(colour, connectivity=8)
    if count <= 1:
        return np.zeros((h, w), dtype=np.uint8)
    centre = labels[y0:y1, x0:x1].ravel()
    centre = centre[centre > 0]
    if centre.size:
        label = int(np.bincount(centre).argmax())
    else:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = np.where(labels == label, 255, 0).astype(np.uint8)
    # Reject implausibly broad components rather than falling back to the
    # full photo. That is safer than treating a floor as body skin.
    area_fraction = float(np.count_nonzero(region)) / float(region.size)
    if not 0.015 <= area_fraction <= 0.72:
        return np.zeros((h, w), dtype=np.uint8)
    # Do not count the sharp body/background boundary as a hair.
    region = cv2.erode(region, np.ones((7, 7), np.uint8), iterations=1)
    return region


def visible_hair_mask(bgr: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Initial dark-hair mask used consistently for both comparison images.

    It measures visible dark, narrow structures relative to nearby skin. This
    is deliberately a baseline, not a trained hair-segmentation model.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    skin_values = blackhat[skin > 0]
    if skin_values.size < 200:
        return np.zeros_like(gray)
    # Otsu selects the break between ordinary skin texture and dark, narrow
    # structures.  Unlike a fixed percentile, it does not force a fixed share
    # of every photograph to become "hair" when the skin is almost hairless.
    otsu_threshold, _ = cv2.threshold(
        skin_values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = max(8.0, float(otsu_threshold))
    hair = np.where(blackhat >= threshold, 255, 0).astype(np.uint8)
    hair = cv2.bitwise_and(hair, skin)
    hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Remove isolated speckles but retain thin, interrupted visible strands.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    cleaned = np.zeros_like(hair)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == label] = 255

    # Black-hat edges are often several pixels wider than the visible hair.
    # Reducing each accepted strand to a one-pixel centreline avoids inflating
    # coverage merely because of blur, focus, or camera resolution.
    skeleton = np.zeros_like(cleaned)
    working = cleaned.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(32):
        if not np.any(working):
            break
        eroded = cv2.erode(working, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
        working = eroded
    return cv2.bitwise_and(skeleton, skin)


def analyse_photo(image: Image.Image) -> dict[str, object]:
    image = prepare_photo(image)
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    quality = photo_quality(rgb)
    human = person_mask(bgr)
    analysis_region = central_skin_region(bgr, human)
    hair = visible_hair_mask(bgr, analysis_region)
    analysis_pixels = int(np.count_nonzero(analysis_region))
    # Coverage is hair-pixel area / detected human skin analysis area. The
    # denominator is deliberately never the whole photograph or its background.
    coverage = (100.0 * np.count_nonzero(hair) / analysis_pixels) if analysis_pixels else 0.0

    if analysis_pixels < int(0.015 * analysis_region.size):
        quality = {
            **quality,
            "状态": "建议重拍",
            "提示": "未能稳定定位画面中央的人体皮肤区域。请只拍摄一小片皮肤、减少地板和物品背景后重拍。",
        }

    overlay = rgb.copy()
    overlay[hair > 0] = (245, 55, 145)
    contours, _ = cv2.findContours(analysis_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (65, 200, 115), 2)
    overlay = cv2.addWeighted(rgb, 0.72, overlay, 0.28, 0)
    return {
        "rgb": rgb,
        "overlay": overlay,
        "mask": hair,
        "quality": quality,
        "coverage": round(float(coverage), 2),
        "skin_fraction": round(float(analysis_pixels / analysis_region.size), 3),
        "analysis_region": analysis_region,
    }


def comparison_decision(before: dict[str, object], after: dict[str, object]) -> tuple[str, str]:
    before_quality = before["quality"]
    after_quality = after["quality"]
    assert isinstance(before_quality, dict) and isinstance(after_quality, dict)
    if before_quality["状态"] == "建议重拍" or after_quality["状态"] == "建议重拍":
        return "建议重拍", "至少一张照片不达标；不显示覆盖率变化，避免把模糊或曝光差异误当成毛发变化。"

    brightness_gap = abs(float(before_quality["亮度"]) - float(after_quality["亮度"]))
    before_skin = float(before["skin_fraction"])
    after_skin = float(after["skin_fraction"])
    skin_ratio = min(before_skin, after_skin) / max(0.001, max(before_skin, after_skin))
    if brightness_gap > 30 or skin_ratio < 0.65:
        return "建议重拍", "两张照片的光线或取景范围差异较大，无法可靠比较同一区域。请在相同距离、角度和光线下重拍。"
    if before_quality["状态"] == "部分可分析" or after_quality["状态"] == "部分可分析" or brightness_gap > 18:
        return "部分可比较", "可作初步参考，但细小毛发可能受清晰度或光线影响；建议按同一姿势复拍确认。"
    return "可比较", "两张照片满足当前演示版的基础可比性条件。"


def display_image(result: dict[str, object], mode: str) -> np.ndarray:
    if mode == "原图":
        return result["rgb"]  # type: ignore[return-value]
    if mode == "AI 标记图":
        return result["overlay"]  # type: ignore[return-value]
    return result["mask"]  # type: ignore[return-value]


st.markdown(
    """
    <style>
      .block-container { max-width: 760px; padding-top: 1.1rem; padding-bottom: 3rem; }
      [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] .main { background:#fff; }
      header[data-testid="stHeader"] { background: rgba(255,255,255,0); }
      .veet-nav { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee3e8; padding:0 0 1.05rem; margin-bottom:2.2rem; color:#211c29; }
      .veet-nav b { font-size:1.15rem; } .veet-nav span { color:#766a74; font-size:.95rem; }
      .pink-tag { display:inline-block; color:#d90068; background:#fff0f7; border-radius:9px; padding:.38rem .72rem; font-weight:700; }
      .flow-title { font-size:2.15rem; font-weight:800; line-height:1.18; color:#211c29; margin:.75rem 0 1.45rem; }
      .summary-card { border:1px solid #eadfe5; background:#fff; border-radius:22px; padding:1.05rem 1.2rem; margin:1rem 0 1.25rem; color:#322a36; }
      .result-card { background:#fff2f7; border-radius:25px; padding:1.55rem 1.7rem; margin:1.25rem 0; color:#211c29; }
      .analysis-card { background:#fff4f8; border-radius:25px; padding:1.45rem 1.7rem; margin:1.35rem 0; color:#211c29; }
      .analysis-card .label { color:#615869; font-size:1.03rem; } .analysis-card .value { color:#dc0069; font-size:3.7rem; line-height:1.08; font-weight:800; margin:.55rem 0 0; }
      .coverage-metrics { display:grid; grid-template-columns:repeat(3, 1fr); gap:.7rem; margin:1.2rem 0 1rem; }
      .coverage-metric { background:rgba(255,255,255,.67); border-radius:14px; padding:.8rem .65rem; }
      .coverage-metric b { display:block; color:#dc0069; font-size:1.42rem; margin-top:.28rem; }
      .coverage-metric small { color:#736a74; font-size:.84rem; }
      .result-card .change-line { color:#dc0069; font-size:1.05rem; font-weight:700; margin:.45rem 0; }
      .compare-card { background:#fff4f8; border-radius:25px; padding:1.65rem 1.7rem; margin:1.55rem 0; color:#211c29; }
      .compare-topline { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
      .compare-topline small { color:#6f6571; font-size:1rem; } .compare-topline b { display:block; font-size:2.75rem; line-height:1.2; margin-top:.32rem; }
      .compare-divider { border:0; border-top:1px solid #edcfdd; margin:1.5rem 0 1.35rem; }
      .compare-card h3 { font-size:1.25rem; margin:0; } .compare-card .percent { color:#dc0069; font-size:4.15rem; line-height:1.05; font-weight:800; margin:.65rem 0; }
      .soft-pink-notice { background:#fff0f5; border:1px solid #f5d6e3; color:#9d295d; border-radius:16px; padding:1rem 1.1rem; margin:1rem 0; line-height:1.6; }
      .result-card h3 { margin:0; font-size:1.25rem; } .result-card .percent { color:#dc0069; font-size:4.1rem; font-weight:800; line-height:1.15; margin:.65rem 0; }
      .microcopy { color:#736a74; line-height:1.65; }
      div.stButton > button[kind="primary"] { background:#df0069; border-color:#df0069; border-radius:16px; min-height:3.6rem; font-size:1.1rem; font-weight:700; }
      div.stButton > button[kind="secondary"] { color:#d70065; background:#fff0f7; border:0; border-radius:16px; min-height:3.2rem; }
      [data-testid="stFileUploader"] { border:1px dashed #e5b5cd; border-radius:18px; padding:.65rem; background:#fffafd; }
      [data-testid="stImage"] img { border-radius:21px; }
      @media (max-width: 640px) { .block-container { padding:1rem 1.05rem 2rem; } .flow-title { font-size:1.85rem; } .result-card .percent, .compare-card .percent { font-size:3.4rem; } .coverage-metrics { gap:.45rem; } .coverage-metric { padding:.7rem .45rem; } .coverage-metric b { font-size:1.16rem; } .compare-topline b { font-size:2.3rem; } }
    </style>
    <div class="veet-nav"><span>‹ 返回</span><b>Hair Coverage Assistant</b><span>••• ◉</span></div>
    """,
    unsafe_allow_html=True,
)


def quality_card(result: dict[str, object]) -> None:
    quality = result["quality"]
    assert isinstance(quality, dict)
    if quality["状态"] == "建议重拍":
        st.error(f"**建议重拍**：{quality['提示']}")
    elif quality["状态"] == "部分可分析":
        st.warning(f"**部分可分析**：{quality['提示']}")
    else:
        st.markdown(f'<div class="summary-card"><b>照片质量：良好</b><br><span class="microcopy">{quality["提示"]}</span></div>', unsafe_allow_html=True)


def clear_v2_record() -> None:
    for key in ("v2_before_result", "v2_after_result", "v2_comparison"):
        st.session_state.pop(key, None)
    st.session_state["v2_stage"] = "before"
    st.session_state["photo_record"] = st.session_state.get("photo_record", 0) + 1


def photo_source(label: str, stage_key: str):
    record = st.session_state.get("photo_record", 0)
    mode = st.radio(
        "获取照片方式", ["上传照片", "网页内拍照"], horizontal=True,
        key=f"photo_source_{stage_key}_{record}",
    )
    if mode == "网页内拍照":
        st.caption("允许浏览器使用摄像头，在网页内完成拍照。若无法开启摄像头，可切换为上传照片。")
        return st.camera_input(
            label.replace("上传", "拍摄"), resolution="1080p",
            key=f"camera_{stage_key}_{record}",
        )
    st.caption("手机端建议选择图库中的照片；需要现场拍摄时，请切换到“网页内拍照”。")
    return st.file_uploader(
        label, type=["jpg", "jpeg", "png"],
        key=f"upload_{stage_key}_{record}",
    )


stage = st.session_state.get("v2_stage", "before")
body_part = st.selectbox("护理部位", ["小腿", "手臂", "腋下", "比基尼线", "其他"], key="v2_part")

if stage == "before":
    st.markdown('<span class="pink-tag">局部照片</span><div class="flow-title">确认护理前照片</div>', unsafe_allow_html=True)
    st.markdown('<p class="microcopy">请拍摄皮肤局部近照。对焦在毛发上，避免滤镜、强反光和大面积背景。</p>', unsafe_allow_html=True)
    before_upload = photo_source("上传护理前照片", "before")
    if before_upload:
        try:
            before_image = read_uploaded_photo(before_upload)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        preview_quality = photo_quality(np.asarray(before_image.convert("RGB")))
        st.image(before_image, caption="护理前局部照片", width="stretch")
        st.markdown(
            f'<div class="summary-card"><b>护理部位：{body_part}</b><br>'
            f'<b>照片质量：{preview_quality["状态"]}</b></div>',
            unsafe_allow_html=True,
        )
        if preview_quality["状态"] == "建议重拍":
            st.error(preview_quality["提示"])
        elif preview_quality["状态"] == "部分可分析":
            st.warning(preview_quality["提示"])
        before_button = "仍然分析（结果可能不可靠）" if preview_quality["状态"] == "建议重拍" else "确认并分析"
        if st.button(before_button, type="primary", use_container_width=True):
            with st.spinner("正在识别护理前的可见毛发…"):
                st.session_state["v2_before_result"] = analyse_photo(before_image)
            st.session_state["v2_stage"] = "after"
            st.rerun()
    else:
        st.markdown('<div class="soft-pink-notice">先上传护理前照片，系统将先检查照片质量。</div>', unsafe_allow_html=True)

elif stage == "after":
    before_result = st.session_state["v2_before_result"]
    assert isinstance(before_result, dict)
    st.markdown('<span class="pink-tag">毛发分析</span><div class="flow-title">你的毛发分析</div>', unsafe_allow_html=True)
    before_quality = before_result["quality"]
    assert isinstance(before_quality, dict)
    st.image(before_result["overlay"], caption="AI 标记图｜粉色为可见毛发，绿色为人体皮肤分析区域", width="stretch")
    st.markdown(
        f'<div class="analysis-card"><div class="label">脱毛前可见毛发覆盖率</div>'
        f'<div class="value">{float(before_result["coverage"]):.2f}%</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="microcopy"><b>照片质量：</b>{before_quality["状态"]}</p>', unsafe_allow_html=True)
    st.markdown('<p class="microcopy">结果为图像估计，受光线、角度和毛发颜色影响，不用于皮肤诊断。覆盖率 = 识别到的粉色毛发面积 ÷ 绿色人体皮肤分析区域面积。</p>', unsafe_allow_html=True)
    if st.toggle("核对 AI 识别细节", key="v2_before_mask_toggle"):
        raw_before, marked_before, mask_before = st.columns(3)
        raw_before.image(before_result["rgb"], caption="原图", width="stretch")
        marked_before.image(before_result["overlay"], caption="AI 标记图", width="stretch")
        mask_before.image(before_result["mask"], caption="二值掩膜", width="stretch")

    st.markdown('<div class="flow-title" style="font-size:1.85rem; margin-top:2rem;">记录护理后照片</div>', unsafe_allow_html=True)
    st.markdown('<p class="microcopy">请尽量对准同一片皮肤，并保持相近距离、角度和光线。</p>', unsafe_allow_html=True)
    after_upload = photo_source("上传护理后照片", "after")
    if after_upload:
        try:
            after_image = read_uploaded_photo(after_upload)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        preview_quality = photo_quality(np.asarray(after_image.convert("RGB")))
        st.image(after_image, caption="护理后局部照片", width="stretch")
        quality_text = f"护理部位：{body_part}<br><b>照片质量：{preview_quality['状态']}</b>"
        st.markdown(f'<div class="summary-card">{quality_text}</div>', unsafe_allow_html=True)
        if preview_quality["状态"] == "建议重拍":
            st.error(preview_quality["提示"])
        elif preview_quality["状态"] == "部分可分析":
            st.warning(preview_quality["提示"])
        after_button = "仍然对比（结果可能不可靠）" if preview_quality["状态"] == "建议重拍" else "确认并对比"
        if st.button(after_button, type="primary", use_container_width=True):
            with st.spinner("正在核对两张照片并计算可见毛发覆盖率变化…"):
                after_result = analyse_photo(after_image)
                status, explanation = comparison_decision(before_result, after_result)
            st.session_state["v2_after_result"] = after_result
            st.session_state["v2_comparison"] = {"status": status, "explanation": explanation}
            st.session_state["v2_stage"] = "result"
            st.rerun()
    else:
        st.markdown('<div class="soft-pink-notice">上传护理后照片后，系统才会进行前后覆盖率比较。</div>', unsafe_allow_html=True)
    if st.button("重新开始本次记录", use_container_width=True):
        clear_v2_record()
        st.rerun()

else:
    before_result = st.session_state["v2_before_result"]
    after_result = st.session_state["v2_after_result"]
    comparison = st.session_state["v2_comparison"]
    assert isinstance(before_result, dict) and isinstance(after_result, dict) and isinstance(comparison, dict)
    status = str(comparison["status"])
    explanation = str(comparison["explanation"])
    st.markdown('<span class="pink-tag">护理反馈</span><div class="flow-title">记录本次护理</div>', unsafe_allow_html=True)

    if status == "建议重拍":
        st.error(f"**建议重拍**：{explanation}")
    else:
        if status == "部分可比较":
            st.warning(f"**部分可比较**：{explanation}")
        before_coverage = float(before_result["coverage"])
        after_coverage = float(after_result["coverage"])
        relative_drop = ((before_coverage - after_coverage) / before_coverage * 100) if before_coverage >= 0.25 else None
        previews_left, previews_right = st.columns(2)
        previews_left.image(before_result["rgb"], caption="护理前", width="stretch")
        previews_right.image(after_result["rgb"], caption="护理后", width="stretch")
        if status == "可比较" and relative_drop is not None:
            point_change = before_coverage - after_coverage
            change_title = "可见毛发覆盖减少" if point_change >= 0 else "可见毛发覆盖增加"
            point_label = "减少" if point_change >= 0 else "增加"
            relative_label = "相对减少" if point_change >= 0 else "相对增加"
            st.markdown(
                '<div class="compare-card"><div class="compare-topline">'
                f'<div><small>脱毛前覆盖率</small><b>{before_coverage:.2f}%</b></div>'
                f'<div><small>脱毛后覆盖率</small><b>{after_coverage:.2f}%</b></div>'
                '</div><hr class="compare-divider">'
                f'<h3>{change_title}</h3><div class="percent">{abs(relative_drop):.0f}%</div>'
                f'<div class="microcopy">覆盖率由 {before_coverage:.2f}% 变为 {after_coverage:.2f}%，{point_label} {abs(point_change):.2f} 个百分点。</div>'
                '<div class="microcopy" style="margin-top:.75rem;">仅反映同一区域可见毛发的图像变化。</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("前图覆盖率过低或照片仅部分可比较，因此不显示百分比变化结论。")

    st.subheader("核对 AI 识别")
    view_mode = st.radio("查看方式", ["原图", "AI 标记图", "二值掩膜"], horizontal=True)
    checked_before, checked_after = st.columns(2)
    checked_before.image(display_image(before_result, view_mode), caption=f"护理前｜{view_mode}", width="stretch")
    checked_after.image(display_image(after_result, view_mode), caption=f"护理后｜{view_mode}", width="stretch")

    with st.expander("开发调试信息", expanded=False):
        st.json({"前图质量": before_result["quality"], "后图质量": after_result["quality"], "比较状态": status})
    if st.button("开始新的护理记录", type="primary", use_container_width=True):
        clear_v2_record()
        st.rerun()
