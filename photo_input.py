"""Bound memory use before quality assessment, display and segmentation."""
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_SIDE = 2048
MAX_SOURCE_PIXELS = 80_000_000


def prepare_photo(image: Image.Image) -> Image.Image:
    if image.width * image.height > MAX_SOURCE_PIXELS:
        raise ValueError("照片分辨率过大，请使用普通拍照模式，关闭超高清模式后重试。")
    # JPEG draft decoding avoids a full-resolution RGB allocation when possible.
    image.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def read_uploaded_photo(upload) -> Image.Image:
    try:
        upload.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(upload) as image:
                return prepare_photo(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("无法读取这张照片，请使用 JPG 或 PNG 格式，或将照片存入图库后重新上传。") from exc
