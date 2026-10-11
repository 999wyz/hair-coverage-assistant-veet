"""Check high-resolution camera uploads, EXIF rotation and invalid files."""
from io import BytesIO

from PIL import Image

from photo_input import read_uploaded_photo

photo = Image.new("RGB", (4000, 3000), (170, 130, 110))
exif = Image.Exif()
exif[274] = 6
uploaded = BytesIO()
photo.save(uploaded, format="JPEG", exif=exif)
result = read_uploaded_photo(uploaded)
assert result.mode == "RGB"
assert max(result.size) <= 2048
assert result.height > result.width, "Camera EXIF orientation not applied"
assert result.getexif().get(274) is None
print(f"PASS 12MP camera JPEG, corrected orientation: {result.size}")

small = Image.new("RGBA", (200, 100), (100, 100, 100, 255))
uploaded = BytesIO()
small.save(uploaded, format="PNG")
result = read_uploaded_photo(uploaded)
assert result.size == (200, 100) and result.mode == "RGB"
print("PASS small image keeps resolution")

try:
    read_uploaded_photo(BytesIO(b"incomplete camera upload"))
except ValueError:
    print("PASS invalid upload gives a controlled error")
else:
    raise AssertionError("Invalid upload accepted")
