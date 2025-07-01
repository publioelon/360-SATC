import numpy as np
import cv2
import time
from turbojpeg import TurboJPEG, TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJFLAG_PROGRESSIVE

jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")

# Use a larger image for a more realistic benchmark
imgs = [(np.random.rand(2160,3840,3)*255).astype(np.uint8) for _ in range(100)]

def opencv_encode(img):
    return cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]

def turbojpeg_encode(img):
    # Use all available speed flags
    flags = TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT | TJFLAG_PROGRESSIVE
    return jpeg.encode(img, quality=90, flags=flags)

t0 = time.time()
for img in imgs:
    opencv_encode(img)
opencv_batch = (time.time() - t0) * 1000

t1 = time.time()
for img in imgs:
    turbojpeg_encode(img)
turbojpeg_batch = (time.time() - t1) * 1000

print(f"OpenCV batch encode:   {opencv_batch:.2f} ms")
print(f"PyTurboJPEG batch encode: {turbojpeg_batch:.2f} ms")
