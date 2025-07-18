# 360-SATC Saliency-Aware-Tile-Compression-for-Real-Time-360 Video-Streaming-at-the-Edge

<img src="https://github.com/publioelon/360-SATC/blob/main/proposed_v3.svg">

# Abstract

The increasing adoption of 360° video and extended reality applications is driving new requirements for Quality of Experience in immersive streaming. This work presents 360-SATC, a real-time framework deployed at the network edge that leverages optical flow and saliency prediction to identify frame regions most likely to attract user attention. The framework operates server-side without reliance on continuous user feedback or explicit viewport prediction. Saliency maps guide adaptive per-tile compression, allowing the framework to allocate higher quality to the most relevant areas of each frame while compressing other regions more aggressively in accordance with current network conditions. Evaluation demonstrates that 360-SATC substantially reduces transmitted data under both low- and high-bandwidth scenarios, processes video at 65 frames per second, and supports display rates up to 115 frames per second. Measured perceptual quality remains high, with average PSNR, SSIM, and LPIPS of 36.9 dB, 0.98, and 0.043 in low-bandwidth mode, and 44.5 dB, 0.99, and 0.013 in high-bandwidth mode. Results indicate that 360-SATC achieves higher throughput and comparable perceptual quality relative to baseline tile-based compression methods.

For our experiments, we used the following:
```
numpy==2.1.2
opencv-python==4.11.0.86
tqdm==4.67.1
torch==2.6.0+cu126
onnx==1.17.0
onnxruntime==1.22.0
onnxruntime-gpu==1.20.1
pillow==11.0.0
scipy==1.15.2
lpips==0.1.4
```
For our GPU, we used CUDA version `12.6` and cuDNN version `8`. 





