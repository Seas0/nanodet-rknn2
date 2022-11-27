# [NanoDet](https://github.com/RangiLyu/nanodet) on [RKNN2](https://github.com/rockchip-linux/rknpu2)

## Python Version

- Based on [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) and [rknn-toolkit-lite2](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn_toolkit_lite2)
- use OpenCV for image capture & process
  - Capture image `// TODO: Set attributes of capture stream`
  - Resize it to (320, 320) and convert to RGB
  - Feed converted image to RKNN, get result of inference
  - Parse result, draw box as overlay of origin image `// TODO`
  - Async / Multithread support maybe? `//TODO`

## CPP Version `// TODO`

- Based on [rknn-rt2](https://github.com/rockchip-linux/rknpu2/tree/master/runtime)
- Use V4L2 / GStreamer / librga / mpp...?
  - Maybe capture YUYV, then convert YUYV to NV12?
- Other, TBD...
