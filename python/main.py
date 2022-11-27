import os
import cv2
import platform
import numpy as np

# TODO: Remove all hard-coded path references

# TODO: Refactor these hard-coded constants below
#       to arguments of this script
VIDEO_DEV_NUM = 1
TARGET_PLATFORM = 'rk3588'
OUTPUT_FILE = './output.png'


def get_frame(cap: cv2.VideoCapture):
    state, frame = cap.read()
    if state:
        # TODO: Implement frame cropping, to maintain aspect ratio
        frame = cv2.resize(frame, (320, 320))
        # NanoDet takes BGR format pixel as input
        # and OpenCV use BGR internally
        # so no color format convert needed
        ### return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) ###
        return frame
    raise Exception('Capture FAILED!')


def run_lite():
    from rknnlite.api import RKNNLite
    rknn_lite = RKNNLite()
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(path='../model/nanodet-plus-m_320.rknn')
    if ret != 0:
        raise Exception('RKNN Model load FAILED')
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    if ret != 0:
        raise Exception('RKNN Runtime init FAILED')
    # print('--> Load Image file')
    # img = cv2.imread('./dataset/test_img.jpg')
    # img = cv2.resize(img, (320, 320))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('--> Init camera')
    cap = cv2.VideoCapture(VIDEO_DEV_NUM)
    # TODO: Set attributes of `cap`
    # Like resolution, cap format, exposure time, etc.
    while True:
        img = get_frame(cap)
        output = rknn_lite.inference(inputs=[img])
        # TODO: Add output parsing
        # STAT: Currently unaware of how to deal with the nn output
        cv2.imwrite(OUTPUT_FILE, img)
        print(type(output), output)
    rknn_lite.release()
    # while True:


def run():
    from rknn.api import RKNN
    rknn = RKNN()
    print('--> Config model')
    rknn.config(target_platform=TARGET_PLATFORM,
                mean_values=[[103.53, 116.28, 123.657]],
                std_values=[[57.375638304, 57.1200091392, 58.3941605839]],
                # inputs_yuv_fmt=['nv12'],
                compress_weight=True,
                single_core_mode=True)
    print('--> Loading model')
    try:
        ret = rknn.load_rknn('../model/nanodet-plus-m_320.rknn')
        if ret != 0:
            raise Exception('Load RKNN model FAILED!')
    except:
        rknn.load_onnx(model='../model/nanodet-plus-m_320.onnx')
        print('--> Building ONNX model')
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise Exception('Build ONNX model FAILED!')
        print('--> Export to RKNN model')
        ret = rknn.export_rknn('../model/nanodet-plus-m_320.rknn')
        if ret != 0:
            raise Exception('Export RKNN model FAILED!')

    if ret != 0:
        raise Exception('Build model FAILED!')
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=TARGET_PLATFORM)
    if ret != 0:
        raise Exception('RKNN Runtime init FAILED!')
    img = cv2.imread('../dataset/test_img.jpg')
    img = cv2.resize(img, (320, 320))
    # NanoDet takes BGR format pixel as input
    # and OpenCV use BGR internally
    # so no color format convert needed
    ### img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ###
    output = rknn.inference(inputs=[img])
    print(type(output), output)
    rknn.release()


if __name__ == '__main__':
    if platform.machine() == 'aarch64':
        run_lite()
    else:
        run()
