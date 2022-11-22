#!/bin/env python3
from rknn.api import RKNN
import os

if __name__ == '__main__':
    platform = 'rk3588'
    exp = 'yolov5s'
    Width = 640
    Height = 640
    MODEL_PATH = './onnx_models/yolov5s_rm_transpose.onnx'

    # Create RKNN object
    rknn = RKNN()

    OUT_DIR = "rknn_models"
    RKNN_MODEL_PATH = './{}/{}_rm_transpose.rknn'.format(
        OUT_DIR, exp + '-' + str(Width) + '-' + str(Height))
    DATASET = './dataset.txt'

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform)
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(MODEL_PATH)
    if ret != 0:
        print('load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret != 0:
        print('build model failed.')
        exit(ret)
    print('done')

    # Export rknn model
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export rknn model failed.')
        exit(ret)
    print('done')

    rknn.release()

