# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS # TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc']
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer

# checker = RequirementsChecker()
# checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

# 在预测开始时初始化跟踪器
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for. # 要初始trackers的predictor
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False. # 是否持续保存已经存在的跟踪器
    """
    print('\n on_predict_start----\n')
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')   # ROOT/boxmot/configs/XXX.ymal
    trackers = [] # 跟踪器列表
    for i in range(predictor.dataset.bs):  #    # predictor.dataset.bs default is 1  ?
        print('predictor.dataset.bs is {}\n'.format(predictor.dataset.bs))
        # 根据batchSize，创建跟踪器
        # reate_tracker(tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None)
        tracker = create_tracker(
            predictor.custom_args.tracking_method,  # tracker_type,定义了实例化的哪一种跟踪器 如StrongSORT、Deepsort
            tracking_config,  # 设置各个跟踪器的超参数
            predictor.custom_args.reid_model,  # 对应函数接收参数：reid_weights
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        # 检查 tracker 对象是否具有 model 属性（即跟踪器中是否包含模型），如果有，则调用 tracker.model.warmup() 对模型进行热身，这通常是为了在实际使用前将模型加载到 GPU 上，以减少推理时的延迟。
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    
    ul_models = ['yolov8', 'yolov9', 'yolov10', 'rtdetr', 'sam']

    yolo = YOLO(
        args.yolo_model if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt',
    )



    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,    # default is True
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    # 向模型中添加回调函数。回调函数会在特定的事件或时机被触发，比如在预测开始时、训练结束时等。
    # 'on_predict_start'是回调函数触发的时机，即在预测开始时执行。此时，YOLO 模型还没有真正开始处理输入数据，只是准备好进行预测。
    # [partial对原函数的某些参数进行预设，返回一个新的函数。]  即：每次触发 on_predict_start 事件时，都会调用这个部分应用的 on_predict_start 函数，并且 persist 参数的值始终为 True。
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not any(yolo in str(args.yolo_model) for yolo in ul_models):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    print("self.predictor is {}:".format(yolo.predictor))

    # 对结果进行一些处理
    print(results)
    print("遍历results\n")
    for i,r in enumerate(results):    # 每帧产生一个results
        print("遍历第{}个".format(i))
        #把结果()画在图像上。
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        # img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)

        # if args.show is True:
        #     print("args.show is True")
        #     cv2.imshow('BoxMOT', img)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord(' ') or key == ord('q'):
        #         break
        if args.show is True:
            print("args.show is True")

            # 设置目标分辨率，例如 1280x720
            width, height = 1080, 640
            img_resized = cv2.resize(img, (width, height))  # 调整分辨率

            cv2.imshow('BoxMOT', img_resized)  # 显示调整后的图像
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break


def parse_opt():
    parser = argparse.ArgumentParser()

    # 检测模型及其权重文件
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n_QCow.pt', help='yolo model path')

    # ReID模型及其权重文件
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_selfCowData.pth', help='reid model path')

    # 跟踪方法（eepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc）
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    # 跟踪数据 文件位置
    parser.add_argument('--source', type=str, default=r'/media/v10016/实验室备份/XingshiXu/boxmot-master/CowMulDataFPS5/20240425_10_01_D3_fps5.mp4', help='file/dir/URL/glob, 0 for webcam')



    # 图像尺寸大小
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')

    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否视频show
    parser.add_argument('--show', default=True, action='store_true',
                        help='display tracking video results')
    # 是否保存跟踪结果
    parser.add_argument('--save', default=True, action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')

    ##### 结果保存 #####
    parser.add_argument('--project', default=ROOT / 'runs' / 'track_LunWen',
                        help='save results to project/name')
    parser.add_argument('--name', default='exppPP',
                        help='save results to project/name')


    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', default=False, action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', default=True,action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
