# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
from pathlib import Path
from pathlib import PosixPath

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import queue
import select
import re
import os
import torch
from functools import partial
import threading
import sys

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES, DATA
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils.files import increment_path
from ultralytics.data.utils import VID_FORMATS

from tracking.detectors import get_yolo_inferer
from tracking.utils import convert_to_mot_format, tracks_add_emb_convert_to_mot_format, write_mot_results,write_mot_results_ADDED, download_mot_eval_tools, download_mot_dataset, unzip_mot_dataset, eval_setup
from boxmot.appearance.reid_auto_backend import ReidAutoBackend

# checker = RequirementsChecker()
# checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


def prompt_overwrite(path_type: str, path: str, ci: bool = False) -> bool:
    """
    Prompts the user to confirm overwriting an existing file.

    Args:
        path_type (str): Type of the path (e.g., 'Detections and Embeddings', 'MOT Result').
        path (str): The path to check.
        ci (bool): If True, automatically reuse existing file without prompting (for CI environments).

    Returns:
        bool: True if user confirms to overwrite, False otherwise.
    """
    if ci:
        print(f"{path_type} {path} already exists. Use existing due to no UI mode.")
        return False

    def input_with_timeout(prompt, timeout=3.0):
        print(prompt, end='', flush=True)
        
        result = []
        input_received = threading.Event()

        def get_input():
            user_input = sys.stdin.readline().strip().lower()
            result.append(user_input)
            input_received.set()

        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True  # Ensure thread does not prevent program exit
        input_thread.start()
        input_thread.join(timeout)

        if input_received.is_set():
            return result[0] in ['y', 'yes']
        else:
            print("\nNo response, not proceeding with overwrite...")
            return False

    return input_with_timeout(f"{path_type} {path} already exists. Overwrite? [y/N]: ")



def generate_dets_embs(args: argparse.Namespace, y: Path) -> None:
    """
    Generates detections and embeddings for the specified YOLO model and arguments.

    Args:
        args (Namespace): Parsed command line arguments.
        y (Path): Path to the YOLO model file.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    
    ul_models = ['yolov8', 'yolov9', 'yolov10', 'rtdetr', 'sam']

    yolo = YOLO(y if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt')
    
    results = yolo(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        stream=True,
        device=args.device,
        verbose=False,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )

    if not any(yolo in str(args.yolo_model) for yolo in ul_models):
        m = get_yolo_inferer(y)
        model = m(model=y, device=yolo.predictor.device, args=yolo.predictor.args)
        yolo.predictor.model = model

    cow_embeddings = {} # 初始化方向性字典

    reids = []
    for r in args.reid_model:
        model = ReidAutoBackend(weights=args.reid_model, device=yolo.predictor.device, half=args.half).model
        reids.append(model)
        embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / r.stem / (Path(args.source).parent.name + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)

        if os.path.getsize(embs_path) > 0:
            open(embs_path, 'w').close()

    yolo.predictor.custom_args = args

    dets_path = args.project / 'dets_n_embs' / y.stem / 'dets' / (Path(args.source).parent.name + '.txt')
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)

    if os.path.getsize(dets_path) > 0:
        open(dets_path, 'w').close()

    # with open(str(dets_path), 'ab+', ) as f:
    #     np.savetxt(f, [], fmt='%f', header=str(args.source))
    with open(str(dets_path), 'a+', encoding="utf-8") as f:
        np.savetxt(f, [], fmt='%f', header=str(args.source))

    for frame_idx, r in enumerate(tqdm(results, desc="Frames")):
        nr_dets = len(r.boxes)
        frame_idx = torch.full((1, 1), frame_idx + 1).repeat(nr_dets, 1)

        if r.boxes.data.is_cpu:
            dets = r.boxes.data[:, 0:4].numpy()
        else:
            dets = r.boxes.data[:, 0:4].cpu().numpy()

        img = r.orig_img

        dets = np.concatenate(
            [
                frame_idx,
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
            ], axis=1
        )

        with open(str(dets_path), 'ab+') as f:
            np.savetxt(f, dets, fmt='%f')

        for reid, reid_model_name in zip(reids, args.reid_model):
            embs = reid.get_features(dets[:, 1:5], img)
            embs_path = args.project / "dets_n_embs" / y.stem / 'embs' / reid_model_name.stem / (Path(args.source).parent.name + '.txt')
            with open(str(embs_path), 'ab+') as f:
                np.savetxt(f, embs, fmt='%f')


def generate_mot_results(args: argparse.Namespace, config_dict: dict = None) -> None:
    """
    Generates MOT results for the specified arguments and configuration.

    Args:
        args (Namespace): Parsed command line arguments.
        config_dict (dict, optional): Additional configuration dictionary.
    """
    args.device = select_device(args.device)
    tracker = create_tracker(
        args.tracking_method,
        TRACKER_CONFIGS / (args.tracking_method + '.yaml'),    # 跟踪器的配置文件
        args.reid_model[0].with_suffix('.pt'),    # 加载一个ReID模型
        args.device,
        False,
        False,
        config_dict
    )

    #  打开检测结果，读取第一行的内容，即“数据源的路径”
    with open(args.dets_file_path, 'r') as file:
        args.source = file.readline().strip().replace("# ", "")

    # 输出日志信息
    LOGGER.info(f"\nStarting tracking on:\n\t{args.source}\nwith preloaded dets\n\t({args.dets_file_path.relative_to(ROOT)})\nand embs\n\t({args.embs_file_path.relative_to(ROOT)})\nusing\n\t{args.tracking_method}")

    # 读取检测结果和嵌入特征数据：
    dets = np.loadtxt(args.dets_file_path, skiprows=1)
    embs = np.loadtxt(args.embs_file_path)
    # 将检测结果和嵌入向量拼接在一起
    dets_n_embs = np.concatenate([dets, embs], axis=1)

    # 加载视频或者图片数据
    dataset = LoadImagesAndVideos(args.source)

    # 创建一个路径，用于保存MOT结果的文本文件
    # 创建空列表，  用于保存所有帧的跟踪结果
    txt_path = args.exp_folder_path / (Path(args.source).parent.name + '.txt')
    txt_path_ADDED = args.exp_folder_path / (Path(args.source).parent.name + '_emb_added.txt')
    all_mot_results = []
    all_mot_results_added_emb = []

    for frame_idx, d in enumerate(tqdm(dataset, desc="Frames")):
        if frame_idx == len(dataset):
            break # 如果当前帧是最后一帧，跳出循环

        im = d[1][0]
        frame_dets_n_embs = dets_n_embs[dets_n_embs[:, 0] == frame_idx + 1] # 获取当前帧（frame_idx + 1）的dets_n_embs 作为frame_dets_n_embs

        dets = frame_dets_n_embs[:, 1:7] # 当前帧检测到的(多个)目标框 位置
        embs = frame_dets_n_embs[:, 7:]  # 当前帧检测到的(多个)目标框 对应的emb
        tracks, tracks_add_emb = tracker.update(dets, im, embs) #### # 使用跟踪器的update方法更新跟踪结果，并返回跟踪轨迹“tracks”

        # 如果有轨迹，调用函数将轨迹转为MOT的格式，并将结果添加到all_mot_results列表中。
        if tracks.size > 0:
            mot_results = convert_to_mot_format(tracks, frame_idx + 1)
            all_mot_results.append(mot_results)

        # 添加。如果有轨迹，将结果添加到  all_mot_results_added_emb 列表中
        if tracks_add_emb.size > 0:
            # mot_results_emb_added = convert_to_mot_format(tracks_add_emb, frame_idx + 1)
            mot_results_emb_added = tracks_add_emb_convert_to_mot_format(tracks_add_emb, frame_idx + 1)
            all_mot_results_added_emb.append(mot_results_emb_added)

    # 遍历完成之后将所有帧的结果堆叠，并将结果写在文本文件中
    if all_mot_results:
        all_mot_results = np.vstack(all_mot_results)
        write_mot_results(txt_path, all_mot_results)

    if all_mot_results_added_emb: # xxs添加用于保存带特征嵌入的结果文件
        all_mot_results_added_emb = np.vstack(all_mot_results_added_emb)
        write_mot_results_ADDED(txt_path_ADDED, all_mot_results_added_emb)##


def parse_mot_results(results: str) -> dict:
    """
    Extracts the COMBINED HOTA, MOTA, IDF1 from the results generated by the run_mot_challenge.py script.

    Args:
        results (str): MOT results as a string.

    Returns:
        dict: A dictionary containing HOTA, MOTA, and IDF1 scores.
    """
    combined_results = results.split('COMBINED')[2:-1]
    combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]

    results_dict = {}
    for key, value in zip(["HOTA", "MOTA", "IDF1"], combined_results):
        results_dict[key] = value

    return results_dict


def trackeval(args: argparse.Namespace, seq_paths: list, save_dir: Path, MOT_results_folder: Path, gt_folder: Path, metrics: list = ["HOTA", "CLEAR", "Identity"]) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        MOT_results_folder (Path): Folder containing MOT results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """
    d = [seq_path.parent.name for seq_path in seq_paths]

    args = [
        sys.executable, EXAMPLES / 'val_utils' / 'scripts' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", "",
        "--TRACKERS_FOLDER", args.exp_folder_path,
        "--TRACKERS_TO_EVAL", "",
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--SEQ_INFO", *d
    ]

    p = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        print("Standard Error:\n", stderr)
    return stdout


def run_generate_dets_embs(opt: argparse.Namespace) -> None:
    """
    Runs the generate_dets_embs function for all YOLO models and source directories.

    Args:
        opt (Namespace): Parsed command line arguments.
    """

    mot_folder_paths = [item for item in Path(opt.source).iterdir()]

    # 如果你希望 opt.yolo_model 是一个列表，而不是单个路径
    if isinstance(opt.yolo_model, PosixPath):
        print("######")
        opt.yolo_model = [opt.yolo_model]  # 将其转换为单个元素的列表
        print("*****")
    print("opt.yolo_model is {}".format(opt.yolo_model))
    for y in opt.yolo_model:
        print(f'Processing model at: {y}')
        for i, mot_folder_path in enumerate(mot_folder_paths):

            dets_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets' / (mot_folder_path.name + '.txt')

            # 确保 opt.reid_model 是一个可下标的列表，如果是 PosixPath 则转换成列表
            if isinstance(opt.reid_model, PosixPath):
                opt.reid_model = [opt.reid_model]  # 转换为包含单个元素的列表
            embs_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / (opt.reid_model[0].stem) / (mot_folder_path.name + '.txt')
            if dets_path.exists() and embs_path.exists():
                if prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    LOGGER.info(f'Overwriting detections and embeddings for {mot_folder_path}...')
                else:
                    LOGGER.info(f'Skipping generation for {mot_folder_path} as they already exist.')
                    continue
            LOGGER.info(f'Generating detections and embeddings for data under {mot_folder_path} [{i + 1}/{len(mot_folder_paths)} seqs]')
            opt.source = mot_folder_path / 'img1'
            generate_dets_embs(opt, y)


def run_generate_mot_results(opt: argparse.Namespace, evolve_config: dict = None) -> None:
    """
    Runs the generate_mot_results function for all YOLO models and detection/embedding files.

    Args:
        opt (Namespace): Parsed command line arguments.
        evolve_config (dict, optional): Additional configuration dictionary.
    """
    for y in opt.yolo_model:
        exp_folder_path = Path(opt.project)/'mot'/(y.stem + "_" + opt.reid_model[0].stem + "_" + opt.tracking_method)

        # """"添加"""
        # if isinstance(exp_folder_path, PosixPath):
        #     print("######")
        #     exp_folder_path = [exp_folder_path]  # 将其转换为单个元素的列表
        #     print("*****")
        # """添加结束"""
        # exp_folder_path = opt.project / 'mot' / (str(y.stem) + "_" + str(opt.reid_model[0].stem) + "_" + str(opt.tracking_method))
        exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=False)
        opt.exp_folder_path = exp_folder_path
        dets_file_paths = [item for item in (opt.project / "dets_n_embs" / y.stem / 'dets').glob('*.txt') if not item.name.startswith('.')]
        embs_file_paths = [item for item in (opt.project / "dets_n_embs" / y.stem / 'embs' / opt.reid_model[0].stem).glob('*.txt') if not item.name.startswith('.')]
        for d, e in zip(dets_file_paths, embs_file_paths):
            mot_result_path = exp_folder_path / (d.stem + '.txt')
            if mot_result_path.exists():
                if prompt_overwrite('MOT Result', mot_result_path, opt.ci):
                    LOGGER.info(f'Overwriting MOT result for {d.stem}...')
                else:
                    LOGGER.info(f'Skipping MOT result generation for {d.stem} as it already exists.')
                    continue
            opt.dets_file_path = d
            opt.embs_file_path = e
            generate_mot_results(opt, evolve_config)


def run_trackeval(opt: argparse.Namespace) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, opt.val_tools_path)
    trackeval_results = trackeval(opt, seq_paths, save_dir, MOT_results_folder, gt_folder)
    hota_mota_idf1 = parse_mot_results(trackeval_results)
    if opt.verbose:
        print(trackeval_results)
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(hota_mota_idf1))
    print(json.dumps(hota_mota_idf1))
    return hota_mota_idf1


def run_all(opt: argparse.Namespace) -> None:
    """
    Runs all stages of the pipeline: generate_dets_embs, generate_mot_results, and trackeval.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    run_generate_dets_embs(opt)
    run_generate_mot_results(opt)
    run_trackeval(opt)
    

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Global arguments
    parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8nQ.pt', help='yolo model path')
    parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_selfCowData.pth', help='reid model path')
    parser.add_argument('--source', type=str, default="/media/v10016//boxmot-master/tracking/val_utils/data/MOT17/train",help='file/dir/URL/glob, 0 for webcam')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runsCOWLunwenCow_onlyFB_Direction_', type=Path, help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--ci', action='store_true', help='Automatically reuse existing due to no UI in CI')

    parser.add_argument('--tracking-method', type=str, default='botsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--dets-file-path', type=Path, help='path to detections file')
    parser.add_argument('--embs-file-path', type=Path, help='path to embeddings file')
    parser.add_argument('--exp-folder-path', type=Path, help='path to experiment folder')
    parser.add_argument('--benchmark', type=str, default='MOT17', help='MOT16, MOT17, MOT20, MOT17-mini, or:<your-custom-dataset-cow>')
    parser.add_argument('--split', type=str, default='train', help='existing project/name ok, do not increment')

    parser.add_argument('--verbose', default=True, action='store_true', help='print results')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--n-trials', type=int, default=4, help='nr of trials for evolution')
    parser.add_argument('--objectives', type=str, nargs='+', default=["HOTA", "MOTA", "IDF1"], help='set of objective metrics: HOTA,MOTA,IDF1')
    parser.add_argument('--val-tools-path', type=Path, default=EXAMPLES / 'val_utils', help='path to store trackeval repo in')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for generate_dets_embs
    generate_dets_embs_parser = subparsers.add_parser('generate_dets_embs', help='Generate detections and embeddings')
    generate_dets_embs_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    generate_dets_embs_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_dets_embs_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_dets_embs_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    # Subparser for generate_mot_results
    generate_mot_results_parser = subparsers.add_parser('generate_mot_results', help='Generate MOT results')
    generate_mot_results_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_mot_results_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_mot_results_parser.add_argument('--tracking-method', type=str, default='botsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    generate_mot_results_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    # Subparser for trackeval
    trackeval_parser = subparsers.add_parser('trackeval', help='Evaluate tracking results')
    trackeval_parser.add_argument('--exp-folder-path', type=Path, required=True, help='path to experiment folder')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    # download MOT benchmark
    download_mot_eval_tools(opt.val_tools_path)
    zip_path = download_mot_dataset(opt.val_tools_path, opt.benchmark)
    unzip_mot_dataset(zip_path, opt.val_tools_path, opt.benchmark)

    if opt.command == 'generate_dets_embs':
        print("run_generate_dets_embs")
        run_generate_dets_embs(opt)
    elif opt.command == 'generate_mot_results':
        print("run_generate_mot_results")
        run_generate_mot_results(opt)
    elif opt.command == 'trackeval':
        run_trackeval(opt)
    else:
        run_all(opt)
        