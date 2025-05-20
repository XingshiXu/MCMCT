# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from typing import Union
from pathlib import Path
import os
import sys
import git
import requests
import zipfile
import subprocess
from git import Repo, exc
from boxmot.utils import logger as LOGGER
from tqdm import tqdm
from boxmot.utils import EXAMPLES, ROOT


def download_mot_eval_tools(val_tools_path):
    """
    Download the official evaluation tools for MOT metrics from the GitHub repository.
    
    Parameters:
        val_tools_path (Path): Path to the destination folder where the evaluation tools will be downloaded.
    
    Returns:
        None. Clones the evaluation tools repository and updates deprecated numpy types.
    """
    val_tools_url = "https://github.com/JonathonLuiten/TrackEval"

    try:
        # Clone the repository
        Repo.clone_from(val_tools_url, val_tools_path)
        LOGGER.info('Official MOT evaluation repo downloaded successfully.')
    except exc.GitError as err:
        LOGGER.info(f'Evaluation repo already downloaded or an error occurred: {err}')

    # Fix deprecated np.float, np.int & np.bool by replacing them with native Python types
    deprecated_types = {'np.float': 'float', 'np.int': 'int', 'np.bool': 'bool'}
    
    for file_path in val_tools_path.rglob('*'):
        if file_path.suffix in {'.py', '.txt'}:  # only consider .py and .txt files
            try:
                content = file_path.read_text(encoding='utf-8')
                updated_content = content
                for old_type, new_type in deprecated_types.items():
                    updated_content = updated_content.replace(old_type, new_type)
                
                if updated_content != content:  # Only write back if there were changes
                    file_path.write_text(updated_content, encoding='utf-8')
                    LOGGER.info(f'Replaced deprecated types in {file_path}.')
            except Exception as e:
                LOGGER.error(f'Error processing {file_path}: {e}')


def download_mot_dataset(val_tools_path, benchmark):
    """
    Download a specific MOT dataset zip file.
    
    Parameters:
        val_tools_path (Path): Path to the destination folder where the MOT benchmark zip will be downloaded.
        benchmark (str): The MOT benchmark to download (e.g., 'MOT20', 'MOT17').
    
    Returns:
        Path: The path to the downloaded zip file.
    """
    url = f'https://motchallenge.net/data/{benchmark}.zip'
    zip_dst = val_tools_path / f'{benchmark}.zip'

    if not zip_dst.exists():
        try:
            response = requests.head(url, allow_redirects=True)
            # Consider any status code less than 400 (e.g., 200, 302) as indicating that the resource exists
            if response.status_code < 400:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for HTTP request errors
                total_size_in_bytes = int(response.headers.get('content-length', 0))

                with open(zip_dst, 'wb') as file, tqdm(
                    desc=zip_dst.name,
                    total=total_size_in_bytes,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
                LOGGER.info(f'{benchmark}.zip downloaded successfully.')
            else:
                LOGGER.warning(f'{benchmark} is not downloadeable from {url}')
                zip_dst = None
        except requests.HTTPError as e:
            LOGGER.error(f'HTTP Error occurred while downloading {benchmark}.zip: {e}')
        except Exception as e:
            LOGGER.error(f'An error occurred: {e}')
    else:
        LOGGER.info(f'{benchmark}.zip already exists.')
    return zip_dst


def unzip_mot_dataset(zip_path, val_tools_path, benchmark):
    """
    Unzip a downloaded MOT dataset zip file into the specified directory.
    
    Parameters:
        zip_path (Path): Path to the downloaded MOT benchmark zip file.
        val_tools_path (Path): Base path to the destination folder where the dataset will be unzipped.
        benchmark (str): The MOT benchmark that was downloaded (e.g., 'MOT20', 'MOT17').
    
    Returns:
        None
    """
    if zip_path is None:
        LOGGER.warning(f'No zip file. Skipping unzipping')
        return None

    extract_path = val_tools_path / 'data' / benchmark
    if not extract_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # folder will be called as the original fetched file
                zip_ref.extractall(val_tools_path / 'data')

            LOGGER.info(f'{benchmark}.zip unzipped successfully.')
        except zipfile.BadZipFile:
            LOGGER.error(f'{zip_path.name} is corrupted. Try deleting the file and run the script again.')
        except Exception as e:
            LOGGER.error(f'An error occurred while unzipping {zip_path.name}: {e}')
    else:
        LOGGER.info(f'{benchmark} folder already exists.')
        return extract_path


def eval_setup(opt, val_tools_path):
    """
    Initializes and sets up evaluation paths for MOT challenge datasets.
    
    This function prepares the directories and paths needed for evaluating
    object tracking algorithms on MOT datasets like MOT17 or custom datasets like MOT17-mini.
    It filters sequence paths based on the detector (for MOT17), sets up the ground truth,
    sequences, and results directories according to the provided options.
    
    Parameters:
    - opt: An object with attributes that include benchmark (str), split (str),
      eval_existing (bool), project (str), and name (str). These options dictate
      the dataset to use, the split of the dataset, whether to evaluate on an
      existing setup, and the naming for the project and evaluation results directory.
    - val_tools_path: A string or Path object pointing to the base directory where
      the validation tools and datasets are located.
    
    Returns:
    - seq_paths: A list of Path objects pointing to the sequence directories to be evaluated.
    - save_dir: A Path object pointing to the directory where evaluation results will be saved.
    - MOT_results_folder: A Path object pointing to the directory where MOT challenge
      formatted results should be placed.
    - gt_folder: A Path object pointing to the directory where ground truth data is located.
    """

    # Convert val_tools_path to Path object if it's not already one
    val_tools_path = Path(val_tools_path)
    
    # Initial setup for paths based on benchmark and split options
    mot_seqs_path = val_tools_path / 'data' / opt.benchmark / opt.split
    # mot_seqs_path = Path('/media/v10016/实验室备份/XingshiXu/MyData/CowMulTrack_Data') / opt.split
    gt_folder = mot_seqs_path  # Assuming gt_folder is the same as mot_seqs_path initially
    
    # Handling different benchmarks
    if opt.benchmark == 'MOT17':
        # Filter for FRCNN sequences in MOT17
        # seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir() and 'FRCNN' in str(p)]
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir()]
    elif opt.benchmark == 'MOT17-mini':
        # Adjust paths for MOT17-mini
        base_path = ROOT / 'assets' / opt.benchmark / opt.split
        mot_seqs_path = gt_folder = base_path
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir()]
    else:
        # Default handling for other datasets
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir()]

    # Determine save directory
    save_dir = Path(opt.project) / opt.name


    # Setup MOT results folder
    MOT_results_folder = val_tools_path / 'data' / 'trackers' / 'mot_challenge' / opt.benchmark / save_dir.name / 'data'
    MOT_results_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    return seq_paths, save_dir, MOT_results_folder, gt_folder

"""
增加以用于保存输出结果
results_add_emb 是一个ndarry(num_track,2568).前八个是xyxy,id,conf,cls,det_ind.后面每512个是一个表观特征 分别是“前后右左”
"""
def tracks_add_emb_convert_to_mot_format(results_add_emb, frame_idx):
    # Check if results are not empty
    if results_add_emb.size != 0:
        if isinstance(results_add_emb, np.ndarray):
            # print("isinstance(results, np.ndarray)")
            # Convert numpy array results to MOT format
            tlwh = ops.xyxy2ltwh(results_add_emb[:, 0:4])
            frame_idx_column = np.full((results_add_emb.shape[0], 1), frame_idx, dtype=np.int32)
            mot_results = np.column_stack((
                frame_idx_column,  # frame index                    【1列】
                results_add_emb[:, 4].astype(np.int32),  # track id 【2列】
                tlwh.astype(np.int32),  # top,left,width,height     【3-6列】
                np.ones((results_add_emb.shape[0], 1), dtype=np.int32),  # "not ignored"
                results_add_emb[:, 6].astype(np.int32),  # class    【8列】
                results_add_emb[:, 5],  # confidence (float)        【9列】

                results_add_emb[:, 7:(7+1+512)],
                results_add_emb[:, (7 + 513):(7 +1+ 1024)],
                results_add_emb[:, (7 + 1025):(7 +1+ 1536)],
                results_add_emb[:, (7 + 1537):(7 +1+ 2048)],
                results_add_emb[:, (7 + 2049):],

            ))
            return mot_results
        else:
            print("注意：此处不是ndarray的数组")
            # Convert ultralytics results to MOT format
            num_detections = len(results_add_emb.boxes)
            frame_indices = torch.full((num_detections, 1), frame_idx + 1, dtype=torch.int32)
            not_ignored = torch.ones((num_detections, 1), dtype=torch.int32)

            mot_results = torch.cat([
                frame_indices,  # frame index
                results_add_emb.boxes.id.unsqueeze(1).astype(np.int32),  # track id
                ops.xyxy2ltwh(results_add_emb.boxes.xyxy).astype(np.int32),  ## top,left,width,height
                not_ignored,  # "not ignored"
                results_add_emb.boxes.cls.unsqueeze(1).astype(np.int32),  # class
                results_add_emb.boxes.conf.unsqueeze(1).astype(np.float32),  # confidence (float)
            ], dim=1)

            return mot_results.numpy()

    pass

def convert_to_mot_format(results: Union[Results, np.ndarray], frame_idx: int) -> np.ndarray:
    """
    用于将单帧的跟踪结果转换为 MOT（Multiple Object Tracking）挑战赛的标准格式。它可以处理两种类型的输入数据：numpy 数组和 ultralytics 自定义结果对象。
    Converts tracking results for a single frame into MOT challenge format.

    This function supports inputs as either a custom object with a 'boxes' attribute or a numpy array.
    For custom object inputs, 'boxes' should contain 'id', 'xyxy', 'conf', and 'cls' sub-attributes.
    For numpy array inputs, the expected format per row is: (xmin, ymin, xmax, ymax, id, conf, cls).

    Parameters:
    - results (Union[Results, np.ndarray]): Tracking results for the current frame.
    - frame_idx (int): The zero-based index of the frame being processed.

    Returns:
    - np.ndarray: An array containing the MOT formatted results for the frame.
    """

    # Check if results are not empty
    if results.size != 0:
        if isinstance(results, np.ndarray):
            print("isinstance(results, np.ndarray)")
            # Convert numpy array results to MOT format
            tlwh = ops.xyxy2ltwh(results[:, 0:4])
            frame_idx_column = np.full((results.shape[0], 1), frame_idx, dtype=np.int32)
            mot_results = np.column_stack((
                frame_idx_column, # frame index
                results[:, 4].astype(np.int32),  # track id
                tlwh.astype(np.int32),  # top,left,width,height
                np.ones((results.shape[0], 1), dtype=np.int32),  # "not ignored"
                results[:, 6].astype(np.int32),  # class
                results[:, 5],  # confidence (float)
            ))
            return mot_results
        else:
            # Convert ultralytics results to MOT format
            num_detections = len(results.boxes)
            frame_indices = torch.full((num_detections, 1), frame_idx + 1, dtype=torch.int32)
            not_ignored = torch.ones((num_detections, 1), dtype=torch.int32)

            mot_results = torch.cat([
                frame_indices, # frame index
                results.boxes.id.unsqueeze(1).astype(np.int32), # track id
                ops.xyxy2ltwh(results.boxes.xyxy).astype(np.int32),  ## top,left,width,height
                not_ignored, # "not ignored"
                results.boxes.cls.unsqueeze(1).astype(np.int32), # class
                results.boxes.conf.unsqueeze(1).astype(np.float32), # confidence (float)
            ], dim=1)

            return mot_results.numpy()


def write_mot_results(txt_path: Path, mot_results: np.ndarray) -> None:
    """
    Writes the MOT challenge formatted results to a text file.

    Parameters:
    - txt_path (Path): The path to the text file where results are saved.
    - mot_results (np.ndarray): An array containing the MOT formatted results.

    Note: The text file will be created if it does not exist, and the directory
    path to the file will be created as well if necessary.
    """
    if mot_results is not None:
        if mot_results.size != 0:
            # Ensure the parent directory of the txt_path exists
            txt_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure the file exists before opening
            txt_path.touch(exist_ok=True)

            # Open the file in append mode and save the MOT results
            with open(str(txt_path), 'a') as file:
                np.savetxt(file, mot_results, fmt='%d,%d,%d,%d,%d,%d,%d,%d,%.6f')

def write_mot_results_ADDED(txt_path: Path, mot_results: np.ndarray) -> None:

    if mot_results is not None:
        if mot_results.size != 0:
            # Ensure the parent directory of the txt_path exists
            txt_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure the file exists before opening
            txt_path.touch(exist_ok=True)

            # Open the file in append mode and save the MOT results
            with open(str(txt_path), 'a') as file:
                num_cols = mot_results.shape[1]
                print("---{}".format(num_cols))
                fmt = '%d,' * 8 + '%.6f，' * (num_cols-9) + '%.6f'

                np.savetxt(file, mot_results, fmt=fmt)
