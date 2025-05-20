import cv2
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def extract_frame(frame_info):
    frame_number, frame, output_folder, image_format = frame_info
    image_filename = f"{output_folder}/{str(frame_number).zfill(8)}.{image_format}"
    cv2.imwrite(image_filename, frame)

def extract_frames(video_path, output_folder, frame_interval=1, image_format='png'):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化帧计数器和图像编号
    frame_count = 0
    image_count = 0

    pbar = tqdm(total=total_frames, desc="Extracting frames")

    pool = Pool()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 只截取每隔 frame_interval 帧
        if frame_count % frame_interval == 0:
            # 并行处理保存图像
            pool.apply_async(extract_frame, args=((image_count, frame, output_folder, image_format),))

            # 更新图像编号
            image_count += 1

        pbar.update(1)

    pbar.close()
    cap.release()
    pool.close()
    pool.join()

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video_path", default=r'CowMulDataFPS5\20240425_3_02_D2_fps5.mp4', help="Path to the input video file")
    parser.add_argument("--output_folder", default=r'CowFPS\20240425_3_02_D2\img1', help="Path to the output folder to save frames")
    parser.add_argument("-i", "--interval", type=int, default=1, help="Frame extraction interval (default: 1)")
    parser.add_argument("-f", "--format", default='png', help="Image format (default: png)")

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.interval, args.format)

if __name__ == "__main__":
    main()
