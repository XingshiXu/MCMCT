'''
将mot的gt标签批量转换为yolo的标签
每个序列的图片必须以帧数命名，且文件名为八位数，不足的用0补齐
'''

import os
import os.path as osp
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def copy_image(src_img_path, dest_img_path):
    """
    复制单张图片
    """
    try:
        shutil.copy2(src_img_path, dest_img_path)  # shutil.copy2保留文件的元数据
    except Exception as e:
        print(f"复制文件失败 {src_img_path} -> {dest_img_path}, 错误: {e}")

def copy_images_to_yolo_multithread(input_folder, output_folder, max_workers=8):
    """
    使用多线程加速图片复制操作
    """
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    video_folders = os.listdir(input_folder)
    tasks = []

    # 使用线程池管理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_folder in video_folders:
            video_folder_path = osp.join(input_folder, video_folder)
            img_folder_path = osp.join(video_folder_path, 'img1')

            if not osp.exists(img_folder_path):
                print(f"视频文件夹 {video_folder_path} 下没有 img1 文件夹，跳过。")
                continue

            imgs = os.listdir(img_folder_path)
            for img_name in imgs:
                # 图片的源路径
                src_img_path = osp.join(img_folder_path, img_name)
                # 拼接新的图片名称
                new_img_name = f"{video_folder}_{img_name}"
                # 图片的目标路径
                dest_img_path = osp.join(output_folder, new_img_name)

                # 提交任务给线程池
                tasks.append(executor.submit(copy_image, src_img_path, dest_img_path))

        # 通过 tqdm 监控进度
        for task in tqdm(as_completed(tasks), total=len(tasks), desc="复制图片"):
            task.result()  # 确保捕获异常

    print("所有图片复制完成。")

def main():
    # 输入输出路径
    input_folder = r'E:\CowMulTrack_Data\train'  # 输入文件夹路径
    output_folder = r'E:\CowMulTrack_Data_YOLO\train\image'  # 输出文件夹路径

    # 使用多线程进行图片复制
    copy_images_to_yolo_multithread(input_folder, output_folder, max_workers=16)  # 你可以调整线程数

if __name__ == '__main__':
    main()
