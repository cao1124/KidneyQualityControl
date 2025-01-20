#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control
@File    ：get_img_from_video.py
@IDE     ：PyCharm
@Author  ：cao xu
@Date    ：2025/1/20 上午10:54
"""
import cv2
import os
import glob


def extract_frames(video_path, output_folder, frame_count=30):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算间隔，确保每个视频提取30张图像
    interval = total_frames // frame_count if total_frames >= frame_count else 1

    frames = []
    for i in range(frame_count):
        frame_num = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    # 保存提取的帧到输出文件夹
    for idx, frame in enumerate(frames):
        frame_filename = os.path.join(output_folder, f"frame_{idx + 1}.jpg")
        cv2.imwrite(frame_filename, frame)


def process_videos():
    # 获取当前路径下所有常见的视频文件
    video_files = glob.glob(os.path.join(os.getcwd(), "*.[mM][pP]4"))
    video_files += glob.glob(os.path.join(os.getcwd(), "*.[aA][vV][iI]"))
    video_files += glob.glob(os.path.join(os.getcwd(), "*.[mM][oO][vV]"))
    video_files += glob.glob(os.path.join(os.getcwd(), "*.[mM][kK][vV]"))

    for video_path in video_files:
        # 为每个视频创建一个以视频名命名的文件夹
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(os.getcwd(), video_name)
        os.makedirs(output_folder, exist_ok=True)

        # 提取视频帧
        extract_frames(video_path, output_folder)


if __name__ == "__main__":
    process_videos()
