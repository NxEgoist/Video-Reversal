"""Local, conservative black-frame candidates; no source intervals are removed."""
from bisect import bisect_left

import cv2
import numpy as np
import clips
import engine


def detect(job):
    """仅标记抽样帧均为暗场的镜头，不删除或裁剪源视频。"""
    frames = clips.frame_index(job)
    shots = job['shots']
    selected = {}
    times = [f['time_ms'] for f in frames]
    for shot in shots:
        inside = frames[bisect_left(times, shot['start_ms']):bisect_left(times, shot['end_ms'])]
        chosen = []
        for frame in inside:
            if not chosen or frame['time_ms'] - chosen[-1]['time_ms'] >= 250:
                chosen.append(frame)
        if inside and (not chosen or chosen[-1]['index'] != inside[-1]['index']):
            chosen.append(inside[-1])
        for frame in chosen:
            selected[frame['index']] = (shot, frame['time_ms'])
    values = {s['id']: [] for s in shots}
    cap = cv2.VideoCapture(str(engine.DATA / job['id'] / job['video']['file']))
    if not cap.isOpened():
        raise ValueError('无法读取视频画面。')
    try:
        for index in range(max(selected, default=-1) + 1):
            if not cap.grab():
                raise ValueError('视频解码不完整，未保存黑场检测结果。')
            if index in selected:
                ok, image = cap.retrieve()
                if not ok:
                    raise ValueError('视频画面读取失败，未保存黑场检测结果。')
                shot, ms = selected[index]
                grey = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2GRAY)
                values[shot['id']].append((round(ms, 3), float(np.mean(grey <= 18))))
    finally:
        cap.release()
    for shot in shots:
        # 保存检测时的镜头范围，便于切点变化后判断结果是否仍然适用。
        samples = values[shot['id']]
        shot['black_screen'] = {'candidate': bool(samples) and all(ratio >= .995 for _, ratio in samples),
                                'sample_times_ms': [ms for ms, _ in samples],
                                'start_ms': shot['start_ms'], 'end_ms': shot['end_ms']}
