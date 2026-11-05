"""镜头内采样：基础覆盖、变化积分和定时兜底。"""
import cv2
import numpy as np


def change_score(previous, current):
    a = cv2.cvtColor(cv2.resize(previous, (64, 36)), cv2.COLOR_BGR2HSV).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(current, (64, 36)), cv2.COLOR_BGR2HSV).astype(np.float32)
    delta = np.abs(a - b)
    hue = np.minimum(delta[:, :, 0], 180 - delta[:, :, 0]) * (255 / 180)
    hue *= np.minimum(a[:, :, 1], b[:, :, 1]) / 255
    return float(np.mean((hue + delta[:, :, 1] + delta[:, :, 2]) / 3))


def adaptive_samples(path, start, end):
    if start < 0 or end <= start:
        raise ValueError('无效的镜头时间范围。')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('无法读取视频画面。')
    # 保证起、中、尾及每两秒的覆盖；动态帧只作为补充。
    marks = {start: {'基础'}, (start + end)//2: {'基础'}, max(start, end-100): {'基础'}}
    for t in range(start, end, 2000): marks.setdefault(t, set()).add('时间兜底')
    previous = None
    last_scan = start - 250
    last_dynamic = start - 250
    accumulated = 0.0
    last_pts = -1
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start)
        while True:
            ok, image = cap.read()
            if not ok: break
            pts = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
            if pts < last_pts:
                raise ValueError('无法可靠读取视频时间戳，请选择等间隔采样。')
            last_pts = pts
            if pts < start: continue
            if pts >= end: break
            if pts - last_scan < 250: continue
            last_scan = pts
            if previous is not None:
                score = change_score(previous, image)
                accumulated += score
                if (score >= 8 or accumulated >= 18) and pts-last_dynamic >= 250:
                    marks.setdefault(pts, set()).add('画面变化')
                    last_dynamic = pts
                    accumulated = 0.0
            previous = image
    finally:
        cap.release()
    if last_pts < 0:
        raise ValueError('无法读取视频画面。')
    return [{'ms':ms, 'reasons':sorted(reasons)} for ms,reasons in sorted(marks.items())]


def plan(path, start, end, strategy, manual=None):
    if strategy == '变化驱动':
        marks = {p['ms']:set(p['reasons']) for p in adaptive_samples(path,start,end)}
    elif strategy == '等间隔':
        marks = {t:{'等间隔'} for t in range(start,end,500)}
        for t in (start,(start+end)//2,max(start,end-100)): marks.setdefault(t,set()).add('基础')
    else:
        raise ValueError('未知的抽帧策略。')
    for ms in manual or []:
        if start <= ms < end: marks.setdefault(ms,set()).add('人工补帧')
    return [{'ms':ms,'reasons':sorted(reasons)} for ms,reasons in sorted(marks.items())]
