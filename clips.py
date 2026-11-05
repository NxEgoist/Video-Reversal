"""Editable analysis intervals and frame-accurate boundary evidence."""
import bisect
import copy
import json
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path


def ensure(job):
    if 'clips' not in job:
        job['clips'] = [_new(s['id'], s['start_ms'], s['end_ms']) for s in job['shots']]
    return job['clips']


def _new(shot_id, start, end, sources=None):
    return {'id': uuid.uuid4().hex, 'shot_id': shot_id, 'start_ms': start,
            'end_ms': end, 'source_clip_ids': sources or [], 'status': 'pending'}


def split(job, clip_id, ms):
    items = ensure(job)
    index = next((i for i, c in enumerate(items) if c['id'] == clip_id), None)
    if index is None: raise ValueError('找不到片段。')
    clip = items[index]
    ms = int(ms)
    if not clip['start_ms'] < ms < clip['end_ms']: raise ValueError('切点必须位于片段内部。')
    left = _new(clip['shot_id'], clip['start_ms'], ms, [clip_id])
    right = _new(clip['shot_id'], ms, clip['end_ms'], [clip_id])
    job.setdefault('clip_history', []).append(copy.deepcopy(clip))
    items[index:index+1] = [left, right]
    return left, right


def merge(job, left_id, right_id):
    items = ensure(job)
    index = next((i for i, c in enumerate(items) if c['id'] == left_id), None)
    if index is None or index + 1 >= len(items) or items[index+1]['id'] != right_id:
        raise ValueError('请选择相邻且按顺序排列的两个片段。')
    left, right = items[index:index+2]
    if left['shot_id'] != right['shot_id'] or left['end_ms'] != right['start_ms']:
        raise ValueError('只能合并同一镜头内相邻的片段。')
    result = _new(left['shot_id'], left['start_ms'], right['end_ms'], [left_id, right_id])
    job.setdefault('clip_history', []).extend(copy.deepcopy([left, right]))
    items[index:index+2] = [result]
    return result


def reset_shot(job, shot_id):
    """Archive obsolete clips after a shot boundary edit and create a fresh clip."""
    items = ensure(job)
    shot = next((s for s in job['shots'] if s['id'] == shot_id), None)
    if shot is None: raise ValueError('找不到镜头。')
    indices = [i for i, c in enumerate(items) if c['shot_id'] == shot_id]
    if not indices: raise ValueError('该镜头没有片段。')
    old = [items[i] for i in indices]
    job.setdefault('clip_history', []).extend(copy.deepcopy(old))
    result = _new(shot_id, shot['start_ms'], shot['end_ms'], [c['id'] for c in old])
    items[indices[0]:indices[-1]+1] = [result]
    return result


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if result.returncode: raise ValueError('读取视频帧失败：' + result.stderr[-500:])
    return result.stdout


def frame_index(job):
    import engine
    path = engine.DATA / job['id'] / job['video']['file']
    raw = json.loads(_run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=time_base:frame=best_effort_timestamp',
        '-show_frames', '-of', 'json', str(path)]))
    streams = raw.get('streams', [])
    if not streams or not raw.get('frames'): raise ValueError('无法读取视频帧时间戳。')
    time_base = streams[0].get('time_base')
    try:
        scale = Fraction(time_base)
        pts = [int(f['best_effort_timestamp']) for f in raw['frames']]
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('视频帧时间戳不完整。') from exc
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('视频帧时间戳不递增。')
    return [{'index': i, 'pts': p, 'time_base': time_base,
             'time_ms': float((p - pts[0]) * scale * 1000)} for i, p in enumerate(pts)]


def evidence(job, clip_id, span_ms=500):
    import engine
    clip = next((c for c in ensure(job) if c['id'] == clip_id), None)
    if clip is None: raise ValueError('找不到片段。')
    if span_ms < 0: raise ValueError('边界证据时长不能为负。')
    frames = frame_index(job)
    times = [f['time_ms'] for f in frames]
    first = bisect.bisect_left(times, clip['start_ms'])
    last = bisect.bisect_left(times, clip['end_ms']) - 1
    if first >= len(frames) or last < first: raise ValueError('片段区间内没有可解码帧，请调整切点。')
    # 250 ms temporal samples plus exact endpoints; not every decoded frame.
    def sample(a, b):
        indices = {a, b}
        t = times[a] + 250
        while t < times[b]:
            indices.add(min(b, bisect.bisect_left(times, t, a, b+1)))
            t += 250
        return sorted(indices)
    head_end = bisect.bisect_right(times, min(times[last], times[first]+span_ms))-1
    tail_start = bisect.bisect_left(times, max(times[first], times[last]-span_ms))
    head, tail = sample(first, head_end), sample(tail_start, last)
    folder = engine.DATA / job['id'] / 'clip_frames'
    folder.mkdir(parents=True, exist_ok=True)
    video = engine.DATA / job['id'] / job['video']['file']
    assets = {}
    for i in sorted(set(head + tail)):
        path = folder / f'{i:08d}.jpg'
        if not path.exists():
            _run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', f'select=eq(n\\,{i})',
                  '-vsync', '0', '-frames:v', '1', '-q:v', '2', '-y', str(path)])
            if not path.exists(): raise ValueError('无法提取边界帧。')
        assets[i] = {**frames[i], 'path': str(path)}
    return {'source_head': assets[first], 'source_tail': assets[last],
            'head_frames': [assets[i] for i in head], 'tail_frames': [assets[i] for i in tail],
            'span_ms': span_ms}
