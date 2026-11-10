"""Editable analysis intervals and frame-accurate boundary evidence."""
import bisect
import copy
import json
import subprocess
import uuid
import tempfile
from functools import lru_cache
from fractions import Fraction
from pathlib import Path


def ensure(job):
    """为旧任务补建片段：每个已有镜头对应一个片段。"""
    if 'clips' not in job:
        job['clips'] = [_new(s['id'], s['start_ms'], s['end_ms']) for s in job['shots']]
    return job['clips']


def _new(shot_id, start, end, sources=None):
    return {'id': uuid.uuid4().hex, 'shot_id': shot_id, 'start_ms': start,
            'end_ms': end, 'source_clip_ids': sources or [], 'status': 'pending'}


def split(job, clip_id, ms):
    """将一个片段拆为两个子片段，并归档拆分前的状态。"""
    items = ensure(job)
    index = next((i for i, c in enumerate(items) if c['id'] == clip_id), None)
    if index is None: raise ValueError('找不到片段。')
    clip = items[index]
    ms = int(ms)
    if not clip['start_ms'] < ms < clip['end_ms']: raise ValueError('切点必须位于片段内部。')
    left = _new(clip['shot_id'], clip['start_ms'], ms, [clip_id])
    right = _new(clip['shot_id'], ms, clip['end_ms'], [clip_id])
    if clip.get('director_note'):
        left['director_note'] = right['director_note'] = clip['director_note']
    job.setdefault('clip_history', []).append(copy.deepcopy(clip))
    items[index:index+1] = [left, right]
    return left, right


def merge(job, left_id, right_id):
    """仅合并同一镜头内时间连续的相邻片段。"""
    items = ensure(job)
    index = next((i for i, c in enumerate(items) if c['id'] == left_id), None)
    if index is None or index + 1 >= len(items) or items[index+1]['id'] != right_id:
        raise ValueError('请选择相邻且按顺序排列的两个片段。')
    left, right = items[index:index+2]
    if left['shot_id'] != right['shot_id'] or left['end_ms'] != right['start_ms']:
        raise ValueError('只能合并同一镜头内相邻的片段。')
    result = _new(left['shot_id'], left['start_ms'], right['end_ms'], [left_id, right_id])
    notes = [c.get('director_note', '') for c in (left, right)]
    if any(notes):
        result['director_note'] = '\n\n'.join(f"{c['start_ms']/1000:.2f}–{c['end_ms']/1000:.2f} 秒：\n{note}" for c, note in zip((left, right), notes) if note)
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
    notes = [c for c in old if c.get('director_note')]
    if notes:
        result['director_note'] = '\n\n'.join(f"{c['start_ms']/1000:.2f}–{c['end_ms']/1000:.2f} 秒：\n{c['director_note']}" for c in notes)
    items[indices[0]:indices[-1]+1] = [result]
    return result


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if result.returncode: raise ValueError('读取视频帧失败：' + result.stderr[-500:])
    return result.stdout


def frame_index(job):
    """返回新的列表视图，避免调用方意外修改缓存索引。"""
    import engine
    path = engine.DATA / job['id'] / job['video']['file']
    stat=path.stat()
    return [dict(frame) for frame in _frame_index(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ino)]


@lru_cache(maxsize=4)
def _frame_index(path, size, mtime_ns, inode):
    # 将文件身份纳入缓存键；源视频替换或修改后会重新探测时间戳。
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


def _boundary_indices(frames, clip, span_ms):
    """按左闭右开时间范围，为片段选取已解码的视频帧。"""
    if span_ms < 0: raise ValueError('边界证据时长不能为负。')
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
    return first,last,head,tail


def _extract(job, indexes):
    """分批提取缺失帧；一批全部成功后才写入正式缓存目录。"""
    import engine
    folder = engine.DATA / job['id'] / 'clip_frames'
    folder.mkdir(parents=True, exist_ok=True)
    video = engine.DATA / job['id'] / job['video']['file']
    missing=[i for i in indexes if not (folder / f'{i:08d}.jpg').exists()]
    for start in range(0,len(missing),200):
        batch=missing[start:start+200]
        expression='+'.join(f'eq(n\\,{i})' for i in batch)
        with tempfile.TemporaryDirectory(prefix='.frames-',dir=folder) as tmp:
            _run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select={expression}',
                  '-vsync','0','-frames:v',str(len(batch)),'-q:v','2','-start_number','0',str(Path(tmp)/'%08d.jpg')])
            files=[Path(tmp)/f'{n:08d}.jpg' for n in range(len(batch))]
            if not all(p.exists() for p in files): raise ValueError('无法提取全部边界帧。')
            for i,p in zip(batch,files): p.replace(folder/f'{i:08d}.jpg')
    return folder


def preload(job, span_ms=500):
    """导出前为所有片段预热共享帧缓存。"""
    frames=frame_index(job)
    indexes=set()
    for clip in ensure(job):
        _,_,head,tail=_boundary_indices(frames,clip,span_ms)
        indexes.update(head+tail)
    _extract(job,sorted(indexes))


def evidence(job, clip_id, span_ms=500):
    """返回源视频首尾帧及片段两端的稀疏上下文帧。"""
    clip = next((c for c in ensure(job) if c['id'] == clip_id), None)
    if clip is None: raise ValueError('找不到片段。')
    frames=frame_index(job)
    first,last,head,tail=_boundary_indices(frames,clip,span_ms)
    indexes=sorted(set(head+tail))
    folder=_extract(job,indexes)
    assets = {}
    for i in indexes:
        path = folder / f'{i:08d}.jpg'
        assets[i] = {**frames[i], 'path': str(path)}
    return {'source_head': assets[first], 'source_tail': assets[last],
            'head_frames': [assets[i] for i in head], 'tail_frames': [assets[i] for i in tail],
            'span_ms': span_ms}
