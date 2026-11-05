"""User-reviewed, versioned context and continuity notes for analysis clips.

This module only mutates the supplied job; callers persist it with engine.save.
It never infers continuity or overwrites source observations.
"""
import copy
import re


KINDS = {'角色', '服装', '道具', '环境'}
STATUSES = {'候选', '已确认', '冲突'}
BOUNDARIES = {'同镜头拆分', '硬切', '转场', '跳时或跳空间', '未知'}
REVIEWS = {'待核对', '符合', '冲突', '无法判断'}
ID = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def _clips(clips):
    result = {}
    for clip in clips:
        ident = clip['id']
        if ident in result:
            raise ValueError('片段 ID 重复。')
        result[ident] = clip
    return result


def _evidence(value, lookup):
    if not isinstance(value, list):
        raise ValueError('证据必须是列表。')
    result = []
    for item in value:
        if not isinstance(item, dict) or item.get('clip_id') not in lookup:
            raise ValueError('证据必须引用存在的片段。')
        clip = lookup[item['clip_id']]
        ms = item.get('time_ms')
        if isinstance(ms, bool) or not isinstance(ms, int) or not clip['start_ms'] <= ms < clip['end_ms']:
            raise ValueError('证据时间必须是片段核心范围内的原视频毫秒数。')
        result.append({'clip_id': item['clip_id'], 'time_ms': ms})
    return result


def read(job, clips):
    """Return a detached context and mark records invalidated by clip changes."""
    context = copy.deepcopy(job.get('clip_context') or {'version': 0, 'overview': '', 'assets': [], 'states': [], 'edges': []})
    lookup = _clips(clips)
    adjacent = {(a['id'], b['id']) for a, b in zip(clips, clips[1:])}
    for asset in context['assets']:
        asset['stale'] = any(cid not in lookup for cid in asset['clip_ids']) or any(
            e['clip_id'] not in lookup or not lookup[e['clip_id']]['start_ms'] <= e['time_ms'] < lookup[e['clip_id']]['end_ms']
            for e in asset['evidence'])
    for state in context['states']:
        clip = lookup.get(state['clip_id'])
        state['stale'] = clip is None or state['clip_range_ms'] != [clip['start_ms'], clip['end_ms']]
    for edge in context['edges']:
        pair = (edge['from_clip'], edge['to_clip'])
        edge['stale'] = pair not in adjacent or any(
            lookup.get(cid) is None or edge['clip_ranges_ms'].get(cid) != [lookup[cid]['start_ms'], lookup[cid]['end_ms']]
            for cid in pair)
    return context


def save(job, clips, value):
    """Validate and atomically replace user-entered context; increment version."""
    if not isinstance(value, dict):
        raise ValueError('全片上下文必须是 JSON 对象。')
    lookup = _clips(clips)
    adjacent = {(a['id'], b['id']) for a, b in zip(clips, clips[1:])}
    overview = value.get('overview', '')
    if not isinstance(overview, str) or len(overview) > 4000:
        raise ValueError('全片概览必须是 4000 字以内文本。')
    assets = []
    seen = set()
    for item in value.get('assets', []):
        if not isinstance(item, dict): raise ValueError('素材卡必须是对象。')
        ident = item.get('id', '')
        if not isinstance(ident, str) or not ID.fullmatch(ident) or ident in seen:
            raise ValueError('素材 ID 必须唯一且仅含字母、数字、下划线或短横线。')
        seen.add(ident)
        kind, status, description = item.get('kind'), item.get('status', '候选'), item.get('description', '')
        if kind not in KINDS or status not in STATUSES or not isinstance(description, str) or not description.strip() or len(description) > 1000:
            raise ValueError('素材卡类型、状态或描述无效。')
        ids = item.get('clip_ids', [])
        if not isinstance(ids, list) or any(cid not in lookup for cid in ids):
            raise ValueError('素材卡关联了不存在的片段。')
        assets.append({'id': ident, 'kind': kind, 'description': description.strip(), 'status': status,
                       'clip_ids': list(dict.fromkeys(ids)), 'evidence': _evidence(item.get('evidence', []), lookup)})
    states = []
    state_keys = set()
    for item in value.get('states', []):
        if not isinstance(item, dict): raise ValueError('片段状态必须是对象。')
        cid, asset = item.get('clip_id'), item.get('asset_id')
        if cid not in lookup or asset not in seen or (cid, asset) in state_keys:
            raise ValueError('片段状态必须引用唯一的已有片段和素材卡。')
        state_keys.add((cid, asset))
        start, end = item.get('start', ''), item.get('end', '')
        if not all(isinstance(v, str) and len(v) <= 1000 for v in (start, end)):
            raise ValueError('起止状态须为 1000 字以内文本。')
        status = item.get('status', '候选')
        if status not in STATUSES: raise ValueError('片段状态的确认状态无效。')
        clip = lookup[cid]
        if item.get('clip_range_ms', [clip['start_ms'], clip['end_ms']]) != [clip['start_ms'], clip['end_ms']]:
            raise ValueError('片段范围已变化；请重新核对并删除该状态的 clip_range_ms 后再保存。')
        states.append({'clip_id': cid, 'asset_id': asset, 'start': start, 'end': end, 'status': status,
                       'evidence': _evidence(item.get('evidence', []), lookup),
                       'clip_range_ms': [clip['start_ms'], clip['end_ms']]})
    edges = []
    edge_keys = set()
    for item in value.get('edges', []):
        if not isinstance(item, dict): raise ValueError('衔接记录必须是对象。')
        pair = (item.get('from_clip'), item.get('to_clip'))
        if pair not in adjacent or pair in edge_keys:
            raise ValueError('衔接记录必须引用唯一的相邻片段。')
        edge_keys.add(pair)
        kind, review, note = item.get('type', '未知'), item.get('review', '待核对'), item.get('note', '')
        if kind not in BOUNDARIES or review not in REVIEWS or not isinstance(note, str) or len(note) > 2000:
            raise ValueError('衔接类型、核对状态或说明无效。')
        ranges = {cid: [lookup[cid]['start_ms'], lookup[cid]['end_ms']] for cid in pair}
        if item.get('clip_ranges_ms', ranges) != ranges:
            raise ValueError('衔接范围已变化；请重新核对并删除该衔接的 clip_ranges_ms 后再保存。')
        edges.append({'from_clip': pair[0], 'to_clip': pair[1], 'type': kind, 'review': review,
                      'note': note, 'evidence': _evidence(item.get('evidence', []), lookup),
                      'clip_ranges_ms': ranges})
    previous = job.get('clip_context') or {}
    result = {'version': previous.get('version', 0) + 1, 'overview': overview,
              'assets': assets, 'states': states, 'edges': edges}
    job['clip_context'] = result
    return copy.deepcopy(result)


def for_clip(job, clips, clip_id):
    """Return only confirmed, current context linked to the requested clip."""
    if clip_id not in _clips(clips): raise ValueError('片段不存在。')
    context = read(job, clips)
    assets = [a for a in context['assets'] if not a['stale'] and a['status'] == '已确认' and clip_id in a['clip_ids']]
    asset_ids = {a['id'] for a in assets}
    return {'version': context['version'], 'overview': context['overview'],
            'assets': assets,
            'states': [s for s in context['states'] if not s['stale'] and s['status'] == '已确认' and s['clip_id'] == clip_id and s['asset_id'] in asset_ids],
            'edges': [e for e in context['edges'] if not e['stale'] and clip_id in (e['from_clip'], e['to_clip'])]}
