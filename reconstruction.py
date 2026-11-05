"""Build a reproducible handoff from saved evidence; no model calls or mutations."""
import copy


def build(job):
    shots = []
    for shot in job.get('shots', []):
        start, end = shot['start_ms'], shot['end_ms']
        ref = shot.get('reference_ms', start)
        observation = shot.get('observation') or {}
        prompts = shot.get('prompts') or {}
        checks = []
        if shot.get('status') != 'complete':
            checks.append('当前分析未完整完成或已有修改，请核对并按需重新生成 Prompt。')
        if not observation:
            checks.append('缺少画面观察，不能视为已完成视频逆向。')
        if not all(prompts.get(key, '').strip() for key in ('text_zh', 'text_en', 'image_zh', 'image_en')):
            checks.append('双语提示词尚未齐全。')
        valid_ref = start <= ref < end
        if not valid_ref:
            checks.append('参考帧超出镜头范围，请重新选择。')
        elif ref > start:
            checks.append(f'图生视频只覆盖参考帧之后；前 {(ref-start)/1000:.3f} 秒需另做片段或改选首帧，不能假装覆盖全镜头。')
        if observation.get('incomplete_review'):
            checks.append('动作补看未完成，请继续分析。')
        for field, label in (('missing_ids', '动作覆盖待核对'), ('reference_uncertain_ids', '跨参考帧动作待核对')):
            ids = shot.get('coverage_review', {}).get(field, [])
            if ids:
                checks.append(label + '：' + ', '.join(ids))
        for issue in observation.get('camera_review', []) + shot.get('prompt_review', []):
            checks.append(issue['warning'])
        stages = []
        for prefix, windows in (('w', observation.get('windows', [])), ('s', observation.get('supplemental', []))):
            for i, window in enumerate(windows, 1):
                checks.extend(window.get('unknowns', []))
                if window.get('needs_review_ranges'):
                    checks.append(f'{prefix}{i} 曾请求补看，请结合补看结果核对动作。')
                for j, beat in enumerate(window.get('beats', []), 1):
                    stages.append({'id': f'{prefix}{i}b{j}', **copy.deepcopy(beat)})
        available = valid_ref and bool(observation or shot.get('reference_user_selected'))
        shots.append({
            'id': shot['id'], 'source_range_ms': [start, end],
            'target_duration_ms': end-start,
            'reference': {'source_ms': ref, 'kind': '原视频截图，未经抠图或修复',
                          'export_path': f"references/{shot['id']}_{ref}.jpg" if available else None},
            'routes': {
                'text_to_video': {'source_range_ms': [start, end], 'duration_ms': end-start,
                                  'prompt_zh': prompts.get('text_zh', ''), 'prompt_en': prompts.get('text_en', '')},
                'image_to_video': {'source_range_ms': [ref, end] if valid_ref else None,
                                   'duration_ms': end-ref if valid_ref else None,
                                   'prompt_zh': prompts.get('image_zh', ''), 'prompt_en': prompts.get('image_en', '')}},
            'subjects': [{'id': ident, **copy.deepcopy(job.get('subjects', {})[ident])}
                         for ident in shot.get('subject_ids', []) if ident in job.get('subjects', {})],
            'evidence_stages': stages, 'user_overrides': copy.deepcopy(shot.get('overrides', {})),
            'review_items': list(dict.fromkeys(checks)),
        })
    return {'schema_version': '1.0', 'kind': '候选复现方案，不是原始制作流程鉴定',
            'source_job_id': job.get('id'),
            'unrecoverable_from_video': ['原始提示词原文', '原始生成模型及版本', '随机种子', '采样器与内部参数'],
            'generation_settings': {'model': None, 'supported_duration_seconds': None,
                                    'resolution': None, 'seed': None},
            'audio': copy.deepcopy(job.get('audio_transcript')),
            'shots': shots}


def markdown(plan):
    lines = ['# 视频复现计划', '', plan['kind'], '',
             '本计划按已保存内容即时整理，不调用生成服务。提示词为候选复现描述；观察和动作阶段仍需对照原片。', '',
             '## 1. 准备素材与设置', '',
             '- 先核对镜头切点、主体外观与参考图；截图可能保留字幕、水印、遮挡和背景。',
             '- 选择目标生成平台后确认支持的时长、分辨率、比例及参考图能力；本计划没有猜测原片模型或种子。',
             '- 文生视频使用完整画面提示词；图生视频输入参考图和运动提示词。两条路线择一，不要求依次生成。',
             '- 目标时长是原片复现需求，不代表平台支持该长度。长镜头按动作停顿或转场人工拆分，并重写每段提示词；不要机械截断动作。', '',
             '## 2. 逐镜头生成', '']
    for shot in plan['shots']:
        a, b = shot['source_range_ms']
        lines += [f"### {shot['id']} · {a/1000:.3f}–{b/1000:.3f} 秒", '',
                  f"全镜头目标时长：{shot['target_duration_ms']/1000:.3f} 秒。",
                  f"参考帧：{shot['reference']['source_ms']/1000:.3f} 秒；{shot['reference']['export_path'] or '尚无可导出的有效参考帧'}。", '']
        for subject in shot['subjects']:
            lines.append(f"- 已确认主体 {subject['id']}：{subject.get('description', '')}")
        for route, title in (('text_to_video', '文生视频'), ('image_to_video', '图生视频')):
            item = shot['routes'][route]
            duration = item['duration_ms']
            lines += ['', f"#### {title} · 目标 {duration/1000:.3f} 秒" if duration is not None else f'#### {title} · 参考帧无效', '',
                      item['prompt_zh'] or '待生成中文提示词', '', item['prompt_en'] or '待生成英文提示词', '']
        lines += ['**动作证据（原片绝对时间，重叠窗口可能重复）**', '']
        for beat in shot['evidence_stages']:
            lines.append(f"- {beat['id']}：{beat.get('start_ms')}–{beat.get('end_ms')} ms；{beat.get('action', '')}；证据帧 {beat.get('evidence_ms', [])}")
        if not shot['evidence_stages']:
            lines.append('暂无结构化动作证据，不代表画面静止。')
        if shot['user_overrides']:
            import json
            lines += ['', '**用户修正（需与当前提示词核对）**', '', json.dumps(shot['user_overrides'], ensure_ascii=False)]
        lines += ['', '**生成前复核**', '']
        lines.extend('- ' + item for item in shot['review_items'])
        if not shot['review_items']:
            lines.append('- 未发现程序记录的待办；仍需人工确认主体、动作方向、运镜与时间顺序。')
    lines += ['', '## 3. 剪辑、声音与验收', '',
              '- 按原片镜头顺序拼接，核对动作接点、转场、构图与色彩；字幕和图形叠加单独制作。',
              '- 若生成时长超过目标，先确认动作完成再裁切；不要默认变速，因为它会改变动作与声音。',
              '- 对白转写没有时间戳，须人工对齐；音乐、音效及配音需另行准备，不能从视觉结果推定。']
    if plan['audio']:
        lines += ['', '全片对白（未经镜头对齐）：', '', plan['audio'].get('edited_text', plan['audio'].get('text', ''))]
    lines += ['', '- 对照原片逐项检查：主体外观、服装道具、环境、动作顺序和方向、运镜、时长、声音。',
              '- 每轮只调整一个主要差异，记录目标模型版本、输入参考图、提示词、实际参数及生成结果，避免无法追溯。',
              '- 原始提示词、生成模型、种子和内部参数均未知；不得以相似画面宣称已恢复原始配置。', '']
    return '\n'.join(lines)
