"""Local HTTP interface for the existing video analysis engine."""
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import clips
import engine
import provider_settings

app = FastAPI(title='Video Reversal API')
lock = threading.RLock()


@app.middleware('http')
async def local_origin(request: Request, call_next):
    origin = request.headers.get('origin')
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1') or parsed.port not in (5173, 8000):
            return __import__('fastapi').responses.JSONResponse(status_code=403, content={'detail': '不允许的来源'})
    return await call_next(request)


def load(job_id):
    try:
        return engine.load(job_id)
    except (ValueError, OSError):
        raise HTTPException(404, '任务不存在')


def public(job):
    """构造 API 展示数据，不把兼容结果写回任务文件。"""
    items=copy.deepcopy(clips.ensure(job))
    for clip in items:
        shot=matching_shot(job,clip)
        if not clip.get('analysis') and shot and shot.get('prompts') and shot.get('observation'):
            clip['analysis']=copy.deepcopy(shot)
            clip['analysis_source']='shot'
            clip['status']=shot.get('status','pending')
    return {'id': job['id'], 'video': job['video'], 'shots': job['shots'], 'clips': items, 'api_task': job.get('api_task'), 'subjects':job.get('subjects',{}), 'audio_transcript':job.get('audio_transcript')}


def matching_shot(job,clip):
    # 只有片段范围与镜头完全一致时，才可复用镜头分析结果。
    return next((s for s in job['shots'] if s['id']==clip['shot_id'] and s['start_ms']==clip['start_ms'] and s['end_ms']==clip['end_ms']),None)


@app.post('/api/jobs')
def create_job(video: UploadFile = File(...)):
    suffix = Path(video.filename or '').suffix.lower()
    if suffix not in ('.mp4', '.mov'):
        raise HTTPException(400, '请选择 MP4 或 MOV 视频')
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ('upload' + suffix)
            size = 0
            with path.open('wb') as target:
                while chunk := video.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > engine.MAX_SIZE:
                        raise HTTPException(413, '视频超过 200 MB')
                    target.write(chunk)
            with lock:
                job = engine.create(str(path))
                clips.ensure(job)
                engine.save(job)
                return public(job)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        video.file.close()


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with lock:
        job = load(job_id)
        if job_id in running: return public(job)
        if job.get('api_task', {}).get('status') == 'processing' and job_id not in running:
            job['api_task']['status'] = 'interrupted'
            job['api_task']['error'] = '服务已重启，请重新发起分析。'
        result = public(job)
        engine.save(job)
        return result


@app.get('/api/jobs/{job_id}/video')
def video_file(job_id: str):
    job = load(job_id)
    name = job['video']['file']
    if Path(name).name != name or Path(name).suffix.lower() not in ('.mp4', '.mov'):
        raise HTTPException(400, '无效视频文件')
    return FileResponse(engine.DATA / job_id / name)


@app.get('/api/jobs/{job_id}/clips/{clip_id}/evidence')
def evidence(job_id: str, clip_id: str):
    with lock:
        job = load(job_id)
        try:
            result = clips.evidence(job, clip_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        for item in [result['source_head'], result['source_tail'], *result['head_frames'], *result['tail_frames']]:
            item.pop('path', None)
            item['url'] = f"/api/jobs/{job_id}/frames/{item['index']}"
        return result


@app.get('/api/jobs/{job_id}/frames/{index}')
def frame_file(job_id: str, index: int):
    load(job_id)
    if index < 0: raise HTTPException(404, '帧不存在')
    path = engine.DATA / job_id / 'clip_frames' / f'{index:08d}.jpg'
    if not path.is_file(): raise HTTPException(404, '帧不存在')
    return FileResponse(path, media_type='image/jpeg')


class DirectorNote(BaseModel):
    text: str


@app.put('/api/jobs/{job_id}/clips/{clip_id}/director-note')
def save_director_note(job_id: str, clip_id: str, body: DirectorNote):
    with lock:
        check_idle(job_id)
        job = load(job_id)
        find_clip(job, clip_id)['director_note'] = body.text
        engine.save(job)
        return public(job)


class Split(BaseModel):
    time_ms: int


@app.post('/api/jobs/{job_id}/clips/{clip_id}/split')
def split_clip(job_id: str, clip_id: str, body: Split):
    with lock:
        check_idle(job_id)
        job = load(job_id)
        try:
            clips.split(job, clip_id, body.time_ms)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        engine.save(job)
        return public(job)

# 单进程本地任务；实际模型密钥只存在工作线程内。
import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
import cases
import clip_context

executor = ThreadPoolExecutor(max_workers=1)
running = {}


def find_clip(job, clip_id):
    clip = next((c for c in clips.ensure(job) if c['id'] == clip_id), None)
    if clip is None: raise HTTPException(404, '片段不存在')
    return clip


def check_idle(job_id):
    if job_id in running: raise HTTPException(409, '任务正在分析，请等待完成')


class AnalysisRequest(BaseModel):
    provider: str = 'Gemini'
    model: str = ''
    url: str = ''
    key: str = ''
    strategy: str = '等间隔'
    image_mode: str = '单帧'
    use_saved: bool = False


class PromptEdit(BaseModel):
    text_zh: str
    text_en: str
    image_zh: str
    image_en: str


class CaseTitle(BaseModel):
    title: str

class SettingsEdit(BaseModel):
    active: str
    name: str
    model: str
    url: str
    key: str = ''

@app.get('/api/settings')
def get_settings():
    return provider_settings.public()

@app.put('/api/settings')
def put_settings(body: SettingsEdit):
    try:
        with lock: return provider_settings.save(body.active,body.name,body.model,body.url,body.key)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

def analysis_connection(body):
    if body.use_saved: return provider_settings.resolve(body.provider)
    return engine.connection(body.provider,body.model,body.url,body.key)


def analyze_clip(job_id, clip_id, task_id, config):
    """基于任务快照分析；仅当片段期间未变化时才提交结果。"""
    temp = None
    source = None
    try:
        with lock:
            job = load(job_id)
            clip = find_clip(job, clip_id)
            source = copy.deepcopy(clip)
            original = next(s for s in job['shots'] if s['id'] == clip['shot_id'])
            context = clip_context.for_clip(job, clips.ensure(job), clip_id)
            evidence = clips.evidence(job, clip_id)
            marks = sorted({int(f['time_ms']) for f in evidence['head_frames'] + evidence['tail_frames']
                            if clip['start_ms'] <= int(f['time_ms']) < clip['end_ms']})
            shot = copy.deepcopy(clip.get('analysis') or {})
            shot.update({'id': clip_id, 'start_ms': clip['start_ms'], 'end_ms': clip['end_ms'],
                         'reference_ms': shot.get('reference_ms', clip['start_ms']), 'status': 'pending',
                         'subject_ids': original.get('subject_ids', []), 'manual_frames': marks})
            shot.setdefault('overrides', {})['已确认的片段上下文'] = context
            temp = copy.deepcopy(job)
            temp['_transient'] = True
            temp['shots'] = [shot]
            temp['usage'] = []
        engine.analyze(temp, 0, config, resume=True)
        with lock:
            current = load(job_id)
            target = find_clip(current, clip_id)
            # 防止并发拆分、合并或编辑后，把过期分析结果写回片段。
            if target != source:
                raise ValueError('片段在分析期间发生变化，结果未覆盖现有内容。')
            target['analysis'] = temp['shots'][0]
            target['status'] = target['analysis']['status']
            current.setdefault('usage', []).extend(temp.get('usage', []))
            current['api_task'] = {'id': task_id, 'clip_id': clip_id, 'status': target['status'], 'error': target['analysis'].get('error')}
            engine.save(current)
    except Exception as exc:
        with lock:
            try:
                current = load(job_id)
                if temp is not None and source is not None:
                    target = find_clip(current, clip_id)
                    # 保留已完成的模型窗口，以便失败后继续分析。
                    if target == source:
                        target['analysis'] = temp['shots'][0]
                        target['status'] = 'failed'
                        current.setdefault('usage', []).extend(temp.get('usage', []))
                message = str(exc)
                if config.get('key'): message = message.replace(config['key'], '[已隐藏]')
                current['api_task'] = {'id': task_id, 'clip_id': clip_id, 'status': 'failed', 'error': message[:500]}
                engine.save(current)
            except Exception:
                pass
    finally:
        with lock:
            running.pop(job_id, None)


@app.get('/api/providers')
def providers():
    return [{'name': name, 'model': values[1]} for name, values in engine.PRESETS.items()]


@app.post('/api/jobs/{job_id}/clips/{clip_id}/analysis', status_code=202)
def start_analysis(job_id: str, clip_id: str, body: AnalysisRequest):
    with lock:
        check_idle(job_id)
        job = load(job_id)
        find_clip(job, clip_id)
        if body.strategy not in ('等间隔', '快速（1 FPS）', '变化驱动', 'Prompt 专用') or body.image_mode not in ('单帧', '拼图'):
            raise HTTPException(400, '无效采样策略或发送方式')
        try:
            config = {**analysis_connection(body),
                      'sampling_strategy': body.strategy, 'image_mode': body.image_mode}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        task_id = uuid.uuid4().hex
        running[job_id] = task_id
        job['api_task'] = {'id': task_id, 'clip_id': clip_id, 'status': 'processing', 'error': None}
        engine.save(job)
        executor.submit(analyze_clip, job_id, clip_id, task_id, config)
        return job['api_task']


@app.put('/api/jobs/{job_id}/clips/{clip_id}/prompts')
def save_clip_prompts(job_id: str, clip_id: str, body: PromptEdit):
    with lock:
        check_idle(job_id)
        job = load(job_id)
        clip = find_clip(job, clip_id)
        analysis = clip.get('analysis')
        if not analysis:
            shot=matching_shot(job,clip)
            if not shot or not shot.get('observation') or not shot.get('prompts'): raise HTTPException(400, '请先分析片段')
            analysis=copy.deepcopy(shot)
            analysis['id']=clip_id
            analysis['source_shot_id']=shot['id']
            clip['analysis']=analysis
            clip['status']=shot.get('status','pending')
        prompts = body.model_dump()
        previous = analysis.get('prompts') or {}
        manual = analysis.setdefault('manual_prompts', {})
        for key, value in prompts.items():
            if value != previous.get(key, ''): manual[key] = value
        analysis['prompts'] = prompts
        analysis['prompt_review'] = engine.review_prompts(prompts)
        if manual and analysis.get('coverage_review'):
            review = analysis['coverage_review']
            review['missing_ids'] = sorted(set(review.get('missing_ids', []) + review.get('reported_ids', [])))
            review['note'] = '提示词含人工修改，请重新核对动作阶段。'
        engine.save(job)
        return public(job)


@app.get('/api/cases')
def list_cases():
    return cases.list_cases()


@app.post('/api/jobs/{job_id}/cases')
def save_case(job_id: str, body: CaseTitle):
    with lock:
        check_idle(job_id)
        try: return cases.save_case(load(job_id), body.title)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@app.post('/api/cases/{case_id}/restore')
def restore_case(case_id: str):
    with lock:
        try: job, _ = cases.restore_case(case_id)
        except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc
        return public(job)


@app.get('/api/jobs/{job_id}/export')
def export_job(job_id: str):
    with lock:
        check_idle(job_id)
        return FileResponse(engine.export(load(job_id)), filename='video-reversal.zip', media_type='application/zip')

import audio
import subjects
import reconstruction

class ContextBody(BaseModel):
    context: dict

class SubjectBody(BaseModel):
    identifier: str
    description: str = ''

class AudioBody(BaseModel):
    text: str

class SamplingBody(BaseModel):
    strategy: str = '等间隔'
    time_ms: int | None = None


@app.post('/api/jobs/{job_id}/clips/{clip_id}/merge')
def merge_clip(job_id: str, clip_id: str):
    with lock:
        check_idle(job_id)
        job=load(job_id); items=clips.ensure(job)
        index=next((i for i,c in enumerate(items) if c['id']==clip_id),None)
        if index is None or index+1>=len(items): raise HTTPException(400,'没有可合并的相邻片段')
        try: clips.merge(job,clip_id,items[index+1]['id'])
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        engine.save(job); return public(job)


@app.get('/api/jobs/{job_id}/context')
def get_context(job_id: str):
    with lock:
        job=load(job_id); return clip_context.read(job,clips.ensure(job))


@app.put('/api/jobs/{job_id}/context')
def put_context(job_id: str, body: ContextBody):
    with lock:
        check_idle(job_id);job=load(job_id)
        try: clip_context.save(job,clips.ensure(job),body.context)
        except (ValueError,TypeError) as exc: raise HTTPException(400,str(exc)) from exc
        for clip in job['clips']:
            if clip.get('analysis'): clip['status']='edited'
        engine.save(job);return public(job)


def shot_index(job, shot_id):
    i=next((i for i,s in enumerate(job['shots']) if s['id']==shot_id),None)
    if i is None: raise HTTPException(404,'镜头不存在')
    return i


@app.post('/api/jobs/{job_id}/shots/{shot_id}/subject')
def link_subject(job_id: str, shot_id: str, body: SubjectBody):
    with lock:
        check_idle(job_id);job=load(job_id)
        try: subjects.link(job,shot_index(job,shot_id),body.identifier,body.description)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        return public(job)


@app.post('/api/jobs/{job_id}/shots/{shot_id}/subject/unlink')
def unlink_subject(job_id: str, shot_id: str, body: SubjectBody):
    with lock:
        check_idle(job_id);job=load(job_id)
        try: subjects.unlink(job,shot_index(job,shot_id),body.identifier.strip())
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        return public(job)


@app.post('/api/jobs/{job_id}/shots/{shot_id}/sampling')
def sample_shot(job_id: str, shot_id: str, body: SamplingBody):
    with lock:
        check_idle(job_id);job=load(job_id); i=shot_index(job,shot_id);shot=job['shots'][i]
        if body.strategy not in ('等间隔','快速（1 FPS）','变化驱动','Prompt 专用'): raise HTTPException(400,'无效抽帧策略')
        if body.time_ms is not None:
            if not shot['start_ms']<=body.time_ms<shot['end_ms']: raise HTTPException(400,'补帧时间必须在镜头内')
            manual=shot.setdefault('manual_frames',[])
            if body.time_ms not in manual: manual.append(body.time_ms);manual.sort()
            if shot.get('observation'): shot['status']='edited'
        try:
            plan=engine.sample_plan(job,i,body.strategy)
            for row in plan: engine.frame(job,row['ms'])
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        shot['sample_preview']={'strategy':body.strategy,'frames':plan};engine.save(job)
        return [{**row,'url':f"/api/jobs/{job_id}/sample/{row['ms']}"} for row in plan]


@app.get('/api/jobs/{job_id}/sample/{ms}')
def sample_frame(job_id: str, ms: int):
    job=load(job_id)
    if not 0<=ms<job['video']['duration_ms']: raise HTTPException(400,'时间无效')
    return FileResponse(engine.frame(job,ms),media_type='image/jpeg')


@app.get('/api/jobs/{job_id}/plan')
def plan(job_id: str):
    return {'markdown':reconstruction.markdown(reconstruction.build(load(job_id)))}


@app.put('/api/jobs/{job_id}/audio')
def edit_audio(job_id: str, body: AudioBody):
    with lock:
        check_idle(job_id); job=load(job_id)
        if not job.get('audio_transcript'): raise HTTPException(400,'请先完成转写')
        job['audio_transcript']['edited_text']=body.text;engine.save(job);return public(job)


def run_other(job_id, task_id, operation, config, shot_id):
    try:
        job=load(job_id)
        errors=[]
        if operation=='connection':
            engine.call_model({'provenance':{},'usage':[]},'返回四个字段，值都填写 ok。',[],engine.Prompts,config)
        elif operation=='audio':
            audio.transcribe(job,config['key'])
        else:
            i=shot_index(job,shot_id) if operation != 'all' else None
            if operation=='shot': engine.analyze(job,i,config,resume=True)
            elif operation=='generate': engine.generate(job,i,config)
            elif operation=='compare':
                results=engine.compare(job,i,config,('等间隔','变化驱动','Prompt 专用'))
                for name,result in results.items():
                    if result.get('status') not in ('complete','complete_partial'):
                        errors.append(f"{name}：{result.get('error','分析失败')}")
                    elif result.get('status')=='complete_partial': errors.append(f'{name}：部分结果')
            elif operation=='all':
                for i,shot in enumerate(job['shots']):
                    if shot.get('status')!='complete' and not shot.get('skip_bulk'):
                        try: engine.analyze(job,i,config,resume=True)
                        except Exception as exc: errors.append(f"{shot['id']}：{exc}")
        with lock:
            current=load(job_id)
            partial=any(s.get('status')=='complete_partial' for s in current['shots']) if operation in ('shot','all') else False
            message='；'.join(errors)[:500]
            if config.get('key'): message=message.replace(config['key'],'[已隐藏]')
            current['api_task']={'id':task_id,'status':'complete_partial' if partial or errors else 'complete','error':message or ('部分动作补看未完成，请核对并续跑。' if partial else None)};engine.save(current)
    except Exception as exc:
        with lock:
            try:
                current=load(job_id);message=str(exc)
                if config.get('key'): message=message.replace(config['key'],'[已隐藏]')
                current['api_task']={'id':task_id,'status':'failed','error':message[:500]};engine.save(current)
            except Exception: pass
    finally:
        with lock: running.pop(job_id,None)


@app.post('/api/jobs/{job_id}/operations/{operation}',status_code=202)
def start_other(job_id: str, operation: str, body: AnalysisRequest, shot_id: str = ''):
    if operation not in ('shot','all','generate','compare','audio','connection'): raise HTTPException(404,'操作不存在')
    if body.strategy not in ('等间隔','快速（1 FPS）','变化驱动','Prompt 专用') or body.image_mode not in ('单帧','拼图'): raise HTTPException(400,'无效采样策略或发送方式')
    with lock:
        check_idle(job_id);job=load(job_id)
        if operation in ('shot','generate','compare'): shot_index(job,shot_id)
        if operation=='audio' and body.provider!='硅基流动': raise HTTPException(400,'对白转写需要选择硅基流动')
        try: config={**analysis_connection(body),'sampling_strategy':body.strategy,'image_mode':body.image_mode}
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        tid=uuid.uuid4().hex;running[job_id]=tid
        job['api_task']={'id':tid,'status':'processing','error':None};engine.save(job)
        executor.submit(run_other,job_id,tid,operation,config,shot_id)
        return job['api_task']


@app.post('/api/jobs/{job_id}/black-screen-detection')
def detect_black_screens(job_id: str):
    import black_frames
    with lock:
        check_idle(job_id); job = load(job_id)
        try: black_frames.detect(job)
        except ValueError as exc: raise HTTPException(400, str(exc)) from exc
        engine.save(job)
        return public(job)


class BulkSkipEdit(BaseModel):
    skip: bool


@app.put('/api/jobs/{job_id}/shots/{shot_id}/bulk-skip')
def edit_bulk_skip(job_id: str, shot_id: str, body: BulkSkipEdit):
    with lock:
        check_idle(job_id); job = load(job_id)
        job['shots'][shot_index(job, shot_id)]['skip_bulk'] = body.skip
        engine.save(job)
        return public(job)

class ShotEdit(BaseModel):
    start_ms: int
    end_ms: int
    reference_ms: int
    overrides: dict = {}
    prompts: PromptEdit


@app.put('/api/jobs/{job_id}/shots/{shot_id}')
def edit_shot(job_id: str, shot_id: str, body: ShotEdit):
    import json
    import app as legacy
    with lock:
        check_idle(job_id);job=load(job_id);i=shot_index(job,shot_id);p=body.prompts
        try:
            legacy.save_edits(job_id,str(i),json.dumps(body.overrides,ensure_ascii=False),body.start_ms/1000,body.end_ms/1000,body.reference_ms/1000,p.text_zh,p.text_en,p.image_zh,p.image_en)
        except Exception as exc: raise HTTPException(400,str(exc)) from exc
        return public(load(job_id))

@app.get('/api/jobs/{job_id}/shots/{shot_id}/reference')
def reference_preview(job_id: str, shot_id: str, time_ms: int):
    with lock:
        job=load(job_id);shot=job['shots'][shot_index(job,shot_id)]
        if not shot['start_ms']<=time_ms<shot['end_ms']: raise HTTPException(400,'参考帧时间必须在镜头内')
        return FileResponse(engine.reference_frame(job,time_ms),media_type='image/jpeg')
