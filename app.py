import copy
import json
import os
from pathlib import Path
import gradio as gr
import engine
import audio
import subjects
import cases
import clips
import clip_context
from datetime import datetime
from zoneinfo import ZoneInfo

CSS = '''
.gradio-container {max-width: 1440px !important; font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;}
header {padding: 28px 0 12px; border-bottom: 1px solid #dededb; margin-bottom: 20px;}
header .eyebrow {font-size: 11px; letter-spacing: .16em; color: #697166; font-weight: 700;}
header h1 {font-size: 36px; letter-spacing: -.04em; margin: 8px 0 4px; color: #1b2521;}
header p {color: #68716a; margin: 0 0 18px; font-size: 14px;}
.note {color: #69716b; font-size: 13px;}
'''

def labels(job):
    names={'pending':'待分析','processing':'处理中','analyzed':'已分析','complete':'已完成','complete_partial':'部分结果','failed':'未完成','edited':'已修改'}
    return [(f"{s['id']}  ·  {s['start_ms']/1000:.2f}–{s['end_ms']/1000:.2f}s  ·  {names.get(s['status'],s['status'])}",str(i)) for i,s in enumerate(job['shots'])]

def make_job(path):
    if not path: raise gr.Error('请先上传视频。')
    try:
        job=engine.create(path)
        return job['id'], gr.update(choices=labels(job),value='0'), f"已识别 {len(job['shots'])} 个镜头。任务 ID：{job['id']}。点击“分析全部镜头”继续。", job['id']
    except Exception as e: raise gr.Error(str(e))

def restore(job_id):
    try:
        job=engine.load(job_id.strip())
        return job['id'],gr.update(choices=labels(job),value='0'),f"已恢复任务 {job['id']}。",job['id'],str(engine.DATA/job['id']/job['video']['file'])
    except Exception as e: raise gr.Error('无法恢复任务：'+str(e))

def comparison_text(result):
    if not result: return '尚未运行对比。'
    if result.get('status') not in ('complete','complete_partial'): return '失败：'+result.get('error','未知错误')
    p=result['prompts']
    usages=result.get('usage',[])
    total=sum((item.get('tokens',{}).get('total_tokens') or item.get('tokens',{}).get('total_token_count') or 0) for item in usages)
    tokens=f'{total} tokens' if total else '未提供用量'
    camera_warnings=(result.get('observation') or {}).get('camera_review',[])
    warning=f'｜运镜待核对：{len(camera_warnings)} 处' if camera_warnings else ''
    if result.get('status')=='complete_partial': warning+='｜部分补看超时，结果不完整'
    if result.get('prompt_review'): warning+=f"｜动作方向待核对：{len(result['prompt_review'])} 处"
    return (f"采样帧：{result['sampling_frame_count']}｜耗时：{result['elapsed_s']} 秒｜用量：{tokens}{warning}\n"
            + '\n'.join(f'{label}：\n{p.get(key, "")}' for key,label in [('text_zh','文生视频 · 中文'),('text_en','文生视频 · English'),('image_zh','图生视频 · 中文'),('image_en','图生视频 · English')]))


def show(job_id,index):
    if not job_id or index is None: return [None,'','',0,0,0,'','','','',[], '暂无采样预览。', '尚未运行对比。','尚未运行对比。','尚未运行对比。']
    job=engine.load(job_id);s=job['shots'][int(index)]; p=s.get('prompts') or {}
    ref=None
    if s.get('observation') or s.get('reference_user_selected'):
        ref=str(engine.reference_frame(job,s['reference_ms']))
    analysis=json.dumps({'observation':s.get('observation'), 'coverage_review':s.get('coverage_review')},ensure_ascii=False,indent=2)
    overrides=json.dumps(s.get('overrides') or {},ensure_ascii=False,indent=2)
    preview=s.get('sample_preview') or {}
    frames=preview.get('frames',[])
    gallery=[(str(engine.frame(job,f['ms'])),f"{f['ms']/1000:.3f}s · {', '.join(f['reasons'])}") for f in frames]
    summary=f"{preview.get('strategy','')} · {len(frames)} 帧（与原视频时间对应）" if frames else '暂无采样预览。'
    camera_warnings=(s.get('observation') or {}).get('camera_review',[])
    if camera_warnings: summary+=f'  \n⚠️ 有 {len(camera_warnings)} 处画面整体位移与镜头描述可能冲突，请在分析证据中核对。'
    if s.get('prompt_review'): summary+=f"  \n⚠️ 有 {len(s['prompt_review'])} 处动作方向表述不明确，请对照画面核对。"
    if (s.get('observation') or {}).get('incomplete_review'): summary+='  \n⚠️ 部分动作补看未完成；当前 Prompt 是部分结果，可点击“继续上次分析”。'
    if s.get('status')=='failed' and s.get('error'): summary+='  \n⚠️ '+s['error']
    comparisons=s.get('comparisons') or {}
    return [ref,analysis,overrides,s['start_ms']/1000,s['end_ms']/1000,s['reference_ms']/1000,p.get('text_zh',''),p.get('text_en',''),p.get('image_zh',''),p.get('image_en',''),gallery,summary,comparison_text(comparisons.get('等间隔')),comparison_text(comparisons.get('变化驱动')),comparison_text(comparisons.get('Prompt 专用'))]

def all_shots(job_id, provider, model, url, key, strategy, image_mode, progress=gr.Progress()):
    if not job_id: raise gr.Error('请先创建任务。')
    try: config={**engine.connection(provider,model,url,key),'sampling_strategy':strategy,'image_mode':image_mode}
    except ValueError as e: raise gr.Error(str(e))
    job=engine.load(job_id); errors=[]; partial=[]
    for i in progress.tqdm(range(len(job['shots'])),desc='逐镜头分析'):
        if job['shots'][i]['status']=='complete': continue
        try: engine.analyze(job,i,config,resume=True)
        except Exception as e: errors.append(f"{job['shots'][i]['id']}：{e}")
        else:
            if job['shots'][i]['status']=='complete_partial': partial.append(job['shots'][i]['id'])
    return gr.update(choices=labels(job),value='0'), '分析完成。' + (' 部分镜头失败：'+'；'.join(errors) if errors else '') + (' 部分补看超时，结果需核对并继续分析：'+'、'.join(partial) if partial else '')

def one(job_id,index,provider,model,url,key,strategy,image_mode):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    job=engine.load(job_id)
    try: engine.analyze(job,int(index),{**engine.connection(provider,model,url,key),'sampling_strategy':strategy,'image_mode':image_mode})
    except Exception as e: raise gr.Error(str(e))
    return gr.update(choices=labels(job),value=index),job['shots'][int(index)].get('error') or '镜头分析完成；人工修正字段已保留。'


def resume_shot(job_id,index,provider,model,url,key,strategy,image_mode):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    job=engine.load(job_id)
    try: engine.analyze(job,int(index),{**engine.connection(provider,model,url,key),'sampling_strategy':strategy,'image_mode':image_mode},resume=True)
    except Exception as e: raise gr.Error(str(e))
    return gr.update(choices=labels(job),value=index),job['shots'][int(index)].get('error') or '分析已继续完成；已复用匹配的成功窗口。'

def save_edits(job_id,index,override_text,start,end,reference,tz,te,iz,ie):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id);i=int(index);s=job['shots'][i]
        override=json.loads(override_text or '{}')
        if not isinstance(override,dict): raise ValueError('人工修正必须是 JSON 对象。')
        a,b,r=round(float(start)*1000),round(float(end)*1000),round(float(reference)*1000)
        left=job['shots'][i-1]['end_ms'] if i else 0
        right=job['shots'][i+1]['start_ms'] if i+1<len(job['shots']) else job['video']['duration_ms']
        # Boundary changes are shared with the neighboring shot, so no source interval is lost.
        if a!=s['start_ms'] or b!=s['end_ms']:
            if not (0<=a<b<=job['video']['duration_ms']) or (i and a<=job['shots'][i-1]['start_ms']) or (i+1<len(job['shots']) and b>=job['shots'][i+1]['end_ms']): raise ValueError('切点必须位于相邻镜头范围内。')
            if i==0 and a!=0 or i==len(job['shots'])-1 and b!=job['video']['duration_ms']: raise ValueError('视频首尾边界不可调整。')
            previous_prompts=s.get('prompts') or {}
            pending={'text_zh':tz,'text_en':te,'image_zh':iz,'image_en':ie}
            manual=s.setdefault('manual_prompts',{})
            for field,value in pending.items():
                if value != previous_prompts.get(field,''): manual[field]=value
            affected=[i]
            if i and a!=s['start_ms']: job['shots'][i-1]['end_ms']=a;affected.append(i-1)
            if i+1<len(job['shots']) and b!=s['end_ms']: job['shots'][i+1]['start_ms']=b;affected.append(i+1)
            s['start_ms']=a;s['end_ms']=b
            for j in affected:
                x=job['shots'][j];x['observation']=None;x['prompts']=None;x['status']='pending';x['reference_ms']=x['start_ms'];x['error']=None
                x.pop('sample_preview',None); x.pop('comparisons',None); x.pop('reference_user_selected',None); x.pop('analysis_checkpoint',None); x.pop('prompt_review',None)
                if 'clips' in job: clips.reset_shot(job,x['id'])
            s['overrides']=override;engine.save(job)
            return gr.update(choices=labels(job),value=index),'切点已更新；受影响镜头需重新分析，人工修正已保留。'
        if not a<=r<b: raise ValueError('参考帧时间必须在镜头内。')
        changed=(r!=s['reference_ms'] or override!=s.get('overrides',{}))
        s['reference_ms']=r;s['reference_user_selected']=True;s['overrides']=override
        new_prompts={'text_zh':tz,'text_en':te,'image_zh':iz,'image_en':ie}
        previous=s.get('prompts') or {}
        manual=s.setdefault('manual_prompts', {})
        for key, value in new_prompts.items():
            if value != previous.get(key, ''): manual[key]=value
        s['prompts']=new_prompts
        s['prompt_review']=engine.review_prompts(new_prompts)
        if any(name in manual for name in ('text_zh','text_en')) and s.get('coverage_review'):
            review=s['coverage_review']
            review['missing_ids']=sorted(set(review.get('missing_ids',[])+review.get('reported_ids',[])))
            review['note']='提示词含人工修改，请重新核对全部动作阶段；模型自报不能证明内容真实。'
        if changed: s['status']='edited'
        engine.save(job)
        return gr.update(choices=labels(job),value=index),('已保存。参考帧或人工修正已改变，请重新生成 Prompt。' if changed else '已保存编辑。')
    except Exception as e: raise gr.Error(str(e))


def select_reference(job_id,index,second):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id);shot=job['shots'][int(index)]
        ms=round(float(second)*1000)
        if not shot['start_ms']<=ms<shot['end_ms']: raise ValueError('参考帧时间必须在镜头内。')
        path=engine.reference_frame(job,ms)
        if ms!=shot.get('reference_ms') and shot.get('observation'): shot['status']='edited'
        shot['reference_ms']=ms;shot['reference_user_selected']=True
        engine.save(job)
        return str(path), '参考帧已保存。'+engine.reference_note(shot)+' 更换后请重新生成提示词。'
    except Exception as e: raise gr.Error(str(e))

def regen(job_id,index,provider,model,url,key):
    try:
        job=engine.load(job_id);engine.generate(job,int(index),engine.connection(provider,model,url,key));return gr.update(choices=labels(job),value=index),(job['shots'][int(index)].get('error') or ('Prompt 已生成；存在待核对动作阶段，请查看分析证据。' if job['shots'][int(index)].get('coverage_review',{}).get('missing_ids') else 'Prompt 已重新生成；模型自报覆盖并非准确性保证。'))
    except Exception as e: raise gr.Error('生成 Prompt：'+str(e))


def preview_sampling(job_id,index,strategy):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id); shot=job['shots'][int(index)]
        shot['sample_preview']={'strategy':strategy,'frames':engine.sample_plan(job,int(index),strategy)}
        engine.save(job)
        return '采样预览已更新，尚未调用模型。'
    except Exception as e: raise gr.Error(str(e))


def add_frame(job_id,index,second,strategy):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id);shot=job['shots'][int(index)]
        ms=round(float(second)*1000)
        if not shot['start_ms'] <= ms < shot['end_ms']: raise ValueError('补帧时间必须位于当前镜头内。')
        manual=shot.setdefault('manual_frames',[])
        if ms not in manual: manual.append(ms);manual.sort()
        shot['sample_preview']={'strategy':strategy,'frames':engine.sample_plan(job,int(index),strategy)}
        if shot.get('observation'): shot['status']='edited'
        engine.save(job)
        return '补帧已保存。重新分析镜头时将包含该帧；已有分析结果保持不变。'
    except Exception as e: raise gr.Error(str(e))


def sampling_view(job_id,index):
    values=show(job_id,index)
    return values[10],values[11]


def compare_shot(job_id,index,provider,model,url,key):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id)
        result=engine.compare(job,int(index),engine.connection(provider,model,url,key),('等间隔','变化驱动','Prompt 专用'))
        succeeded=sum(item['status']=='complete' for item in result.values())
        partial=sum(item['status']=='complete_partial' for item in result.values())
        return f'对比完成：{succeeded}/3 个策略完整，{partial} 个为部分结果。原分析未覆盖；各策略用量分别保存在任务中。', comparison_text(result.get('等间隔')), comparison_text(result.get('变化驱动')), comparison_text(result.get('Prompt 专用'))
    except Exception as e: raise gr.Error(str(e))


def show_audio(job_id):
    if not job_id: return ''
    job=engine.load(job_id);t=job.get('audio_transcript') or {}
    return t.get('edited_text',t.get('text',''))


def transcribe_audio(job_id,provider,key):
    if not job_id: raise gr.Error('请先创建任务。')
    if provider!='硅基流动': raise gr.Error('请先在模型设置选择硅基流动，再使用对白转写。')
    try:
        job=engine.load(job_id);result=audio.transcribe(job,key)
        status='对白转写完成：按全片提供，接口没有时间戳，请人工核对。'
        if not result['text']: status='转写结果为空；这不等于确认没有对白，请人工核对。'
        if 'edited_text' in result: status+=' 已保留此前人工修订。'
        return result.get('edited_text',result['text']),status
    except Exception as e: raise gr.Error(str(e))


def save_audio(job_id,text):
    if not job_id: raise gr.Error('请先创建任务。')
    job=engine.load(job_id)
    if not job.get('audio_transcript'): raise gr.Error('请先完成转写。')
    job['audio_transcript']['edited_text']=text
    engine.save(job)
    return '对白修订已保存，原始转写仍保留。'


def subject_view(job_id,index):
    if not job_id or index is None: return '{}',''
    job=engine.load(job_id)
    return json.dumps(job.get('subjects',{}),ensure_ascii=False,indent=2), ', '.join(job['shots'][int(index)].get('subject_ids',[]))


def link_subject(job_id,index,identifier,description):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id);subjects.link(job,int(index),identifier,description)
        return '主体已保存并关联；受影响的已完成镜头需要重新生成 Prompt。', *subject_view(job_id,index)
    except Exception as e: raise gr.Error(str(e))


def unlink_subject(job_id,index,identifier):
    if not job_id or index is None: raise gr.Error('请先选择镜头。')
    try:
        job=engine.load(job_id);subjects.unlink(job,int(index),identifier.strip())
        return '已解除当前镜头的关联；主体资料及其他镜头保留。', *subject_view(job_id,index)
    except Exception as e: raise gr.Error(str(e))

def preset(provider):
    url, model, _ = engine.PRESETS[provider]
    return model, url, ''

def test_connection(provider,model,url,key):
    try:
        config=engine.connection(provider,model,url,key)
        engine.call_model({'provenance':{}, 'usage':[]}, '返回四个字段，值都填写 ok。', [], engine.Prompts, config)
        return '连接成功，文本与 JSON 输出正常。图片理解能力需在分析视频时验证。'
    except Exception as e:
        raise gr.Error(str(e))


def case_choices():
    return [(f"{datetime.fromtimestamp(item['created_at'],ZoneInfo('Asia/Shanghai')):%Y-%m-%d %H:%M:%S} · {item['title']} · {item['id'][:6]}",item['id']) for item in cases.list_cases()]


def case_snapshot(job_id,index,override_text,start,end,reference,tz,te,iz,ie,audio_text,subject_id,subject_description):
    if not job_id or index is None: raise ValueError('请先选择一个镜头。')
    job=copy.deepcopy(engine.load(job_id));shot=job['shots'][int(index)]
    if round(float(start)*1000)!=shot['start_ms'] or round(float(end)*1000)!=shot['end_ms']:
        raise ValueError('镜头切点有未保存的修改，请先点击“保存修改”，确认结果后再保存案例。')
    overrides=json.loads(override_text or '{}')
    if not isinstance(overrides,dict): raise ValueError('人工修正必须是 JSON 对象。')
    ms=round(float(reference)*1000)
    if not shot['start_ms']<=ms<shot['end_ms']: raise ValueError('参考帧时间必须在镜头内。')
    if subject_id and subject_description:
        subject=job.get('subjects',{}).get(subject_id.strip(),{})
        if subject.get('description')!=subject_description.strip() or subject_id.strip() not in shot.get('subject_ids',[]):
            raise ValueError('主体资料有未关联的修改，请先点击“保存并关联主体”。')
    changed=(ms!=shot.get('reference_ms') or overrides!=shot.get('overrides',{}))
    shot['reference_ms']=ms;shot['reference_user_selected']=True;shot['overrides']=overrides
    previous=shot.get('prompts') or {}
    values={'text_zh':tz or '', 'text_en':te or '', 'image_zh':iz or '', 'image_en':ie or ''}
    manual=shot.setdefault('manual_prompts',{})
    for key,value in values.items():
        if value!=previous.get(key,''): manual[key]=value
    if previous or any(values.values()):
        shot['prompts']=values
        shot['prompt_review']=engine.review_prompts(values)
    if changed and shot.get('observation'): shot['status']='edited'
    if any(key in manual for key in ('text_zh','text_en')) and shot.get('coverage_review'):
        review=shot['coverage_review']
        review['missing_ids']=sorted(set(review.get('missing_ids',[])+review.get('reported_ids',[])))
        review['note']='提示词含人工修改，请重新核对全部动作阶段；模型自报不能证明内容真实。'
    transcript=job.get('audio_transcript')
    if transcript is not None:
        if audio_text!=transcript.get('edited_text',transcript.get('text','')): transcript['edited_text']=audio_text or ''
    elif audio_text and audio_text.strip():
        job['audio_transcript']={'status':'manual','text':'','edited_text':audio_text,'scope':'whole_video','timestamps':'unavailable'}
        job['video']['audio_status']='manual'
    return job


def store_case(job_id,index,title,override_text,start,end,reference,tz,te,iz,ie,audio_text,subject_id,subject_description):
    try:
        job=case_snapshot(job_id,index,override_text,start,end,reference,tz,te,iz,ie,audio_text,subject_id,subject_description)
        record=cases.save_case(job,title,int(index))
        return gr.update(choices=case_choices(),value=record['id']), f"案例“{record['title']}”已保存；当前任务未被覆盖。" + (' 参考帧或分析有变更，请核对并视需要重新生成提示词。' if job['shots'][int(index)].get('status')=='edited' else '')
    except Exception as e: raise gr.Error(str(e))


def refresh_cases():
    options=case_choices()
    return gr.update(choices=options), f'案例库中有 {len(options)} 个案例。'


def open_case(case_id):
    if not case_id: raise gr.Error('请先选择一个案例。')
    try:
        job,index=cases.restore_case(case_id)
        video=str(engine.DATA/job['id']/job['video']['file'])
        return job['id'],gr.update(choices=labels(job),value=str(index)),f"已打开案例的新工作副本，任务 ID：{job['id']}。",job['id'],video
    except Exception as e: raise gr.Error('无法打开案例：'+str(e))

def download(job_id):
    if not job_id: raise gr.Error('请先创建任务。')
    return engine.export(engine.load(job_id))

def clip_list(job_id):
    if not job_id: return gr.update(choices=[],value=None)
    job=engine.load(job_id); items=clips.ensure(job); engine.save(job)
    return gr.update(choices=[(f"{c['shot_id']} · {c['start_ms']/1000:.3f}–{c['end_ms']/1000:.3f} 秒 · {c['status']}",c['id']) for c in items],value=items[0]['id'] if items else None)

def clip_details(job_id, clip_id):
    if not job_id or not clip_id: return None,None,[],''
    try:
        job=engine.load(job_id); c=next(x for x in clips.ensure(job) if x['id']==clip_id)
        ev=clips.evidence(job,clip_id)
        gallery=[(f['path'],f"{f['time_ms']/1000:.3f} 秒 · 帧 {f['index']}") for f in ev['head_frames']+ev['tail_frames']]
        return ev['source_head']['path'],ev['source_tail']['path'],gallery,json.dumps({'clip':c,'head':ev['source_head'],'tail':ev['source_tail']},ensure_ascii=False,indent=2)
    except Exception as e: raise gr.Error(str(e))

def clip_split(job_id,clip_id,second):
    try:
        job=engine.load(job_id); clips.split(job,clip_id,round(float(second)*1000));engine.save(job)
        return clip_list(job_id),'片段已拆分；原镜头及其分析保持不变。'
    except Exception as e: raise gr.Error(str(e))

def clip_merge(job_id,clip_id):
    try:
        job=engine.load(job_id); items=clips.ensure(job); i=next(i for i,c in enumerate(items) if c['id']==clip_id)
        if i+1>=len(items): raise ValueError('没有后一个片段。')
        clips.merge(job,clip_id,items[i+1]['id']);engine.save(job)
        return clip_list(job_id),'已合并相邻片段。'
    except Exception as e: raise gr.Error(str(e))

def clip_analyze(job_id,clip_id,provider,model,url,key,strategy,image_mode):
    try:
        job=engine.load(job_id); c=next(x for x in clips.ensure(job) if x['id']==clip_id)
        original=next(s for s in job['shots'] if s['id']==c['shot_id'])
        context=clip_context.for_clip(job,clips.ensure(job),clip_id)
        evidence=clips.evidence(job,clip_id)
        boundary_times=sorted({int(f['time_ms']) for f in evidence['head_frames']+evidence['tail_frames']
                               if c['start_ms']<=int(f['time_ms'])<c['end_ms']})
        shot={'id':c['id'],'start_ms':c['start_ms'],'end_ms':c['end_ms'],'reference_ms':c['start_ms'],
              'status':'pending','observation':None,'prompts':None,'overrides':{},'error':None,
              'subject_ids':original.get('subject_ids',[]),'manual_frames':boundary_times}
        shot['overrides']['已确认的片段上下文']=context
        temp=copy.deepcopy(job);temp['_transient']=True;temp['shots']=[shot];temp['usage']=[]
        engine.analyze(temp,0,{**engine.connection(provider,model,url,key),'sampling_strategy':strategy,'image_mode':image_mode})
        c['analysis']=temp['shots'][0];c['status']=c['analysis']['status'];job.setdefault('usage',[]).extend(temp.get('usage',[]));engine.save(job)
        return json.dumps(c['analysis'],ensure_ascii=False,indent=2),'片段分析完成；结果独立保存，原镜头分析未覆盖。'
    except Exception as e: raise gr.Error(str(e))

def context_read(job_id):
    try:
        job=engine.load(job_id)
        return json.dumps(clip_context.read(job,clips.ensure(job)),ensure_ascii=False,indent=2)
    except Exception as e: raise gr.Error(str(e))

def context_save(job_id,value):
    try:
        job=engine.load(job_id);clip_context.save(job,clips.ensure(job),json.loads(value))
        for clip in job['clips']:
            if clip.get('analysis'): clip['status']='edited'
        engine.save(job)
        return context_read(job_id),'全片上下文已保存。已有片段提示词需重新分析以应用修改。'
    except Exception as e: raise gr.Error(str(e))

def reconstruction_view(job_id):
    import reconstruction
    if not job_id: raise gr.Error('请先创建任务。')
    return reconstruction.markdown(reconstruction.build(engine.load(job_id)))

DEFAULT_PROVIDER='硅基流动' if (os.getenv('SILICONFLOW_API_KEY') or engine.local_api_key('SILICONFLOW_API_KEY')) else 'Gemini'

with gr.Blocks(title='视频反推 Prompt',css=CSS,theme=gr.themes.Soft(primary_hue='emerald',neutral_hue='stone')) as app:
    gr.HTML('<header><div class="eyebrow">VIDEO → PROMPT / STUDIO</div><h1>把灵感，拆解成画面。</h1><p>逐镜头分析视频，整理成可以继续创作的双语提示词。</p></header>')
    job_state=gr.State('')
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown('### 01 / 输入视频')
            with gr.Accordion('模型与 API 设置',open=True):
                provider=gr.Dropdown(label='服务商', choices=list(engine.PRESETS), value=DEFAULT_PROVIDER)
                model=gr.Textbox(label='模型名称', value=(engine.PRESETS[DEFAULT_PROVIDER][1] if DEFAULT_PROVIDER=='硅基流动' else os.getenv('GEMINI_MODEL','gemini-2.5-flash')))
                url=gr.Textbox(label='API 基础地址', placeholder='Gemini 留空；自定义接口填写以 /v1 结尾的地址')
                key=gr.Textbox(label='API 密钥', type='password', placeholder='可留空以使用对应环境变量；本机服务可无密钥')
                gr.Markdown('<span class="note">可使用项目本地配置或页面输入密钥，密钥不会写入任务或导出；切换服务商时会清空页面输入。模型须支持图片输入。自定义地址使用 OpenAI Chat Completions 格式。</span>')
                test_api=gr.Button('测试连接')
                api_status=gr.Textbox(label='连接状态',interactive=False)

            video=gr.Video(label='上传视频',sources=['upload'],format='mp4')
            gr.Markdown('<span class="note">支持 60 秒以内、200 MB 以内的 MP4 / MOV。分析画面会发送至配置的模型服务。音频可选择单独转写。</span>')
            with gr.Row():
                create=gr.Button('创建任务',variant='primary'); analyze=gr.Button('分析全部镜头')
            with gr.Accordion('恢复已有任务',open=False):
                job_input=gr.Textbox(label='任务 ID');restore_btn=gr.Button('恢复任务')
            status=gr.Textbox(label='处理状态',interactive=False)
            with gr.Accordion('案例库',open=True):
                case_name=gr.Textbox(label='案例名称',placeholder='例如 雪地滑雪橇')
                case_save=gr.Button('保存当前案例')
                saved_cases=gr.Dropdown(label='已保存案例',choices=case_choices(),interactive=True)
                with gr.Row():
                    case_open=gr.Button('打开案例')
                    case_refresh=gr.Button('刷新列表')
                gr.Markdown('案例保存在本机，包含视频、结果和参考帧。打开案例会创建新的工作副本。')
            with gr.Accordion('音频与对白（可选）',open=False):
                gr.Markdown('手动触发后将整段音频发送至硅基流动，可能产生费用。转写没有时间戳，不会自动对应镜头或推断音乐音效。')
                audio_text=gr.Textbox(label='全片对白转写',lines=4)
                with gr.Row():
                    audio_btn=gr.Button('提取对白')
                    audio_save=gr.Button('保存对白修订')
            gr.Markdown('### 02 / 选择镜头')
            shots=gr.Dropdown(label='镜头列表',choices=[],interactive=True)
            with gr.Row():
                retry=gr.Button('重新分析本镜头'); resume_btn=gr.Button('继续上次分析'); export=gr.Button('导出结果')
            file=gr.File(label='下载分析包')
        with gr.Column(scale=6):
            gr.Markdown('### 03 / 镜头工作台')
            with gr.Accordion('Clip 分割与边界证据',open=False):
                gr.Markdown('先加载片段；每个镜头默认对应一个片段，可在镜头内拆分或与后一个片段合并。首尾帧和边界序列为原视频画面，不代表自动保证动作衔接。')
                clip_load=gr.Button('加载片段列表')
                clip_choice=gr.Dropdown(label='分析片段',choices=[],interactive=True)
                with gr.Row():
                    clip_head=gr.Image(label='原片首帧',height=200,interactive=False)
                    clip_tail=gr.Image(label='原片尾帧',height=200,interactive=False)
                clip_gallery=gr.Gallery(label='边界动态证据（原视频时间）',columns=4,height=220,object_fit='contain')
                clip_info=gr.Code(label='片段及边界时间信息',language='json',lines=8,interactive=False)
                with gr.Row():
                    clip_cut=gr.Number(label='拆分时间（原视频秒）',precision=3)
                    clip_split_btn=gr.Button('拆分片段')
                    clip_merge_btn=gr.Button('与下一片段合并')
                clip_run=gr.Button('独立分析此片段')
                clip_result=gr.Code(label='片段分析与提示词',language='json',lines=10,interactive=False)
                clip_status=gr.Textbox(label='片段状态',interactive=False)
                gr.Markdown('全片上下文可记录已确认的素材与衔接；请先加载再编辑 JSON，来源时间使用原视频毫秒。')
                context_load=gr.Button('加载全片上下文')
                context_box=gr.Code(label='全片上下文',language='json',lines=10)
                context_save_btn=gr.Button('保存全片上下文')
            with gr.Accordion('采样与对比',open=True):
                with gr.Row():
                    strategy=gr.Radio(['等间隔','变化驱动','Prompt 专用'],value='等间隔',label='抽帧策略')
                    image_mode=gr.Radio(['单帧','拼图'],value='单帧',label='发送方式')
                gr.Markdown('<span class="note">等间隔保留原策略；变化驱动按画面变化补帧；Prompt 专用结合 TransNetV2 转场、动作三帧组、清晰度与时间覆盖。首次会在本机加载模型。拼图可能丢失细节，实际费用取决于模型。</span>')
                preview_btn=gr.Button('预览采样帧（不调用模型）')
                sampling_summary=gr.Markdown('暂无采样预览。')
                gallery=gr.Gallery(label='采样帧与选择原因',columns=4,height=300,object_fit='contain')
                with gr.Row():
                    manual_second=gr.Number(label='补帧时间（原视频秒）',precision=3)
                    add_btn=gr.Button('添加补帧')
                gr.Markdown('对比会分别运行三种策略，固定使用当前模型和单帧模式，会产生三组分析费用。原始镜头边界不改写。')
                compare_btn=gr.Button('对比三种抽帧策略')
                with gr.Row():
                    compare_old=gr.Textbox(label='原版 · 等间隔',lines=10,interactive=False)
                    compare_new=gr.Textbox(label='新版 · 变化驱动',lines=10,interactive=False)
                    compare_prompt=gr.Textbox(label='Prompt 专用 · TransNetV2',lines=10,interactive=False)

            with gr.Accordion('跨镜头主体（手动确认）',open=False):
                gr.Markdown('使用相同标签关联重复出现的主体；系统不会自动认定人物身份。外观更新后，请重新生成相关镜头的 Prompt。')
                subject_id=gr.Textbox(label='主体标签（英文字母、数字、下划线或短横线）',placeholder='例如 child_1')
                subject_description=gr.Textbox(label='统一外观描述',placeholder='例如 红色雪服、彩色针织帽，坐在黄色雪橇上')
                linked=gr.Textbox(label='当前镜头关联主体',interactive=False)
                subject_registry=gr.Code(label='项目主体表',language='json',interactive=False,lines=5)
                with gr.Row():
                    subject_link=gr.Button('保存并关联主体')
                    subject_unlink=gr.Button('解除当前镜头关联')
            ref=gr.Image(label='生成参考帧',height=260,interactive=False)
            with gr.Row():
                start=gr.Number(label='镜头开始（秒）',precision=3);end=gr.Number(label='镜头结束（秒）',precision=3);reference=gr.Number(label='参考帧（秒）',precision=3)
            gr.Markdown('<span class="note">调整切点会同步调整相邻镜头。参考帧按原始分辨率提取；如果选择镜头中途，图生视频从该时刻开始，请核对之后的动作。</span>')
            reference_preview=gr.Button('预览并选择参考帧')
            with gr.Accordion('分析证据与人工修正',open=False):
                observation=gr.Code(label='分析结果与证据时间',language='json',interactive=False,lines=12)
                override=gr.Code(label='人工修正（JSON 对象）',language='json',lines=6,value='{}')
                gr.Markdown('例如：`{"subject": "穿蓝色外套的女性", "camera": "固定机位"}`')
            with gr.Tabs():
                with gr.Tab('文生视频'):
                    tz=gr.Textbox(label='中文',lines=5);te=gr.Textbox(label='English',lines=5)
                with gr.Tab('图生视频'):
                    iz=gr.Textbox(label='中文 · 动作',lines=5);ie=gr.Textbox(label='English · motion',lines=5)
            with gr.Row():
                save=gr.Button('保存修改',variant='primary'); regenerate=gr.Button('重新生成 Prompt')
            with gr.Accordion('全片复现流程',open=False):
                gr.Markdown('先保存提示词和人工修改，再刷新计划。修改后需重新刷新；导出始终读取最新保存内容。此操作不调用模型。')
                plan_refresh=gr.Button('根据已保存结果刷新复现计划')
                plan_view=gr.Markdown('尚未读取计划。')
    view=[ref,observation,override,start,end,reference,tz,te,iz,ie,gallery,sampling_summary,compare_old,compare_new,compare_prompt]
    create.click(make_job,video,[job_state,shots,status,job_input]).then(show,[job_state,shots],view).then(show_audio,job_state,audio_text)
    restore_btn.click(restore,job_input,[job_state,shots,status,job_input,video]).then(show,[job_state,shots],view).then(show_audio,job_state,audio_text)
    shots.change(show,[job_state,shots],view).then(subject_view,[job_state,shots],[subject_registry,linked])
    provider.change(preset,provider,[model,url,key])
    test_api.click(test_connection,[provider,model,url,key],api_status)
    analyze.click(all_shots,[job_state,provider,model,url,key,strategy,image_mode],[shots,status]).then(show,[job_state,shots],view)
    retry.click(one,[job_state,shots,provider,model,url,key,strategy,image_mode],[shots,status]).then(show,[job_state,shots],view)
    reference_preview.click(select_reference,[job_state,shots,reference],[ref,status])
    resume_btn.click(resume_shot,[job_state,shots,provider,model,url,key,strategy,image_mode],[shots,status]).then(show,[job_state,shots],view)
    save.click(save_edits,[job_state,shots,override,start,end,reference,tz,te,iz,ie],[shots,status]).then(show,[job_state,shots],view)
    regenerate.click(regen,[job_state,shots,provider,model,url,key],[shots,status]).then(show,[job_state,shots],view)
    preview_btn.click(preview_sampling,[job_state,shots,strategy],status).then(sampling_view,[job_state,shots],[gallery,sampling_summary])
    add_btn.click(add_frame,[job_state,shots,manual_second,strategy],status).then(sampling_view,[job_state,shots],[gallery,sampling_summary])
    compare_btn.click(compare_shot,[job_state,shots,provider,model,url,key],[status,compare_old,compare_new,compare_prompt])
    subject_link.click(link_subject,[job_state,shots,subject_id,subject_description],[status,subject_registry,linked])
    subject_unlink.click(unlink_subject,[job_state,shots,subject_id],[status,subject_registry,linked])
    audio_btn.click(transcribe_audio,[job_state,provider,key],[audio_text,status])
    audio_save.click(save_audio,[job_state,audio_text],status)
    case_save.click(store_case,[job_state,shots,case_name,override,start,end,reference,tz,te,iz,ie,audio_text,subject_id,subject_description],[saved_cases,status])
    case_refresh.click(refresh_cases,outputs=[saved_cases,status])
    case_open.click(open_case,saved_cases,[job_state,shots,status,job_input,video]).then(show,[job_state,shots],view).then(show_audio,job_state,audio_text).then(subject_view,[job_state,shots],[subject_registry,linked])
    export.click(download,job_state,file)
    clip_load.click(clip_list,job_state,clip_choice)
    clip_choice.change(clip_details,[job_state,clip_choice],[clip_head,clip_tail,clip_gallery,clip_info])
    clip_split_btn.click(clip_split,[job_state,clip_choice,clip_cut],[clip_choice,clip_status])
    clip_merge_btn.click(clip_merge,[job_state,clip_choice],[clip_choice,clip_status])
    clip_run.click(clip_analyze,[job_state,clip_choice,provider,model,url,key,strategy,image_mode],[clip_result,clip_status])
    context_load.click(context_read,job_state,context_box)
    context_save_btn.click(context_save,[job_state,context_box],[context_box,clip_status])
    plan_refresh.click(reconstruction_view,job_state,plan_view)
    job_state.change(lambda: '任务已切换，请刷新复现计划。',outputs=plan_view)

if __name__=='__main__':
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=int(os.getenv('PORT','7860')),inbrowser=False,allowed_paths=[str(engine.DATA)])
