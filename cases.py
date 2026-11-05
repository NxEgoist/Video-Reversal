"""Local, independent case snapshots and restoration into new working jobs."""
import copy
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
import engine


def _case_dir(case_id):
    if not case_id or len(case_id)!=32 or any(c not in '0123456789abcdef' for c in case_id):
        raise ValueError('无效的案例编号。')
    return engine.DATA/'cases'/case_id


def _source_name(job):
    name=job['video']['file']
    if not isinstance(name,str) or Path(name).name != name or Path(name).suffix.lower() not in ('.mp4','.mov'):
        raise ValueError('案例中的视频文件名无效。')
    return name


def save_case(job,title,selected_index=0):
    title=(title or '').strip()
    if not title or len(title)>100: raise ValueError('请填写不超过 100 字的案例名称。')
    if not job.get('shots') or not 0<=selected_index<len(job['shots']): raise ValueError('请选择有效镜头。')
    source=engine.DATA/job['id']/_source_name(job)
    if not source.is_file(): raise ValueError('原始视频不存在，无法保存案例。')
    snapshot=copy.deepcopy(job);snapshot.pop('_transient',None)
    for shot in snapshot['shots']:
        if shot.get('status')=='processing':
            raise ValueError('任务仍在分析，请等当前分析结束后保存。')
    case_id=uuid.uuid4().hex
    parent=engine.DATA/'cases';parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.case-',dir=parent))
    try:
        shutil.copy2(source,staging/_source_name(snapshot))
        refs=staging/'references';refs.mkdir()
        for shot in snapshot['shots']:
            ms=shot.get('reference_ms',shot['start_ms'])
            reference=engine.reference_frame(job,ms)
            shutil.copy2(reference,refs/f'{ms}.jpg')
        if (engine.DATA/job['id']/'clip_frames').is_dir():
            shutil.copytree(engine.DATA/job['id']/'clip_frames',staging/'clip_frames')
        record={'version':1,'id':case_id,'title':title,'created_at':time.time(),
                'selected_index':selected_index,'job':snapshot}
        (staging/'case.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(staging,_case_dir(case_id))
    except Exception:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return record


def list_cases():
    parent=engine.DATA/'cases'
    if not parent.exists(): return []
    result=[]
    for path in parent.iterdir():
        if not path.is_dir() or len(path.name)!=32 or any(c not in '0123456789abcdef' for c in path.name): continue
        try:
            data=json.loads((path/'case.json').read_text(encoding='utf-8'))
            if data.get('id')!=path.name: continue
            result.append({'id':data['id'],'title':data['title'],'created_at':data['created_at']})
        except (OSError,ValueError,KeyError,TypeError): continue
    return sorted(result,key=lambda item:item['created_at'],reverse=True)


def restore_case(case_id):
    directory=_case_dir(case_id)
    record=json.loads((directory/'case.json').read_text(encoding='utf-8'))
    if record.get('id')!=case_id: raise ValueError('案例编号与内容不一致。')
    job=copy.deepcopy(record['job']);name=_source_name(job)
    video=directory/name
    if not video.is_file(): raise ValueError('案例中的视频文件不存在。')
    index=record.get('selected_index',0)
    if not isinstance(index,int) or not 0<=index<len(job.get('shots',[])): raise ValueError('案例的镜头选择无效。')
    job_id=uuid.uuid4().hex
    staging=Path(tempfile.mkdtemp(prefix='.restore-',dir=engine.DATA))
    try:
        shutil.copy2(video,staging/name)
        if (directory/'references').is_dir(): shutil.copytree(directory/'references',staging/'references')
        if (directory/'clip_frames').is_dir(): shutil.copytree(directory/'clip_frames',staging/'clip_frames')
        job['id']=job_id;job.pop('_transient',None)
        (staging/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(staging,engine.DATA/job_id)
    except Exception:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return job,index
