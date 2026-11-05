"""User-confirmed subject descriptions shared across shots."""
import re
import engine


def link(job,index,identifier,description):
    identifier=identifier.strip()
    description=description.strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',identifier):
        raise ValueError('主体标签仅支持 1–32 位英文字母、数字、下划线或短横线。')
    if not description or len(description)>1000:
        raise ValueError('请填写不超过 1000 字的外观描述。')
    registry=job.setdefault('subjects',{})
    changed=registry.get(identifier,{}).get('description')!=description
    registry[identifier]={'description':description}
    shot=job['shots'][index]
    if identifier not in shot.setdefault('subject_ids',[]):
        shot['subject_ids'].append(identifier)
        if shot.get('status')=='complete': shot['status']='edited'
    if changed:
        for item in job['shots']:
            if identifier in item.get('subject_ids',[]) and item.get('status')=='complete': item['status']='edited'
    engine.save(job)


def unlink(job,index,identifier):
    shot=job['shots'][index];ids=shot.get('subject_ids',[])
    if identifier not in ids: raise ValueError('当前镜头未关联这个主体。')
    ids.remove(identifier)
    if shot.get('status')=='complete':shot['status']='edited'
    engine.save(job)
