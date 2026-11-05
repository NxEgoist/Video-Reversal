"""Transition-aware, quality-ranked temporal evidence selection for prompting."""
import cv2
import numpy as np
from sampling import change_score

VERSION='prompt-evidence-v4'


def scan(path,start,end):
    cap=cv2.VideoCapture(str(path));rows=[];previous=None;last=start-125; last_pts=-1
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,start)
        while True:
            ok,frame=cap.read()
            if not ok: break
            ms=round(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ms<=last_pts: raise ValueError('无法可靠读取视频时间戳。')
            last_pts=ms
            if ms>=end: break
            if ms<start or ms-last<125: continue
            last=ms
            small=cv2.resize(frame,(160,90));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            clarity=float(cv2.Laplacian(gray,cv2.CV_32F).var())
            exposure=float(np.mean((gray<8)|(gray>247)))
            change=0.;local=0.;camera=0.;dx=0.;dy=0.
            if previous is not None:
                change=change_score(previous,small)
                flow=cv2.calcOpticalFlowFarneback(cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),gray,None,.5,3,15,3,5,1.2,0)
                median=np.median(flow.reshape(-1,2),axis=0)
                dx=float(median[0]);dy=float(median[1])
                camera=float(np.linalg.norm(median))
                local=float(np.percentile(np.linalg.norm(flow-median,axis=2),90))
            rows.append({'ms':ms,'clarity':clarity,'quality':float(np.log1p(clarity)*(1-.7*exposure)),
                         'change':change,'local_motion':local,'global_motion':camera,'global_dx':dx,'global_dy':dy})
            previous=small
    finally: cap.release()
    if not rows: raise ValueError('当前镜头没有可用的视频帧。')
    return rows


def select(rows,start,end,transitions,manual=()):
    # Transition evidence is kept as metadata; blended frames are not made reference candidates.
    spans=[];cursor=start
    for t in sorted(transitions,key=lambda t:t['start_ms']):
        a=max(start,t['start_ms']);b=min(end,t['end_ms'])
        if a>=b: continue
        if a>cursor: spans.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<end: spans.append((cursor,end))
    if not spans: raise ValueError('该镜头全部被模型判为转场区间，请检查切点或使用原版策略。')
    selected={}
    def add(row,reason,segment):
        ms=row['ms']
        entry=selected.setdefault(ms,{'ms':ms,'reasons':[], 'segment':segment,
                                     'quality':round(row['quality'],3),
                                     'local_motion':round(row['local_motion'],3),
                                     'global_motion':round(row['global_motion'],3),
                                     'global_dx':round(row.get('global_dx',0.),3),
                                     'global_dy':round(row.get('global_dy',0.),3)})
        if reason not in entry['reasons']:entry['reasons'].append(reason)
    for seg,(a,b) in enumerate(spans):
        candidates=[r for r in rows if a<=r['ms']<b]
        if not candidates: continue
        # Each 12-second unit owns its evidence budget; never truncate the tail of the shot.
        for lo in range(a,b,12000):
            hi=min(b,lo+12000);pool=[r for r in candidates if lo<=r['ms']<hi]
            if not pool: continue
            for target in sorted(set([lo,(lo+hi)//2,hi-1]+list(range(lo,hi,2000)))):
                near=sorted(pool,key=lambda r:abs(r['ms']-target))
                distance=abs(near[0]['ms']-target)
                nearby=[r for r in near if abs(r['ms']-target)<=distance+125]
                row=max(nearby,key=lambda r:r['quality'])
                add(row,'稳定状态 / 时间覆盖',seg)
            # Short windows receive a proportional budget; manual evidence remains exempt.
            budget=min(24,max(8,4+int(np.ceil((hi-lo)/500))))
            # Events are kept as whole before/during/after triplets, with local motion emphasized.
            ranks=[];integral=0.
            for i,r in enumerate(pool):
                integral+=r['change']
                salience=r['local_motion']*3 + min(r['global_motion'],8) + r['change']/8
                if r['local_motion']>=.6 or r['global_motion']>=.8 or integral>=18:
                    ranks.append((salience,i));integral=0.
            chosen_events=[]
            for _,i in sorted(ranks,reverse=True):
                center=pool[i]['ms']
                if any(abs(center-t)<750 for t in chosen_events): continue
                group=[min(pool,key=lambda r:abs(r['ms']-target)) for target in (max(lo,center-300),center,min(hi-1,center+300))]
                group=list({r['ms']:r for r in group}.values())
                current={ms for ms in selected if lo<=ms<hi}
                if len(current | {r['ms'] for r in group})>budget: continue
                for r in group:add(r,'动作前中后',seg)
                chosen_events.append(center)
    for ms in manual:
        if start<=ms<end:
            segment=next((i for i,(a,b) in enumerate(spans) if a<=ms<b),-1)
            add({'ms':ms,'quality':0.,'local_motion':0.,'global_motion':0.},'人工补帧（转场内）' if segment<0 else '人工补帧',segment)
    if not selected: raise ValueError('稳定区间过短，没有采到画面；请使用原版策略。')
    return sorted(selected.values(),key=lambda r:r['ms'])


def windows(plan):
    """Do not cross detected transitions; overlap two frames at budget boundaries."""
    groups=[];current=[];segment=None
    for row in plan:
        if current and row.get('segment')!=segment:
            groups.append(current);current=[]
        segment=row.get('segment');current.append(row['ms'])
    if current: groups.append(current)
    out=[]
    for group in groups:
        i=0
        while i<len(group):
            out.append(group[i:i+24])
            if i+24>=len(group): break
            i+=22
    return out


def motion_evidence(plan, window):
    """Describe coherent screen displacement, not its physical cause."""
    by_time={r['ms']:r for r in plan}
    selected=[by_time[t] for t in window if t in by_time]
    active=[r for r in selected if r.get('global_motion',0)>=0.7]
    if len(active)<3 or len(active)<len(selected)*0.5:
        return {'status':'insufficient','description':'全局像素位移证据不足，不据此判断运镜。'}
    dx=float(np.median([r.get('global_dx',0) for r in active]))
    dy=float(np.median([r.get('global_dy',0) for r in active]))
    magnitude=float(np.hypot(dx,dy))
    if magnitude<0.7:
        return {'status':'insufficient','description':'全局像素位移方向不稳定，不据此判断运镜。'}
    direction=('向右' if dx>0 else '向左') if abs(dx)>=abs(dy) else ('向下' if dy>0 else '向上')
    residual=float(np.median([r.get('local_motion',0) for r in active]))
    # shortcut: coarse optical flow can miss small or occluded actions; use it as a cue, not proof of stillness.
    mostly_coherent=sum(r.get('local_motion',float('inf'))<=max(0.5,r.get('global_motion',0)*0.25) for r in active)>=len(active)*0.75
    return {'status':'coherent_translation','screen_direction':direction,
            'median_dx':round(dx,2),'median_dy':round(dy,2),'median_residual':round(residual,2),
            'predominantly_coherent':mostly_coherent,
            'description':'多处画面纹理呈一致平移；这是画面像素位移，不足以单独确定真实相机运动或人物动作，应结合画面验证。'}
