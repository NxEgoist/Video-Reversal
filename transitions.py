"""Official TransNetV2 SavedModel inference, with native decode timestamps."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

MODEL = Path(__file__).parent/'models'/'transnetv2'


@lru_cache(maxsize=1)
def network():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError('Prompt 专用策略需要 TensorFlow：请安装 requirements-transnet.txt。') from exc
    if not (MODEL/'saved_model.pb').exists():
        raise ValueError('缺少 TransNetV2 官方模型文件，请检查 models/transnetv2。')
    return tf.saved_model.load(str(MODEL))


def transition_ranges(timestamps, single, many, duration, threshold=.5):
    mask=(np.asarray(single)>=threshold) | (np.asarray(many)>=threshold)
    ranges=[]; first=None
    for i, active in enumerate(list(mask)+[False]):
        if active and first is None: first=i
        if not active and first is not None:
            ranges.append({'start_ms':int(timestamps[first]),
                           'end_ms':int(timestamps[i]) if i<len(timestamps) else duration,
                           'probability':float(max(np.max(single[first:i]),np.max(many[first:i])))})
            first=None
    return ranges


def detect(path, duration):
    path=Path(path)
    stamp=path.stat()
    manifest=(MODEL/'source.json').read_bytes() if (MODEL/'source.json').exists() else b'missing'
    identity=hashlib.sha256(manifest+f'{stamp.st_size}:{stamp.st_mtime_ns}:tn-v1'.encode()).hexdigest()
    cache=path.parent/'transnet-v1.json'
    if cache.exists():
        old=json.loads(cache.read_text())
        if old.get('identity')==identity: return old['transitions']
    model=network()
    import tensorflow as tf
    cap=cv2.VideoCapture(str(path));frames=[];timestamps=[]
    try:
        while True:
            ok, frame=cap.read()
            if not ok: break
            ms=round(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamps and ms<=timestamps[-1]:
                raise ValueError('无法可靠读取连续视频时间戳，不能执行 TransNetV2 分析。')
            timestamps.append(ms)
            frames.append(cv2.cvtColor(cv2.resize(frame,(48,27)),cv2.COLOR_BGR2RGB))
    finally: cap.release()
    if not frames: raise ValueError('无法解码视频。')
    arr=np.asarray(frames,dtype=np.uint8);n=len(arr)
    padding=25+(50-n%50)%50
    padded=np.pad(arr,((25,padding),(0,0),(0,0),(0,0)),mode='edge')
    single=[];many=[]
    for i in range(0,n,50):
        logits, extra=model(tf.cast(padded[None,i:i+100],tf.float32))
        single.extend(tf.sigmoid(logits).numpy()[0,25:75,0].tolist())
        many.extend(tf.sigmoid(extra['many_hot']).numpy()[0,25:75,0].tolist())
    found=transition_ranges(timestamps,single[:n],many[:n],duration)
    tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps({'identity':identity,'transitions':found}));tmp.replace(cache)
    return found
