"""Whole-video speech transcription via SiliconFlow; no inferred timestamps."""
import tempfile
from pathlib import Path
import httpx
import engine


def transcribe(job, api_key=''):
    if not job['video'].get('audio_track_present'):
        raise ValueError('该视频没有音轨，无法转写。')
    config=engine.connection('硅基流动',key=api_key)
    source=engine.DATA/job['id']/job['video']['file']
    with tempfile.TemporaryDirectory(prefix='video-prompt-audio-') as directory:
        wav=Path(directory)/'audio.wav'
        engine.run(['ffmpeg','-v','error','-i',str(source),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-y',str(wav)])
        try:
            with wav.open('rb') as f:
                response=httpx.post('https://api.siliconflow.cn/v1/audio/transcriptions',
                    headers={'Authorization':'Bearer '+config['key']},
                    files={'file':('audio.wav',f,'audio/wav')},
                    data={'model':'FunAudioLLM/SenseVoiceSmall'},timeout=120,follow_redirects=False)
            response.raise_for_status()
            result=response.json()
            if not isinstance(result,dict) or not isinstance(result.get('text'),str):
                raise ValueError('语音服务返回格式不兼容。')
        except httpx.HTTPStatusError as exc:
            raise ValueError(f'语音转写失败（HTTP {exc.response.status_code}），请检查密钥、额度和模型权限。') from exc
        except httpx.TimeoutException as exc:
            raise ValueError('语音转写超时，请稍后重试。') from exc
        except httpx.RequestError as exc:
            raise ValueError('语音服务连接失败，请检查网络。') from exc
        except ValueError as exc:
            raise ValueError('语音服务返回格式不兼容。') from exc
    transcript={'status':'complete','model':'FunAudioLLM/SenseVoiceSmall','text':result['text'].strip(),
                'scope':'whole_video','timestamps':'unavailable'}
    previous=job.get('audio_transcript') or {}
    if 'edited_text' in previous: transcript['edited_text']=previous['edited_text']
    job['audio_transcript']=transcript
    job['video']['audio_status']='transcribed'
    engine.save(job)
    return transcript
