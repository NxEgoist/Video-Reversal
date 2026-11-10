import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import engine
from backend.api import app


class ApiTests(unittest.TestCase):
    def test_director_note_persists_and_survives_split_and_merge(self):
        import clips
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, 'DATA', Path(tmp)):
            (Path(tmp)/'abc').mkdir()
            job = {'id':'abc', 'video':{'file':'source.mp4','duration_ms':2000}, 'shots':[{'id':'s1','start_ms':0,'end_ms':2000,'status':'pending'}]}
            clip = clips.ensure(job)[0]; engine.save(job)
            with TestClient(app) as client:
                base = '/api/jobs/abc/clips/'
                self.assertEqual(client.put(base+clip['id']+'/director-note', json={'text':'近景，缓慢推近'}).status_code, 200)
                self.assertEqual(client.get('/api/jobs/abc').json()['clips'][0]['director_note'], '近景，缓慢推近')
                parts = client.post(base+clip['id']+'/split', json={'time_ms':1000}).json()['clips']
                self.assertEqual([c['director_note'] for c in parts], ['近景，缓慢推近']*2)
                merged = client.post(base+parts[0]['id']+'/merge').json()['clips'][0]
                self.assertIn('0.00–1.00 秒', merged['director_note'])
                self.assertIn('近景，缓慢推近', merged['director_note'])


    def test_full_shot_results_are_visible_to_matching_clip_without_overwriting_edits(self):
        import clips
        with tempfile.TemporaryDirectory() as tmp,patch.object(engine,'DATA',Path(tmp)):
            folder=Path(tmp)/'abc';folder.mkdir()
            prompts={'text_zh':'镜头结果','text_en':'scene','image_zh':'动作','image_en':'motion'}
            job={'id':'abc','video':{'file':'source.mp4','duration_ms':2000},'shots':[{'id':'s1','start_ms':0,'end_ms':2000,'status':'complete','prompts':prompts,'observation':{'windows':[]}}]}
            clip=clips.ensure(job)[0];engine.save(job)
            with TestClient(app) as client:
                result=client.get('/api/jobs/abc').json()['clips'][0]
                self.assertEqual(result['analysis_source'],'shot')
                self.assertEqual(result['analysis']['prompts']['text_zh'],'镜头结果')
                self.assertNotIn('analysis',engine.load('abc')['clips'][0])
                edited={**prompts,'text_zh':'人工片段修订'}
                self.assertEqual(client.put(f"/api/jobs/abc/clips/{clip['id']}/prompts",json=edited).status_code,200)
                result=client.get('/api/jobs/abc').json()['clips'][0]
                self.assertEqual(result['analysis']['prompts']['text_zh'],'人工片段修订')
                self.assertEqual(engine.load('abc')['shots'][0]['prompts']['text_zh'],'镜头结果')
                result=client.post(f"/api/jobs/abc/clips/{clip['id']}/split",json={'time_ms':1000})
                self.assertEqual(result.status_code,200)
                self.assertTrue(all('analysis' not in c for c in result.json()['clips']))

    def test_failed_clip_retains_checkpoint_for_retry(self):
        import clips
        from backend import api
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine,'DATA',Path(tmp)):
            folder=Path(tmp)/'abc';folder.mkdir()
            job={'id':'abc','video':{'file':'source.mp4','duration_ms':2000},'shots':[{'id':'s1','start_ms':0,'end_ms':2000,'status':'pending'}]}
            clip=clips.ensure(job)[0];engine.save(job)
            frame={'time_ms':0}; calls=[]
            def analyze(temp,index,config,resume=False):
                calls.append(temp['shots'][0].get('analysis_checkpoint'))
                if len(calls)==1:
                    temp['shots'][0]['analysis_checkpoint']={'fingerprint':'x','windows':{'0':{'scene':'已完成'}}}
                    raise ValueError('分析画面窗口 2/7：API 请求超时')
                temp['shots'][0]['status']='complete'
            with patch.object(api.clips,'evidence',return_value={'head_frames':[frame],'tail_frames':[frame]}),patch.object(engine,'analyze',side_effect=analyze),TestClient(app) as client:
                url=f"/api/jobs/abc/clips/{clip['id']}/analysis"
                body={'provider':'自定义（OpenAI 兼容）','model':'test','url':'http://127.0.0.1:1234/v1'}
                for expected in ('failed','complete'):
                    self.assertEqual(client.post(url,json=body).status_code,202)
                    for _ in range(200):
                        current=client.get('/api/jobs/abc').json()
                        if current['api_task']['status']!='processing':break
                        time.sleep(.01)
                    self.assertEqual(current['api_task']['status'],expected)
                self.assertEqual(calls[1]['windows']['0']['scene'],'已完成')

    def test_video_analysis_case_and_export_end_to_end(self):
        import zipfile
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'sample.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)])
            def fake_model(job,instruction,images,schema,config=None):
                if schema is engine.Observation: return engine.Observation(subject='合成画面').model_dump()
                return engine.Prompts(text_zh='原画面',text_en='scene',image_zh='运动',image_en='motion').model_dump()
            with patch.object(engine,'DATA',root/'data'),patch.object(engine,'call_model',side_effect=fake_model):
                engine.DATA.mkdir()
                with TestClient(app) as client,video.open('rb') as source:
                    job=client.post('/api/jobs',files={'video':('sample.mp4',source,'video/mp4')}).json()
                    cid=job['clips'][0]['id'];base=f"/api/jobs/{job['id']}"
                    settings={'provider':'自定义（OpenAI 兼容）','model':'test','url':'http://127.0.0.1:1234/v1'}
                    self.assertEqual(client.post(f'{base}/clips/{cid}/analysis',json=settings).status_code,202)
                    for _ in range(300):
                        job=client.get(base).json()
                        if job['api_task']['status']!='processing':break
                        time.sleep(.02)
                    self.assertEqual(job['api_task']['status'],'complete',job['api_task'])
                    prompts=job['clips'][0]['analysis']['prompts'];prompts['text_zh']='人工修改'
                    self.assertEqual(client.put(f'{base}/clips/{cid}/prompts',json=prompts).status_code,200)
                    self.assertEqual(client.put(f'{base}/clips/{cid}/director-note',json={'text':'人工观察'}).status_code,200)
                    saved=client.post(f'{base}/cases',json={'title':'合成测试'})
                    self.assertEqual(saved.status_code,200,saved.text)
                    restored=client.post(f"/api/cases/{saved.json()['id']}/restore").json()
                    self.assertEqual(restored['clips'][0]['analysis']['prompts']['text_zh'],'人工修改')
                    self.assertEqual(restored['clips'][0]['director_note'],'人工观察')
                    archive=client.get(f'{base}/export')
                    self.assertEqual(archive.status_code,200)
                    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
                        self.assertIn('clips.json',z.namelist())
                        self.assertIn('人工修改',z.read('clips.json').decode())
                        self.assertIn('人工观察',z.read('clips.json').decode())

    def test_all_shots_continues_after_failure_and_reports_partial(self):
        import clips
        from backend import api
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, 'DATA', Path(tmp)):
            folder=Path(tmp)/'abc';folder.mkdir()
            job={'id':'abc','video':{'file':'source.mp4','duration_ms':2000},'shots':[
                {'id':'first','start_ms':0,'end_ms':1000,'status':'pending'},
                {'id':'second','start_ms':1000,'end_ms':2000,'status':'pending'}]}
            clips.ensure(job);engine.save(job)
            def analyze(job,index,config,resume=False):
                if index==0:
                    job['shots'][index]['status']='failed';engine.save(job);raise ValueError('模拟失败')
                job['shots'][index]['status']='complete';engine.save(job)
            with patch.object(engine,'analyze',side_effect=analyze), TestClient(app) as client:
                body={'provider':'自定义（OpenAI 兼容）','model':'test','url':'http://127.0.0.1:1234/v1'}
                self.assertEqual(client.post('/api/jobs/abc/operations/all',json=body).status_code,202)
                for _ in range(200):
                    current=client.get('/api/jobs/abc').json()
                    if current['api_task']['status']!='processing':break
                    time.sleep(.01)
                self.assertEqual(current['shots'][1]['status'],'complete')
                self.assertEqual(current['api_task']['status'],'complete_partial')
                self.assertIn('first',current['api_task']['error'])

    def test_settings_persist_without_returning_secrets_or_reusing_on_new_host(self):
        import provider_settings
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            with patch.object(provider_settings, 'FILE', path), patch.object(provider_settings, 'opencode_local', return_value='local-go-secret'):
                with TestClient(app) as client:
                    settings=client.get('/api/settings').json()
                    self.assertTrue(settings['profiles']['OpenCode Go']['key_configured'])
                    self.assertNotIn('local-go-secret',str(settings))
                    edit={'active':'OpenCode Go','name':'OpenCode Go','model':'vision-model','url':provider_settings.GO_URL,'key':'saved-secret'}
                    result=client.put('/api/settings',json=edit)
                    self.assertEqual(result.status_code,200,result.text)
                    self.assertNotIn('saved-secret',result.text)
                    self.assertEqual(path.stat().st_mode & 0o777,0o600)
                    edit['key']='';edit['url']='https://example.com/v1'
                    self.assertEqual(client.put('/api/settings',json=edit).status_code,400)
                    self.assertEqual(provider_settings.resolve('OpenCode Go')['key'],'saved-secret')
                    edit={'active':'自定义（OpenAI 兼容）','name':'自定义（OpenAI 兼容）','model':'local','url':'http://127.0.0.1:11434/v1','key':''}
                    self.assertEqual(client.put('/api/settings',json=edit).status_code,200)
                    with patch.dict('os.environ',{'CUSTOM_API_KEY':'unrelated-secret'}):
                        self.assertEqual(provider_settings.resolve('自定义（OpenAI 兼容）')['key'],'')

    def test_analysis_and_manual_prompts_are_preserved(self):
        import clips
        from backend import api
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(engine, 'DATA', root):
                folder = root / 'abc'
                folder.mkdir()
                job = {'id':'abc', 'video':{'file':'source.mp4', 'duration_ms':2000}, 'shots':[{'id':'shot_001','start_ms':0,'end_ms':2000,'status':'pending'}]}
                clip = clips.ensure(job)[0]
                engine.save(job)
                frame = {'time_ms':0}
                def fake_analysis(temp, index, config, resume=False):
                    temp['shots'][0]['status'] = 'complete'
                    temp['shots'][0]['prompts'] = {'text_zh':'中','text_en':'en','image_zh':'动','image_en':'move'}
                    temp['shots'][0]['prompts'].update(temp['shots'][0].get('manual_prompts', {}))
                with patch.object(api.clips, 'evidence', return_value={'head_frames':[frame], 'tail_frames':[frame]}), patch.object(engine, 'analyze', side_effect=fake_analysis):
                    with TestClient(app) as client:
                        url = f"/api/jobs/abc/clips/{clip['id']}"
                        body = {'provider':'自定义（OpenAI 兼容）','model':'test','url':'http://127.0.0.1:1234/v1','key':'secret'}
                        self.assertEqual(client.post(url+'/analysis',json=body).status_code,202)
                        for _ in range(200):
                            current = client.get('/api/jobs/abc').json()
                            if current['api_task']['status'] != 'processing': break
                            time.sleep(.01)
                        self.assertEqual(current['api_task']['status'],'complete')
                        prompts = current['clips'][0]['analysis']['prompts']
                        prompts['text_zh'] = '人工修订'
                        self.assertEqual(client.put(url+'/prompts',json=prompts).status_code,200)
                        self.assertNotIn('secret', (folder/'job.json').read_text())
                        self.assertEqual(client.post(url+'/analysis',json=body).status_code,202)
                        for _ in range(200):
                            current = client.get('/api/jobs/abc').json()
                            if current['api_task']['status'] != 'processing': break
                            time.sleep(.01)
                        self.assertEqual(current['clips'][0]['analysis']['prompts']['text_zh'],'人工修订')
                        self.assertEqual(client.post('/api/jobs/abc/operations/all',json=body).status_code,202)
                        for _ in range(200):
                            current = client.get('/api/jobs/abc').json()
                            if current['api_task']['status'] != 'processing': break
                            time.sleep(.01)
                        self.assertEqual(current['api_task']['status'],'complete', current['api_task'])

    def test_upload_evidence_split_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'sample.mp4'
            engine.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(source)])
            with patch.object(engine, 'DATA', root / 'data'):
                engine.DATA.mkdir()
                with TestClient(app) as client, source.open('rb') as video:
                    response = client.post('/api/jobs', files={'video': ('sample.mp4', video, 'video/mp4')})
                    self.assertEqual(response.status_code, 200, response.text)
                    job = response.json()
                    self.assertEqual(len(job['clips']), 1)
                    cid = job['clips'][0]['id']
                    evidence = client.get(f"/api/jobs/{job['id']}/clips/{cid}/evidence")
                    self.assertEqual(evidence.status_code, 200, evidence.text)
                    head = evidence.json()['source_head']
                    self.assertNotIn('path', head)
                    self.assertEqual(client.get(head['url']).status_code, 200)
                    result = client.post(f"/api/jobs/{job['id']}/clips/{cid}/split", json={'time_ms': 1000})
                    self.assertEqual(result.status_code, 200, result.text)
                    self.assertEqual([(c['start_ms'], c['end_ms']) for c in result.json()['clips']], [(0, 1000), (1000, 2000)])
                    self.assertEqual(len(client.get(f"/api/jobs/{job['id']}").json()['clips']), 2)
                    left = result.json()['clips'][0]['id']
                    merged = client.post(f"/api/jobs/{job['id']}/clips/{left}/merge")
                    self.assertEqual(merged.status_code, 200, merged.text)
                    self.assertEqual(len(merged.json()['clips']), 1)
                    cid = merged.json()['clips'][0]['id']
                    context = {'overview':'test', 'assets':[{'id':'person1','kind':'角色','description':'红衣','status':'已确认','clip_ids':[cid]}], 'states':[], 'edges':[]}
                    response = client.put(f"/api/jobs/{job['id']}/context",json={'context':context})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(client.get(f"/api/jobs/{job['id']}/context").json()['assets'][0]['id'], 'person1')
                    shot_id = job['shots'][0]['id']
                    self.assertEqual(client.post(f"/api/jobs/{job['id']}/shots/{shot_id}/subject",json={'identifier':'person1','description':'红衣'}).status_code, 200)
                    sample = client.post(f"/api/jobs/{job['id']}/shots/{shot_id}/sampling",json={'strategy':'等间隔'})
                    self.assertEqual(sample.status_code, 200, sample.text)
                    self.assertEqual(client.get(sample.json()[0]['url']).status_code, 200)
                    self.assertEqual(client.get(f"/api/jobs/{job['id']}/plan").status_code, 200)
                    self.assertEqual(client.get(f"/api/jobs/{job['id']}/video").status_code, 200)
                    self.assertEqual(client.get('/api/jobs/not-valid').status_code, 404)
                    self.assertEqual(client.post('/api/jobs', files={'video': ('x.txt', b'bad')}).status_code, 400)
                    self.assertEqual(client.get('/api/jobs/not-valid', headers={'Origin': 'https://example.com'}).status_code, 403)

if __name__ == '__main__': unittest.main()
