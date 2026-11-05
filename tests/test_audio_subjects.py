import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
import engine, audio, subjects

class ConfigTests(unittest.TestCase):
    def test_local_key_fallback_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine,'ROOT',Path(tmp)), patch.dict(os.environ,{'SILICONFLOW_API_KEY':''}):
            (Path(tmp)/'.env').write_text('SILICONFLOW_API_KEY=local-test-key\n')
            self.assertEqual(engine.connection('硅基流动')['key'],'local-test-key')
            self.assertEqual(engine.connection('硅基流动',key='explicit')['key'],'explicit')
            os.environ['SILICONFLOW_API_KEY']='environment'
            self.assertEqual(engine.connection('硅基流动')['key'],'environment')

class AudioTests(unittest.TestCase):
    def test_transcription_persists_and_does_not_save_key(self):
        job={'_transient':True,'id':'abc','video':{'file':'source.mp4','audio_track_present':True,'audio_status':'not_analyzed'}}
        response=httpx.Response(200,json={'text':'你好'},request=httpx.Request('POST','https://api.siliconflow.cn/v1/audio/transcriptions'))
        def fake_run(args): Path(args[-1]).write_bytes(b'audio')
        with tempfile.TemporaryDirectory() as tmp,patch.object(engine,'DATA',Path(tmp)),patch.object(engine,'run',side_effect=fake_run),patch('audio.httpx.post',return_value=response) as request:
            result=audio.transcribe(job,'secret-test')
        self.assertEqual(result['text'],'你好')
        self.assertEqual(result['timestamps'],'unavailable')
        self.assertEqual(job['video']['audio_status'],'transcribed')
        self.assertNotIn('secret-test',str(job))
        self.assertEqual(request.call_args.kwargs['data']['model'],'FunAudioLLM/SenseVoiceSmall')

    def test_failure_does_not_overwrite_prior_transcript(self):
        job={'_transient':True,'id':'abc','video':{'file':'source.mp4','audio_track_present':True},'audio_transcript':{'text':'previous'}}
        response=httpx.Response(503,text='secret-test',request=httpx.Request('POST','https://api.siliconflow.cn/v1/audio/transcriptions'))
        def fake_run(args): Path(args[-1]).write_bytes(b'audio')
        with tempfile.TemporaryDirectory() as tmp,patch.object(engine,'DATA',Path(tmp)),patch.object(engine,'run',side_effect=fake_run),patch('audio.httpx.post',return_value=response):
            with self.assertRaises(ValueError) as exc: audio.transcribe(job,'secret-test')
        self.assertNotIn('secret-test',str(exc.exception))
        self.assertEqual(job['audio_transcript']['text'],'previous')

    def test_absent_audio_never_calls_service(self):
        with patch('audio.httpx.post') as request:
            with self.assertRaises(ValueError): audio.transcribe({'video':{'audio_track_present':False}},'key')
            request.assert_not_called()

class SubjectTests(unittest.TestCase):
    def test_link_reuse_update_and_unlink(self):
        job={'_transient':True,'shots':[{'status':'complete'},{'status':'complete'}]}
        subjects.link(job,0,'child_1','红色雪服')
        subjects.link(job,1,'child_1','红色雪服')
        self.assertEqual(job['shots'][0]['subject_ids'],['child_1'])
        job['shots'][0]['status']='complete';job['shots'][1]['status']='complete'
        subjects.link(job,1,'child_1','蓝色雪服')
        self.assertEqual([s['status'] for s in job['shots']],['edited','edited'])
        subjects.unlink(job,0,'child_1')
        self.assertEqual(job['shots'][0]['subject_ids'],[])
        self.assertEqual(job['shots'][1]['subject_ids'],['child_1'])
        self.assertIn('child_1',job['subjects'])

    def test_generation_only_includes_user_linked_subjects(self):
        job={'_transient':True,'subjects':{'child_1':{'description':'红色雪服'},'other':{'description':'蓝色外套'}},'shots':[{'start_ms':0,'end_ms':2000,'reference_ms':0,'overrides':{},'subject_ids':['child_1'],'observation':{'windows':[{'scene':'雪地'}]}}]}
        output={'text_zh':'雪地','text_en':'snow','image_zh':'滑行','image_en':'slides','coverage':['w1']}
        with patch('engine.call_model',return_value=output) as call:
            engine.generate(job,0,{'provider':'test','model':'test'})
        instruction=call.call_args.args[1]
        self.assertIn('child_1',instruction)
        self.assertNotIn('蓝色外套',instruction)

class AudioEditPreservationTests(unittest.TestCase):
    def test_retranscription_keeps_user_edit(self):
        job={'_transient':True,'id':'abc','video':{'file':'source.mp4','audio_track_present':True},'audio_transcript':{'text':'旧识别','edited_text':'用户修订'}}
        response=httpx.Response(200,json={'text':'新识别'},request=httpx.Request('POST','https://api.siliconflow.cn/v1/audio/transcriptions'))
        def fake_run(args): Path(args[-1]).write_bytes(b'audio')
        with tempfile.TemporaryDirectory() as tmp,patch.object(engine,'DATA',Path(tmp)),patch.object(engine,'run',side_effect=fake_run),patch('audio.httpx.post',return_value=response):
            result=audio.transcribe(job,'secret-test')
        self.assertEqual(result['text'],'新识别')
        self.assertEqual(result['edited_text'],'用户修订')

class ExportTests(unittest.TestCase):
    def test_export_contains_user_edits_and_no_local_configuration(self):
        import json,zipfile
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); source=folder/'video.mp4'; data=folder/'data';data.mkdir()
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x90:r=10:d=2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(source)])
            with patch.object(engine,'DATA',data):
                job=engine.create(str(source))
                subjects.link(job,0,'child','红色外套')
                job['audio_transcript']={'status':'complete','text':'初稿','edited_text':'用户修订','timestamps':'unavailable'}
                job['shots'][0]['reference_user_selected']=True
                with zipfile.ZipFile(engine.export(job)) as archive:
                    names=archive.namelist()
                    self.assertTrue(any(name.startswith('references/') for name in names))
                    self.assertNotIn('.env',names)
                    self.assertIn('用户修订',archive.read('prompts.md').decode())
                    self.assertEqual(json.loads(archive.read('analysis.json'))['subjects']['child']['description'],'红色外套')

class CredentialDestinationTests(unittest.TestCase):
    def test_saved_provider_key_is_not_sent_to_an_unrelated_host(self):
        with patch.dict(os.environ,{'SILICONFLOW_API_KEY':'saved-secret'}):
            with self.assertRaisesRegex(ValueError,'明确输入专用密钥'):
                engine.connection('硅基流动',base_url='https://different.example/v1')
            c=engine.connection('硅基流动',base_url='https://different.example/v1',key='explicit-key')
            self.assertEqual(c['key'],'explicit-key')
