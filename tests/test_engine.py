import unittest
import engine

class Tests(unittest.TestCase):
    def test_sampling(self):
        values=engine.times(1000,2300)
        self.assertEqual(values,sorted(set(values)))
        self.assertTrue(all(1000<=v<2300 for v in values))
        self.assertIn(1000,values)
    def test_schema(self):
        with self.assertRaises(Exception): engine.Prompts.model_validate({'text_zh':'x'})

if __name__=='__main__': unittest.main()

class WorkflowTests(unittest.TestCase):
    def test_video_to_export_with_stubbed_model(self):
        import json, tempfile, zipfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = folder / 'source.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=160x90:r=10:d=16','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)])
            def fake_model(job, instruction, images, schema, config=None):
                if schema is engine.Observation:
                    return engine.Observation(subject='红色画面').model_dump()
                return engine.Prompts(text_zh='红色画面',text_en='red frame',image_zh='静止',image_en='static').model_dump()
            with patch.object(engine, 'DATA', folder / 'data'), patch.object(engine, 'call_model', side_effect=fake_model):
                engine.DATA.mkdir()
                job = engine.create(str(video))
                engine.analyze(job,0)
                self.assertEqual(job['shots'][0]['status'],'complete')
                self.assertTrue(engine.frame(job,0).exists())
                before=job['shots'][0]['prompts'].copy()
                result=engine.compare(job,0,{'provider':'test','model':'mock','key':'secret'})
                self.assertEqual(result['等间隔']['status'],'complete')
                self.assertEqual(result['变化驱动']['status'],'complete')
                self.assertEqual(job['shots'][0]['prompts'],before)
                self.assertLess(result['变化驱动']['sampling_frame_count'],result['等间隔']['sampling_frame_count'])
                import clips
                clip=clips.ensure(job)[0]
                with zipfile.ZipFile(engine.export(job)) as archive:
                    self.assertIn('prompts.md',archive.namelist())
                    self.assertIn('reconstruction.md',archive.namelist())
                    plan=json.loads(archive.read('reconstruction.json'))
                    self.assertEqual(plan['shots'][0]['routes']['text_to_video']['prompt_zh'],before['text_zh'])
                    self.assertIn(plan['shots'][0]['reference']['export_path'],archive.namelist())
                    manifest=json.loads(archive.read('clips.json'))
                    self.assertEqual(manifest[0]['clip']['id'],clip['id'])
                    for field in ('source_head','source_tail'):
                        self.assertIn(manifest[0]['evidence'][field]['path'],archive.namelist())
                    self.assertEqual(json.loads(archive.read('analysis.json'))['id'],job['id'])

class ProviderTests(unittest.TestCase):
    def test_custom_base_url_and_local_no_key(self):
        c=engine.connection('自定义（OpenAI 兼容）','vision-model','http://127.0.0.1:11434/v1','')
        self.assertEqual(c['url'],'http://127.0.0.1:11434/v1/chat/completions')
        self.assertEqual(c['key'],'')
        with self.assertRaises(ValueError):
            engine.connection('自定义（OpenAI 兼容）','model','http://example.com/v1','secret')
        with self.assertRaises(ValueError):
            engine.connection('自定义（OpenAI 兼容）','model','https://example.com/v1?key=secret','secret')

    def test_openai_request_and_secret_not_saved(self):
        from unittest.mock import patch
        import httpx, json
        config=engine.connection('自定义（OpenAI 兼容）','vision','https://api.example.com/v1','my-secret')
        job={'provenance':{}}
        data={'choices':[{'message':{'content':json.dumps({'text_zh':'中','text_en':'en','image_zh':'动','image_en':'move'})}}], 'usage':{'total_tokens':10}}
        response=httpx.Response(200,json=data,request=httpx.Request('POST',config['url']))
        with patch('engine.httpx.post',return_value=response) as post:
            result=engine.call_model(job,'test',[],engine.Prompts,config)
        self.assertEqual(result['image_en'],'move')
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'],'Bearer my-secret')
        self.assertNotIn('my-secret',json.dumps(job))
        self.assertEqual(job['usage'][0]['model'],'vision')

    def test_image_payload_and_safe_errors(self):
        from unittest.mock import patch
        from pathlib import Path
        import tempfile, httpx
        config=engine.connection('OpenAI','vision','','test-secret')
        job={'provenance':{}}
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'frame.jpg'; f.write_bytes(b'fake image')
            response=httpx.Response(401,text='test-secret',request=httpx.Request('POST',config['url']))
            with patch('engine.frame',return_value=f), patch('engine.httpx.post',return_value=response) as post:
                with self.assertRaises(ValueError) as ctx:
                    engine.call_model(job,'test',[500],engine.Observation,config)
            self.assertNotIn('test-secret',str(ctx.exception))
            content=post.call_args.kwargs['json']['messages'][0]['content']
            self.assertEqual(content[1]['text'],'原视频时间 500 毫秒')
            self.assertTrue(content[2]['image_url']['url'].startswith('data:image/jpeg;base64,'))

class SamplingComparisonTests(unittest.TestCase):
    def test_adaptive_sampling_static_and_movement(self):
        import tempfile
        from pathlib import Path
        import sampling
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=8','-c:v','libx264','-pix_fmt','yuv420p','-y',str(path)])
            marks=sampling.plan(path,1000,7000,'变化驱动',[1234])
            self.assertTrue(any(m['ms']==1234 and '人工补帧' in m['reasons'] for m in marks))
            self.assertTrue(any('画面变化' in m['reasons'] for m in marks))
            self.assertTrue(all(1000 <= m['ms'] < 7000 for m in marks))
            self.assertEqual([m['ms'] for m in marks], sorted({m['ms'] for m in marks}))

    def test_original_sampling_matches_previous(self):
        import sampling
        marks=sampling.plan('unused',102,5700,'等间隔')
        self.assertEqual([m['ms'] for m in marks],engine.times(102,5700))

    def test_montage_preserves_full_portrait_and_time_order(self):
        from io import BytesIO
        from PIL import Image
        import montage
        pic=Image.new('RGB',(20,100),'red')
        buf=BytesIO();pic.save(buf,format='PNG')
        result=Image.open(BytesIO(montage.create_sheet([(500,buf.getvalue()),(900,buf.getvalue())])))
        self.assertEqual(result.size,(1024,640))
        self.assertGreater(result.getpixel((256,170))[0],200)
        self.assertLess(result.getpixel((5,170))[0],100)
        self.assertGreater(result.getpixel((768,170))[0],200)

    def test_comparison_keeps_original_results(self):
        import copy, tempfile
        from pathlib import Path
        from unittest.mock import patch
        job={'id':'a','shots':[{'prompts':{'text_zh':'original'},'observation':{'saved':True},'manual_prompts':{'text_zh':'manual'}}]}
        def fake_analyze(clone,index,config):
            self.assertTrue(clone['_transient'])
            self.assertNotIn('manual_prompts',clone['shots'][index])
            clone['shots'][index]['prompts']={'text_zh':config['sampling_strategy']}
            clone['shots'][index]['observation']={'evidence_ms':[1,2]}
            clone['shots'][index]['status']='complete'
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine,'DATA',Path(tmp)), patch.object(engine,'analyze',side_effect=fake_analyze):
            (Path(tmp)/'a').mkdir()
            results=engine.compare(job,0,{'provider':'test','model':'test','key':'secret'})
            self.assertEqual(job['shots'][0]['prompts'],{'text_zh':'original'})
            self.assertEqual(job['shots'][0]['observation'],{'saved':True})
            self.assertEqual(results['变化驱动']['prompts']['text_zh'],'变化驱动')
            self.assertNotIn('secret',(Path(tmp)/'a'/'job.json').read_text())

class EmptyResponseTests(unittest.TestCase):
    def test_empty_response_is_retried(self):
        from unittest.mock import patch
        empty=engine.Observation().model_dump()
        valid=engine.Observation(scene='蓝色画面').model_dump()
        with patch('engine._call_model_once',side_effect=[empty,valid]) as call:
            result=engine.call_model({},'describe',[],engine.Observation,{'provider':'test'})
        self.assertEqual(result['scene'],'蓝色画面')
        self.assertEqual(call.call_count,2)
    def test_empty_prompt_fails_after_retry(self):
        from unittest.mock import patch
        empty=engine.Prompts(text_zh='',text_en='',image_zh='',image_en='').model_dump()
        with patch('engine._call_model_once',return_value=empty), self.assertRaisesRegex(ValueError,'返回空'):
            engine.call_model({},'describe',[],engine.Prompts,{'provider':'test'})

class FencedJsonTests(unittest.TestCase):
    def test_fenced_json_is_accepted(self):
        from unittest.mock import patch
        import httpx,json
        config=engine.connection('自定义（OpenAI 兼容）','vision','https://api.siliconflow.cn/v1','test')
        content='```json\n'+json.dumps({'text_zh':'中','text_en':'en','image_zh':'动','image_en':'move'})+'\n```'
        response=httpx.Response(200,json={'choices':[{'message':{'content':content}}]},request=httpx.Request('POST',config['url']))
        with patch('engine.httpx.post',return_value=response) as post:
            self.assertEqual(engine.call_model({'provenance':{}},'test',[],engine.Prompts,config)['text_zh'],'中')
        self.assertEqual(post.call_args.kwargs['json']['response_format'],{'type':'json_object'})

class AssemblyTests(unittest.TestCase):
    def test_multi_window_instructs_model_to_cover_changes(self):
        from unittest.mock import patch
        job={'_transient':True,'shots':[{'start_ms':0,'end_ms':4000,'reference_ms':0,'overrides':{},'observation':{'windows':[{'scene':'彩条'},{'scene':'蓝色'}]}}]}
        value={'text_zh':'彩条后蓝色','text_en':'stripes then blue','image_zh':'切换','image_en':'changes'}
        with patch('engine.call_model',return_value=value) as call:
            engine.generate(job,0,{'provider':'test','model':'test'})
        self.assertIn('不得只描述第一个窗口',call.call_args.args[1])
        self.assertIn('不代表新主体出现',call.call_args.args[1])
        self.assertEqual(job['shots'][0]['prompts']['text_zh'],'彩条后蓝色')

class ShortVideoTests(unittest.TestCase):
    def test_short_video_is_accepted(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);video=folder/'short.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=160x90:r=10:d=2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)])
            with patch.object(engine,'DATA',folder/'data'):
                engine.DATA.mkdir()
                job=engine.create(str(video))
                self.assertEqual(job['video']['duration_ms'],2000)
                self.assertEqual(job['shots'][0]['end_ms'],2000)

class TimestampRetryTests(unittest.TestCase):
    def test_out_of_range_beats_are_retried_not_clamped(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':6000},'shots':[{'start_ms':0,'end_ms':6000,'reference_ms':0,'overrides':{}}]}
        invalid=engine.Observation(beats=[engine.Beat(start_ms=0,end_ms=7000,evidence_ms=[0],action='滑行')]).model_dump()
        valid=engine.Observation(scene='雪地',beats=[engine.Beat(start_ms=0,end_ms=6000,evidence_ms=[0],action='滑行')]).model_dump()
        with patch('engine.sample_plan',return_value=[{'ms':0,'reasons':['基础']}]),patch('engine.call_model',side_effect=[invalid,valid]) as model,patch('engine.frame'),patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(job,0,{})
        self.assertEqual(model.call_count,2)
        self.assertIn('上次时间不合法',model.call_args.args[1])
        self.assertEqual(job['shots'][0]['observation']['windows'][0]['beats'][0]['end_ms'],6000)

class ResumeTests(unittest.TestCase):
    def test_supplement_timeout_returns_labeled_partial_prompt_and_resumes(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':2000,'sha256':'hash'},'shots':[{'start_ms':0,'end_ms':2000,'reference_ms':0,'overrides':{}}]}
        plan=[{'ms':0,'reasons':['基础']},{'ms':1000,'reasons':['基础']}]
        primary=engine.Observation(scene='街道',needs_review_ranges=[[0,1500]]).model_dump()
        supplemental=engine.Observation(scene='街道',beats=[engine.Beat(start_ms=0,end_ms=1000,action='转身',evidence_ms=[0,1000])]).model_dump()
        prompts=engine.Prompts(text_zh='街道',text_en='street',image_zh='街道',image_en='street').model_dump()
        cfg={'provider':'test','model':'A','url':'https://example.com/v1','key':'secret'}
        with patch('engine.sample_plan',return_value=plan),patch('engine.reference_frame'),patch('engine.call_model',side_effect=[primary,engine.ModelTimeoutError('timeout'),prompts]) as api:
            engine.analyze(job,0,cfg)
            shot=job['shots'][0]
            self.assertEqual(shot['status'],'complete_partial')
            self.assertTrue(shot['prompts']['text_zh'])
            self.assertIn('补看请求超时',str(shot['observation']['incomplete_review']))
            self.assertIn('incomplete_review',api.call_args.args[1])
        with patch('engine.sample_plan',return_value=plan),patch('engine.reference_frame'),patch('engine.call_model',side_effect=[supplemental,prompts]) as api:
            engine.analyze(job,0,cfg,resume=True)
            self.assertEqual(api.call_count,2)
            self.assertEqual(job['shots'][0]['status'],'complete')
            self.assertNotIn('incomplete_review',job['shots'][0]['observation'])

    def test_generation_timeout_reports_stage_and_reuses_primary(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':2000,'sha256':'hash'},'shots':[{'start_ms':0,'end_ms':2000,'reference_ms':0,'overrides':{}}]}
        plan=[{'ms':0,'reasons':['基础']}]
        primary=engine.Observation(scene='街道').model_dump()
        prompts=engine.Prompts(text_zh='街道',text_en='street',image_zh='街道',image_en='street').model_dump()
        cfg={'provider':'test','model':'A','url':'https://example.com/v1','key':'secret'}
        with patch('engine.sample_plan',return_value=plan),patch('engine.reference_frame'),patch('engine.call_model',side_effect=[primary,engine.ModelTimeoutError('timeout')]):
            with self.assertRaisesRegex(ValueError,'生成 Prompt.*timeout'): engine.analyze(job,0,cfg)
            self.assertIsNotNone(job['shots'][0]['observation'])
        with patch('engine.sample_plan',return_value=plan),patch('engine.reference_frame'),patch('engine.call_model',return_value=prompts) as api:
            engine.analyze(job,0,cfg,resume=True)
            self.assertEqual(api.call_count,1)
            self.assertEqual(job['shots'][0]['status'],'complete')

    def test_duplicate_review_ranges_are_not_repeated(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':6000,'sha256':'hash'},'shots':[{'start_ms':0,'end_ms':6000,'reference_ms':0,'overrides':{}}]}
        plan=[{'ms':i*200,'reasons':['基础']} for i in range(25)]
        primary=engine.Observation(scene='街道',needs_review_ranges=[[0,1000]]).model_dump()
        prompts=engine.Prompts(text_zh='街道',text_en='street',image_zh='街道',image_en='street').model_dump()
        cfg={'provider':'test','model':'A','url':'https://example.com/v1','key':'secret'}
        with patch('engine.sample_plan',return_value=plan),patch('engine.reference_frame'),patch('engine.call_model',side_effect=[primary,primary,engine.ModelTimeoutError('timeout'),prompts]):
            engine.analyze(job,0,cfg)
        self.assertEqual(len(job['shots'][0]['observation']['incomplete_review']),1)

    def test_resume_reuses_successful_window_and_invalidates_on_model_change(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':6000,'sha256':'hash'},'shots':[{'start_ms':0,'end_ms':6000,'reference_ms':0,'overrides':{}}]}
        plan=[{'ms':i*200,'reasons':['基础']} for i in range(25)]
        obs=engine.Observation(scene='雪地').model_dump()
        cfg={'provider':'test','model':'A','url':'https://example.com/v1','key':'secret'}
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',side_effect=[obs,ValueError('temporary')]) as api,patch('engine.reference_frame'),patch('engine.generate'):
            with self.assertRaisesRegex(ValueError,'temporary'): engine.analyze(job,0,cfg)
            self.assertEqual(api.call_count,2)
            self.assertNotIn('secret',str(job))
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',return_value=obs) as api,patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(job,0,cfg,resume=True)
            self.assertEqual(api.call_count,1)
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',return_value=obs) as api,patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(job,0,{**cfg,'model':'B'},resume=True)
            self.assertEqual(api.call_count,2)

class ReferenceTests(unittest.TestCase):
    def test_reference_frame_keeps_source_resolution(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);jobdir=folder/'abc';jobdir.mkdir()
            video=jobdir/'source.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=1280x720:r=5:d=1','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)])
            job={'id':'abc','video':{'duration_ms':1000,'file':'source.mp4'}}
            with patch.object(engine,'DATA',folder):
                with Image.open(engine.reference_frame(job,0)) as image: self.assertEqual(image.size,(1280,720))
                with Image.open(engine.frame(job,0)) as image: self.assertEqual(image.size,(768,432))

    def test_late_reference_excludes_finished_actions_and_flags_crossing(self):
        from unittest.mock import patch
        j={'_transient':True,'shots':[{'start_ms':0,'end_ms':6000,'reference_ms':3000,'overrides':{},'observation':{'windows':[{'scene':'雪地','beats':[{'start_ms':0,'end_ms':2000,'action':'起身'},{'start_ms':2000,'end_ms':4000,'action':'行走'},{'start_ms':4000,'end_ms':6000,'action':'转身'}]}]}}]}
        out={'text_zh':'文字','text_en':'text','image_zh':'动作','image_en':'action','coverage':['w1b1','w1b2']}
        with patch('engine.call_model',return_value=out) as model: engine.generate(j,0,{'provider':'test','model':'test'})
        instruction=model.call_args.args[1]
        self.assertIn('"crossing_or_uncertain": ["w1b2"]',instruction)
        self.assertEqual(j['shots'][0]['coverage_review']['missing_ids'],['w1b3'])
        self.assertIn('仅供人工核对',j['shots'][0]['coverage_review']['note'])

class ExplicitReferenceTests(unittest.TestCase):
    def test_selected_start_frame_is_not_replaced(self):
        from unittest.mock import patch
        j={'_transient':True,'id':'test','video':{'duration_ms':1000},'shots':[{'start_ms':0,'end_ms':1000,'reference_ms':0,'reference_user_selected':True,'overrides':{},'prompt_sampling':{'transitions':[]}}]}
        plan=[{'ms':125,'reasons':['基础'],'segment':0}]
        result=engine.Observation(scene='雪地').model_dump()
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',return_value=result),patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(j,0,{'sampling_strategy':'Prompt 专用','provider':'test','model':'test'})
        self.assertEqual(j['shots'][0]['reference_ms'],0)

class SupplementalCoverageTests(unittest.TestCase):
    def test_supplemental_evidence_is_included_in_review(self):
        from unittest.mock import patch
        j={'_transient':True,'shots':[{'start_ms':0,'end_ms':6000,'reference_ms':0,'overrides':{},'observation':{'windows':[{'scene':'雪地'}],'supplemental':[{'scene':'','beats':[{'start_ms':2000,'end_ms':3000,'action':'转身'}]}]}}]}
        out={'text_zh':'雪地','text_en':'snow','image_zh':'滑行','image_en':'slides','coverage':['w1']}
        with patch('engine.call_model',return_value=out): engine.generate(j,0,{'provider':'test','model':'test'})
        self.assertIn('s1b1',j['shots'][0]['coverage_review']['missing_ids'])

class RequestWindowTests(unittest.TestCase):
    def test_split_limits_overlap_and_preserves_distinct_shots(self):
        windows=[list(range(16)),list(range(50,55))]
        out=engine.split_windows(windows,8)
        self.assertEqual([len(w) for w in out],[8,8,4,5])
        self.assertEqual(out[0][-2:],out[1][:2])
        self.assertTrue(all(not(min(w)<50<=max(w)) for w in out))
        self.assertEqual(set(sum(out,[])),set(sum(windows,[])))
    def test_siliconflow_uses_smaller_window(self):
        self.assertEqual(engine.frame_request_limit({'provider':'硅基流动'}),8)
        self.assertEqual(engine.frame_request_limit({'url':'https://api.siliconflow.cn/v1/chat/completions'}),8)
        self.assertEqual(engine.frame_request_limit({'provider':'OpenAI'}),24)

class EndFrameTests(unittest.TestCase):
    def test_last_frame_can_be_extracted_and_sampling_is_aligned(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'test.mp4';data=root/'data';data.mkdir()
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=160x90:r=10:d=2.01',
                        '-c:v','libx264','-pix_fmt','yuv420p','-y',str(source)])
            with patch.object(engine,'DATA',data):
                job=engine.create(str(source))
                last=engine.last_frame_ms(job)
                self.assertLess(last,job['video']['duration_ms'])
                self.assertTrue(engine.frame(job,job['video']['duration_ms']-1).is_file())
                self.assertTrue(engine.reference_frame(job,job['video']['duration_ms']-1).is_file())
                plan=engine.sample_plan(job,0,'等间隔')
                self.assertTrue(all(x['ms']<=last for x in plan))
                self.assertTrue(all(engine.frame(job,x['ms']).is_file() for x in plan))

class CameraConflictTests(unittest.TestCase):
    def test_ambiguous_tilt_is_flagged_but_identified_part_is_not(self):
        issues=engine.review_prompts({'text_zh':'杯子向右下方倾斜。','image_zh':'杯口向左倾斜。'})
        self.assertEqual([issue['field'] for issue in issues],['text_zh'])

    def test_coherent_screen_motion_flags_static_camera_claim(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':1000},'shots':[{'start_ms':0,'end_ms':1000,'reference_ms':0,'overrides':{},'prompt_sampling':{'transitions':[]}}]}
        plan=[{'ms':i*200,'segment':0,'reasons':['基础'],'global_motion':2.0,'global_dx':-2.0,'global_dy':0.0,'local_motion':0.2} for i in range(4)]
        obs=engine.Observation(scene='街道',camera_observation='固定机位').model_dump()
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',return_value=obs),patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(job,0,{'sampling_strategy':'Prompt 专用','provider':'test','model':'test'})
        warnings=job['shots'][0]['observation']['camera_review']
        self.assertEqual(len(warnings),1)
        self.assertEqual(warnings[0]['screen_direction'],'向左')
        self.assertIn('可能冲突',warnings[0]['warning'])

    def test_english_fixed_camera_claim_is_flagged(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':1000},'shots':[{'start_ms':0,'end_ms':1000,'reference_ms':0,'overrides':{},'prompt_sampling':{'transitions':[]}}]}
        plan=[{'ms':i*200,'segment':0,'reasons':['基础'],'global_motion':2.0,'global_dx':-2.0,'global_dy':0.0,'local_motion':0.2} for i in range(4)]
        obs=engine.Observation(scene='street',camera_observation='Fixed high-angle camera').model_dump()
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',return_value=obs),patch('engine.reference_frame'),patch('engine.generate'):
            engine.analyze(job,0,{'sampling_strategy':'Prompt 专用','provider':'test','model':'test'})
        self.assertEqual(len(job['shots'][0]['observation']['camera_review']),1)

    def test_coherent_motion_and_overlapping_windows_reach_model_instructions(self):
        from unittest.mock import patch
        job={'_transient':True,'id':'test','video':{'duration_ms':1000},'shots':[{'start_ms':0,'end_ms':1000,'reference_ms':0,'overrides':{},'prompt_sampling':{'transitions':[]}}]}
        plan=[{'ms':i*100,'segment':0,'reasons':['基础'],'global_motion':2.,'global_dx':-2.,'global_dy':0.,'local_motion':.2} for i in range(10)]
        obs=engine.Observation(scene='街道').model_dump()
        prompts=engine.Prompts(text_zh='街道',text_en='street',image_zh='街道',image_en='street').model_dump()
        def respond(job,instruction,images,schema,config):
            return prompts.copy() if schema is engine.Prompts else obs.copy()
        with patch('engine.sample_plan',return_value=plan),patch('engine.call_model',side_effect=respond) as api,patch('engine.reference_frame'):
            engine.analyze(job,0,{'sampling_strategy':'Prompt 专用','provider':'硅基流动','model':'test'})
        calls=[c.args[1] for c in api.call_args_list]
        self.assertTrue(any('残余运动较弱' in instruction for instruction in calls))
        self.assertTrue(any('不要为了逐项覆盖 evidence_stages' in instruction for instruction in calls))
        self.assertTrue(any('不能单独证明人物行走' in instruction for instruction in calls))
