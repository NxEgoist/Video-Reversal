import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import engine,cases,app


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.data=self.root/'data';self.data.mkdir()
        self.patch=patch.object(engine,'DATA',self.data);self.patch.start()
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.patch.stop)
        source=self.root/'sample.mp4'
        engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=320x180:r=10:d=2',
                    '-c:v','libx264','-pix_fmt','yuv420p','-y',str(source)])
        self.job=engine.create(str(source))
        self.job['shots'][0]['observation']={'windows':[{'scene':'蓝色'}]}
        self.job['shots'][0]['prompts']={'text_zh':'原稿','text_en':'original','image_zh':'静止','image_en':'still'}
        self.job['shots'][0]['status']='complete'
        self.job['audio_transcript']={'text':'原始转写','status':'complete'}
        engine.save(self.job)

    def test_snapshot_is_independent_and_restores_video_and_reference(self):
        snapshot=app.case_snapshot(self.job['id'],'0','{}',0,2,0,'手工改稿','original','静止','still','修订对白','','')
        self.assertEqual(engine.load(self.job['id'])['shots'][0]['prompts']['text_zh'],'原稿')
        record=cases.save_case(snapshot,'测试案例',0)
        self.assertEqual(cases.list_cases()[0]['title'],'测试案例')
        self.job['shots'][0]['prompts']['text_zh']='后续修改';engine.save(self.job)
        restored,index=cases.restore_case(record['id'])
        self.assertNotEqual(restored['id'],self.job['id'])
        self.assertEqual(index,0)
        self.assertEqual(restored['shots'][0]['prompts']['text_zh'],'手工改稿')
        self.assertEqual(restored['shots'][0]['manual_prompts']['text_zh'],'手工改稿')
        self.assertEqual(restored['audio_transcript']['edited_text'],'修订对白')
        self.assertTrue((self.data/restored['id']/restored['video']['file']).is_file())
        self.assertTrue((self.data/restored['id']/'references'/'0.jpg').is_file())
        restored['shots'][0]['prompts']['text_zh']='副本修改';engine.save(restored)
        again,_=cases.restore_case(record['id'])
        self.assertEqual(again['shots'][0]['prompts']['text_zh'],'手工改稿')
        self.assertEqual(engine.load(self.job['id'])['shots'][0]['prompts']['text_zh'],'后续修改')
        self.assertFalse((self.data/'cases'/record['id']/'.env').exists())

    def test_restore_task_returns_its_own_video(self):
        restored=app.restore(self.job['id'])
        self.assertEqual(Path(restored[4]),self.data/self.job['id']/self.job['video']['file'])
        self.assertTrue(Path(restored[4]).exists())

    def test_reference_edit_marks_result_for_review(self):
        snapshot=app.case_snapshot(self.job['id'],'0','{}',0,2,1,'原稿','original','静止','still','原始转写','','')
        self.assertEqual(snapshot['shots'][0]['status'],'edited')
        self.assertTrue(snapshot['shots'][0]['reference_user_selected'])
        self.assertEqual(snapshot['shots'][0]['reference_ms'],1000)
        self.assertEqual(engine.load(self.job['id'])['shots'][0]['reference_ms'],0)

    def test_uncommitted_cut_and_subject_edits_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'切点'):
            app.case_snapshot(self.job['id'],'0','{}',0,1,0,'','','','','','','')
        with self.assertRaisesRegex(ValueError,'主体'):
            app.case_snapshot(self.job['id'],'0','{}',0,2,0,'','','','','','child','红衣')

    def test_invalid_case_ids_and_incomplete_saves(self):
        with self.assertRaises(ValueError):cases.restore_case('../outside')
        self.job['shots'][0]['status']='processing'
        with self.assertRaisesRegex(ValueError,'分析'):cases.save_case(self.job,'测试')
        self.assertEqual(cases.list_cases(),[])

if __name__=='__main__': unittest.main()
