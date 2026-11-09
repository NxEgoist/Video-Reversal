import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clips
import engine


class ClipTests(unittest.TestCase):
    def job(self):
        return {'shots': [{'id': 's1', 'start_ms': 0, 'end_ms': 2000},
                          {'id': 's2', 'start_ms': 2000, 'end_ms': 3000}]}

    def test_split_merge_and_history(self):
        job = self.job()
        original = clips.ensure(job)[0]
        original['manual'] = 'preserve'
        a, b = clips.split(job, original['id'], 900)
        self.assertEqual([(c['start_ms'], c['end_ms']) for c in (a, b)], [(0, 900), (900, 2000)])
        self.assertEqual(job['clip_history'][0]['manual'], 'preserve')
        with self.assertRaises(ValueError): clips.merge(job, b['id'], job['clips'][-1]['id'])
        merged = clips.merge(job, a['id'], b['id'])
        self.assertEqual((merged['start_ms'], merged['end_ms']), (0, 2000))
        self.assertEqual(merged['source_clip_ids'], [a['id'], b['id']])

    def test_reset_archives_only_affected_shot(self):
        job = self.job()
        first, second = clips.ensure(job)
        clips.split(job, first['id'], 1000)
        job['shots'][0]['end_ms'] = 1800
        new = clips.reset_shot(job, 's1')
        self.assertEqual(new['end_ms'], 1800)
        self.assertEqual(job['clips'][1]['id'], second['id'])
        self.assertEqual(len(new['source_clip_ids']), 2)

    def test_preload_extracts_multiple_clips_in_one_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'abc').mkdir();video=root/'abc'/'source.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)])
            job={'id':'abc','video':{'file':'source.mp4'},'shots':[{'id':'s1','start_ms':0,'end_ms':2000}]}
            with patch.object(engine,'DATA',root):
                clips.split(job,clips.ensure(job)[0]['id'],1000)
                with patch.object(clips,'_run',wraps=clips._run) as commands:
                    clips.preload(job)
                    first=clips.evidence(job,job['clips'][0]['id'])
                    second=clips.evidence(job,job['clips'][1]['id'])
                self.assertEqual(sum(c.args[0][0]=='ffmpeg' for c in commands.call_args_list),1)
                self.assertEqual(first['source_tail']['time_ms'],900)
                self.assertEqual(second['source_head']['time_ms'],1000)

    def test_real_frames_use_half_open_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'abc').mkdir()
            video = root/'abc'/'source.mp4'
            engine.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=s=160x90:r=10:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(video)])
            job = {'id': 'abc', 'video': {'file': 'source.mp4'},
                   'shots': [{'id': 's1', 'start_ms': 0, 'end_ms': 2000}]}
            with patch.object(engine, 'DATA', root):
                with patch.object(clips, '_run', wraps=clips._run) as probes:
                    clips.frame_index(job)
                    clips.frame_index(job)
                self.assertEqual(sum(call.args[0][0]=='ffprobe' for call in probes.call_args_list),1)
                import os
                st=video.stat()
                os.utime(video,ns=(st.st_atime_ns,st.st_mtime_ns+1000000000))
                with patch.object(clips,'_run',wraps=clips._run) as probes:
                    clips.frame_index(job)
                self.assertEqual(sum(call.args[0][0]=='ffprobe' for call in probes.call_args_list),1)
                left, right = clips.split(job, clips.ensure(job)[0]['id'], 1100)
                with patch.object(clips, '_run', wraps=clips._run) as commands:
                    a = clips.evidence(job, left['id'])
                ffmpeg_calls=[call for call in commands.call_args_list if call.args[0][0]=='ffmpeg']
                self.assertEqual(len(ffmpeg_calls),1)
                b = clips.evidence(job, right['id'])
                self.assertEqual(a['source_head']['time_ms'], 0)
                self.assertEqual(a['source_tail']['time_ms'], 1000)
                self.assertEqual(b['source_head']['time_ms'], 1100)
                self.assertEqual(b['source_tail']['time_ms'], 1900)
                self.assertTrue(Path(a['source_tail']['path']).exists())
                baseline=root/'baseline.jpg'
                clips._run(['ffmpeg','-v','error','-i',str(video),'-vf',f"select=eq(n\\,{a['source_tail']['index']})",'-vsync','0','-frames:v','1','-q:v','2','-y',str(baseline)])
                self.assertEqual(Path(a['source_tail']['path']).read_bytes(),baseline.read_bytes())
                self.assertTrue(all(0 <= f['time_ms'] < 1100 for f in a['tail_frames']))


if __name__ == '__main__': unittest.main()
