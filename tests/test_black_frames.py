import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import black_frames
import clips
import engine
import sampling
from backend.api import app


class BlackScreenTests(unittest.TestCase):
    def test_detection_and_reversible_bulk_skip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(engine, 'DATA', Path(tmp)):
            folder = Path(tmp) / 'abc'
            folder.mkdir()
            path = folder / 'source.mp4'
            engine.run(['ffmpeg','-v','error','-f','lavfi','-i','color=black:s=160x90:r=10:d=1',
                        '-f','lavfi','-i','color=gray:s=160x90:r=10:d=1',
                        '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map','[v]',
                        '-c:v','libx264','-pix_fmt','yuv420p','-y',str(path)])
            job = {'id':'abc', 'video':{'file':'source.mp4','duration_ms':2000}, 'shots':[
                {'id':'black','start_ms':0,'end_ms':1000,'status':'pending'},
                {'id':'picture','start_ms':1000,'end_ms':2000,'status':'pending'}]}
            clips.ensure(job); engine.save(job)
            with TestClient(app) as client:
                result = client.post('/api/jobs/abc/black-screen-detection')
                self.assertEqual(result.status_code, 200, result.text)
                self.assertTrue(result.json()['shots'][0]['black_screen']['candidate'])
                self.assertFalse(result.json()['shots'][1]['black_screen']['candidate'])
                self.assertEqual(len(result.json()['clips']), 2)
                self.assertTrue(client.put('/api/jobs/abc/shots/black/bulk-skip', json={'skip':True}).json()['shots'][0]['skip_bulk'])
                calls = []
                def analyze(job, index, config, resume=False):
                    calls.append(job['shots'][index]['id'])
                from backend.api import run_other
                with patch.object(engine, 'analyze', side_effect=analyze):
                    run_other('abc','test','all',{},'')
                self.assertEqual(calls, ['picture'])
                self.assertFalse(client.put('/api/jobs/abc/shots/black/bulk-skip', json={'skip':False}).json()['shots'][0]['skip_bulk'])

    def test_fast_sampling_keeps_boundaries_and_manual_frames(self):
        fast = sampling.plan('unused', 0, 10000, '快速（1 FPS）', [1234])
        normal = sampling.plan('unused', 0, 10000, '等间隔', [1234])
        marks = {item['ms'] for item in fast}
        self.assertTrue({0, 5000, 9900, 1234} <= marks)
        self.assertLess(len(fast), len(normal))


if __name__ == '__main__':
    unittest.main()
