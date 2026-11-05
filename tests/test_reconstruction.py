import copy
import unittest

import reconstruction


class ReconstructionTests(unittest.TestCase):
    def job(self):
        return {'id': 'abc', 'shots': [{'id': 'shot_001', 'start_ms': 1000,
                'end_ms': 6000, 'reference_ms': 1000, 'status': 'complete',
                'observation': {'windows': [{'beats': [{'start_ms': 2000, 'end_ms': 4000,
                                'action': '杯口向画面右侧倾斜', 'evidence_ms': [2000, 3000]}]}]},
                'prompts': dict(text_zh='人工中文', text_en='manual', image_zh='动作', image_en='motion')}]}

    def test_read_only_and_preserves_prompts_and_evidence(self):
        job = self.job()
        before = copy.deepcopy(job)
        plan = reconstruction.build(job)
        self.assertEqual(job, before)
        shot = plan['shots'][0]
        self.assertEqual(shot['routes']['text_to_video']['prompt_zh'], '人工中文')
        self.assertEqual(shot['evidence_stages'][0]['evidence_ms'], [2000, 3000])
        shot['evidence_stages'][0]['evidence_ms'].append(999)
        self.assertEqual(job, before)
        self.assertIsNone(plan['generation_settings']['seed'])

    def test_midshot_reference_does_not_claim_full_coverage(self):
        job = self.job()
        job['shots'][0]['reference_ms'] = 3500
        shot = reconstruction.build(job)['shots'][0]
        self.assertEqual(shot['routes']['image_to_video']['duration_ms'], 2500)
        self.assertEqual(shot['routes']['text_to_video']['duration_ms'], 5000)
        self.assertTrue(any('前 2.500 秒' in item for item in shot['review_items']))

    def test_pending_old_job_and_invalid_reference(self):
        job = self.job()
        shot = job['shots'][0]
        shot.update(observation=None, prompts=None, status='pending', reference_ms=6000)
        result = reconstruction.build(job)['shots'][0]
        self.assertIsNone(result['reference']['export_path'])
        self.assertIsNone(result['routes']['image_to_video']['duration_ms'])
        self.assertGreaterEqual(len(result['review_items']), 4)
        self.assertIn('参考帧无效', reconstruction.markdown(reconstruction.build(job)))

    def test_partial_conflicting_and_manual_results_remain_visible(self):
        job = self.job()
        shot = job['shots'][0]
        shot['observation'].update(incomplete_review=[{}], camera_review=[{'warning': '运镜冲突'}])
        shot['observation']['windows'][0]['unknowns'] = ['手部遮挡']
        shot['coverage_review'] = {'missing_ids': ['w1b1'], 'reference_uncertain_ids': ['w1b2']}
        shot['overrides'] = {'camera': '固定'}
        job['audio_transcript'] = {'text': '旧对白', 'edited_text': '修订对白'}
        text = reconstruction.markdown(reconstruction.build(job))
        for value in ('运镜冲突', '手部遮挡', 'w1b1', 'w1b2', '固定', '修订对白', '补看未完成'):
            self.assertIn(value, text)
        self.assertNotIn('旧对白', text)

    def test_subjects_are_only_user_linked(self):
        job = self.job()
        job['subjects'] = {'child': {'description': '红衣'}, 'other': {'description': '蓝衣'}}
        job['shots'][0]['subject_ids'] = ['child', 'deleted']
        self.assertEqual(reconstruction.build(job)['shots'][0]['subjects'], [{'id': 'child', 'description': '红衣'}])
