import copy
import unittest

import clip_context


class ClipContextTests(unittest.TestCase):
    def setUp(self):
        self.job = {}
        self.clips = [{'id': 'a', 'start_ms': 0, 'end_ms': 1000},
                      {'id': 'b', 'start_ms': 1000, 'end_ms': 2000}]
        self.value = {'overview': '雪地',
                      'assets': [{'id': 'child', 'kind': '角色', 'description': '红衣', 'status': '已确认',
                                  'clip_ids': ['a', 'b'], 'evidence': [{'clip_id': 'a', 'time_ms': 100}]}],
                      'states': [{'clip_id': 'a', 'asset_id': 'child', 'start': '坐下', 'end': '滑行', 'status': '已确认'}],
                      'edges': [{'from_clip': 'a', 'to_clip': 'b', 'type': '硬切', 'review': '待核对', 'note': '确认人物'}]}

    def test_confirmation_and_no_mutable_alias(self):
        result = clip_context.save(self.job, self.clips, self.value)
        result['assets'][0]['description'] = 'changed'
        self.assertEqual(self.job['clip_context']['assets'][0]['description'], '红衣')
        self.assertEqual(clip_context.for_clip(self.job, self.clips, 'a')['assets'][0]['id'], 'child')
        self.assertEqual(clip_context.for_clip(self.job, self.clips, 'b')['states'], [])
        self.value['assets'][0]['status'] = '候选'
        clip_context.save(self.job, self.clips, self.value)
        self.assertEqual(clip_context.for_clip(self.job, self.clips, 'a')['assets'], [])
        self.assertEqual(self.job['clip_context']['version'], 2)

    def test_bad_evidence_does_not_replace_saved_context(self):
        clip_context.save(self.job, self.clips, self.value)
        before = copy.deepcopy(self.job)
        self.value['assets'][0]['evidence'][0]['time_ms'] = 1000
        with self.assertRaises(ValueError): clip_context.save(self.job, self.clips, self.value)
        self.assertEqual(self.job, before)

    def test_boundary_changes_mark_stale_and_cannot_silently_reconfirm(self):
        clip_context.save(self.job, self.clips, self.value)
        self.clips[0]['end_ms'] = 900
        context = clip_context.read(self.job, self.clips)
        self.assertTrue(context['states'][0]['stale'])
        self.assertTrue(context['edges'][0]['stale'])
        self.assertEqual(clip_context.for_clip(self.job, self.clips, 'a')['states'], [])
        with self.assertRaises(ValueError): clip_context.save(self.job, self.clips, context)

    def test_removed_clip_and_nonadjacent_edge(self):
        clip_context.save(self.job, self.clips, self.value)
        context = clip_context.read(self.job, self.clips[:1])
        self.assertTrue(context['assets'][0]['stale'])
        self.assertTrue(context['edges'][0]['stale'])
        self.assertEqual(clip_context.for_clip(self.job, self.clips[:1], 'a')['assets'], [])
        with self.assertRaises(ValueError): clip_context.save(self.job, self.clips[:1], self.value)

    def test_no_automatic_continuity_claim(self):
        self.assertEqual(clip_context.read({}, self.clips)['edges'], [])
        self.assertEqual(clip_context.for_clip({}, self.clips, 'a')['assets'], [])
