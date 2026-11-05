import unittest
import numpy as np
from prompt_sampling import select, windows
from transitions import transition_ranges


def rows(start=0,end=24000):
    return [{'ms':t,'quality':2.,'change':0.,'local_motion':0.,'global_motion':0.} for t in range(start,end,125)]


class EvidenceTests(unittest.TestCase):
    def test_transition_exclusion_and_manual_override(self):
        plan=select(rows(0,6000),0,6000,[{'start_ms':2000,'end_ms':3000}],manual=[2500])
        self.assertEqual([r['ms'] for r in plan if 2000<=r['ms']<3000],[2500])
        self.assertIn('人工补帧（转场内）',next(r for r in plan if r['ms']==2500)['reasons'])
        for window in windows(plan):
            self.assertFalse(min(window)<2000 and max(window)>=3000)

    def test_budget_tail_and_triplets(self):
        data=rows()
        for r in data:
            r['local_motion']=2.;r['change']=5.
        plan=select(data,0,24000,[])
        for a in (0,12000): self.assertLessEqual(sum(a<=r['ms']<a+12000 for r in plan),24)
        self.assertGreater(plan[-1]['ms'],23000)
        self.assertTrue(any('动作前中后' in r['reasons'] for r in plan))
        self.assertTrue(all(len(w)<=24 for w in windows(plan)))

    def test_quality_choice_and_temporal_coverage(self):
        data=rows(0,6000);data[1]['quality']=10.
        plan=select(data,0,6000,[])
        self.assertIn(125,[r['ms'] for r in plan])
        self.assertTrue(all(b['ms']-a['ms']<=2250 for a,b in zip(plan,plan[1:])))

    def test_grouped_transition_timestamps(self):
        result=transition_ranges([0,40,80,120,160],[0,.7,.8,0,0],[0,0,0,0,.9],200)
        self.assertEqual([(r['start_ms'],r['end_ms']) for r in result],[(40,120),(160,200)])

    def test_overlap_does_not_cross_segments(self):
        plan=[{'ms':i*100,'segment':0 if i<30 else 1} for i in range(60)]
        chunks=windows(plan)
        self.assertEqual(chunks[0][-2:],chunks[1][:2])
        self.assertTrue(all(not(min(c)<3000<=max(c)) for c in chunks))

class DiversityTests(unittest.TestCase):
    def test_continuous_motion_remains_temporally_distributed(self):
        data=rows(0,6000)
        for item in data:
            item['local_motion']=3.;item['change']=5.
        result=select(data,0,6000,[])
        times=[item['ms'] for item in result]
        self.assertLess(len(times),24)
        self.assertTrue(all(any(t<=x<t+2000 for x in times) for t in (0,2000,4000)))
        self.assertLessEqual(max(b-a for a,b in zip(times,times[1:])),2250)

class MotionEvidenceTests(unittest.TestCase):
    def test_coherent_screen_motion_and_stationary_are_distinguished(self):
        from prompt_sampling import motion_evidence
        moving=[{'ms':i*100,'global_motion':2,'global_dx':-2,'global_dy':0.1,'local_motion':0.2} for i in range(5)]
        evidence=motion_evidence(moving,[0,100,200,300,400])
        self.assertEqual(evidence['status'],'coherent_translation')
        self.assertEqual(evidence['screen_direction'],'向左')
        self.assertTrue(evidence['predominantly_coherent'])
        stationary=[{**x,'global_motion':0.01,'global_dx':0.01} for x in moving]
        self.assertEqual(motion_evidence(stationary,[0,100,200,300,400])['status'],'insufficient')
    def test_motion_evidence_does_not_assume_physical_camera_move(self):
        from prompt_sampling import motion_evidence
        moving=[{'ms':i*100,'global_motion':2,'global_dx':-2,'global_dy':0,'local_motion':5} for i in range(4)]
        evidence=motion_evidence(moving,[0,100,200,300])
        self.assertIn('不足以单独确定',evidence['description'])
        self.assertFalse(evidence['predominantly_coherent'])
