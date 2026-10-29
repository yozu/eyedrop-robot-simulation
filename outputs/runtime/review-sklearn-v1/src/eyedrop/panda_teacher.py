"""現在の状態と過去の動作で切り替える教師。絶対時刻は使わない。"""
from .panda_world import BOTTLE_DEPTH_M,ACTION_HZ

MIN_GAP_M=.0129
RELEASE_TRIGGER_GAP_M=.01405
GAP_SETTLED_M=.0001
OPEN_MIN=.9
RETREAT_READY_GAP_M=BOTTLE_DEPTH_M+.001
START_OPEN_MIN_STEPS=round(.6*ACTION_HZ)
START_OPEN_MAX_STEPS=round(1.*ACTION_HZ)


class ReactiveTeacher:
    def __init__(self,rest_gap,*,wait_for_blink=False):
        self.rest_gap=rest_gap
        self.phase='prepare'
        self.wait_for_blink=wait_for_blink
        self.blink_observed=False
        self.open_steps=0

    def decide(self,*,openness,gap,aligned):
        self.open_steps=self.open_steps+1 if openness>=OPEN_MIN else 0
        if self.phase=='prepare':
            # 回復実演では、実際に観測した閉眼からの再開を待つ。未来の周期は使わない。
            if self.wait_for_blink:
                if openness<OPEN_MIN:
                    self.blink_observed=True
                    return 'wait',self.rest_gap
                if (not self.blink_observed
                    or not START_OPEN_MIN_STEPS<=self.open_steps<=START_OPEN_MAX_STEPS):
                    return 'target',BOTTLE_DEPTH_M
            if openness<OPEN_MIN:return 'hold',gap
            if aligned and gap<=BOTTLE_DEPTH_M+GAP_SETTLED_M:
                self.phase='press'
                return 'target',MIN_GAP_M
            return 'target',BOTTLE_DEPTH_M
        if self.phase=='press':
            if openness<OPEN_MIN:
                self.phase='abort'
                return 'hold',BOTTLE_DEPTH_M
            if gap<=RELEASE_TRIGGER_GAP_M:
                self.phase='release'
                return 'target',self.rest_gap
            return 'target',MIN_GAP_M
        if self.phase=='abort':
            if gap>=BOTTLE_DEPTH_M-GAP_SETTLED_M:self.phase='prepare'
            return 'hold',BOTTLE_DEPTH_M
        if self.phase=='release':
            if gap>=RETREAT_READY_GAP_M:
                self.phase='retreat'
                return 'wait',self.rest_gap
            return 'target',self.rest_gap
        if self.phase=='retreat':return 'wait',self.rest_gap
        raise RuntimeError('教師の状態が不正です')
