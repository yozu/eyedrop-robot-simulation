"""画像判定と実測指幅だけで固定押圧を切り替える。"""
import math

OPEN_THRESHOLD = .95
BOTTLE_GAP_M = .0159
PRESS_GAP_M = .0129
RELEASE_TRIGGER_M = .01405
GAP_TOLERANCE_M = .0001
RETREAT_GAP_M = BOTTLE_GAP_M + .001
ACTION_HZ = 20
OPEN_START_MIN = 12
OPEN_START_MAX = 20
ALIGN_FRAMES = 5
PRESS_TIMEOUT_STEPS = 2 * ACTION_HZ


class FixedEyeDrop:
    def __init__(self, *, rest_gap,press_steps_limit=None,standby_gap=None,continue_approach_after_first_open=False,guard_prepare_contact=False):
        if type(continue_approach_after_first_open) is not bool:raise ValueError('接近継続の指定が不正です')
        if type(guard_prepare_contact) is not bool or guard_prepare_contact and standby_gap is None:
            raise ValueError('準備接触の制限には非圧縮の待機幅を明示してください')
        if not math.isfinite(rest_gap) or rest_gap<=RETREAT_GAP_M:
            raise ValueError('解除幅が不正です')
        if press_steps_limit is not None and (type(press_steps_limit) is not int or not 0<press_steps_limit<=PRESS_TIMEOUT_STEPS):
            raise ValueError('押圧パルスの長さが不正です')
        if standby_gap is not None and (not math.isfinite(standby_gap) or not BOTTLE_GAP_M+GAP_TOLERANCE_M<standby_gap<=rest_gap):
            raise ValueError("非圧縮待機幅が範囲外です")
        self.standby_gap=rest_gap if standby_gap is None else standby_gap
        self.continue_approach_after_first_open=continue_approach_after_first_open
        self.guard_prepare_contact=guard_prepare_contact
        self.press_steps_limit=press_steps_limit
        self.rest_gap = rest_gap
        self.phase = 'prepare'
        self.seen_closed = False
        self.ever_opened = False
        self.open_count = self.aligned_count = self.press_steps = 0
        self.aborted = False

    def decide(self, *, openness, gap, aligned, position_valid=True):
        if (not math.isfinite(openness) or not 0<=openness<=1
                or not math.isfinite(gap) or gap<=0 or type(aligned) is not bool or type(position_valid) is not bool):
            raise ValueError('画像判定または自己状態が不正です')
        opened = openness>=OPEN_THRESHOLD
        self.ever_opened |= opened
        self.seen_closed |= not opened
        self.open_count = self.open_count+1 if opened and position_valid else 0
        self.aligned_count = self.aligned_count+1 if aligned and position_valid else 0
        if self.phase == 'prepare':
            ready = (self.seen_closed and OPEN_START_MIN<=self.open_count<=OPEN_START_MAX
                     and self.aligned_count>=ALIGN_FRAMES and abs(gap-BOTTLE_GAP_M)<=GAP_TOLERANCE_M)
            if ready:
                self.phase = 'press'
        elif self.phase == 'press':
            if not opened or not aligned or not position_valid:
                self.aborted = True
                self.phase = 'release'
            elif gap<=RELEASE_TRIGGER_M:
                self.phase = 'release'
            elif self.press_steps_limit is not None and self.press_steps>=self.press_steps_limit:
                self.phase = 'release'
            elif self.press_steps>=PRESS_TIMEOUT_STEPS:
                self.aborted = True
                self.phase = 'release'
        if self.phase == 'release' and gap>=RETREAT_GAP_M:
            self.phase = 'retreat'
        if self.phase == 'press':
            self.press_steps += 1
        gap_command = PRESS_GAP_M if self.phase=='press' else BOTTLE_GAP_M if self.phase=='prepare' and opened else self.rest_gap
        if self.phase=='prepare' and not opened and self.ever_opened:
            gap_command=self.standby_gap
        if self.guard_prepare_contact and self.phase=='prepare' and opened and (not aligned or self.open_count>OPEN_START_MAX):
            gap_command=self.standby_gap
        if not position_valid and self.phase=='prepare':
            gap_command=self.standby_gap if self.ever_opened else self.rest_gap
        approach=opened or self.continue_approach_after_first_open and self.ever_opened
        arm = 'target' if self.phase=='prepare' and approach and position_valid else 'retreat' if self.phase=='retreat' else 'hold'
        return {'phase': self.phase, 'gap': gap_command, 'arm': arm, 'aborted': self.aborted}
