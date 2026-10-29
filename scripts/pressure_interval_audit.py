"""描画側の開眼採点と物理側の実測指幅を全物理時点で監査する。制御には使わない。"""
import math

OPEN_THRESHOLD = .9
TIME_TOLERANCE_S = 1e-8


def audit_pressure_interval(samples, *, start_time, end_time, command_gap_m,
                            press_threshold_m, physics_hz, open_threshold=OPEN_THRESHOLD):
    values = (start_time, end_time, command_gap_m, press_threshold_m, physics_hz, open_threshold)
    if (not all(math.isfinite(v) for v in values) or physics_hz <= 0
            or end_time <= start_time or command_gap_m <= 0 or press_threshold_m <= 0 or not 0<open_threshold<=1):
        raise ValueError('押圧監査の区間・閾値が不正です')
    intervals = (end_time-start_time)*physics_hz
    count = round(intervals)
    if count < 1 or abs(intervals-count) > TIME_TOLERANCE_S*physics_hz or len(samples) != count+1:
        raise ValueError('物理時点の全区間記録がありません')
    closed = commands = measured = 0
    for index, sample in enumerate(samples):
        try:
            t, openness, gap = (sample[key] for key in ('time', 'open', 'gap_m'))
        except KeyError as error:
            raise ValueError('押圧監査の記録項目が欠けています') from error
        if (not all(math.isfinite(v) for v in (t, openness, gap)) or not 0 <= openness <= 1
                or gap <= 0 or abs(t-(start_time+index/physics_hz)) > TIME_TOLERANCE_S):
            raise ValueError('押圧監査の時刻・開眼値・指幅が不正です')
        if openness < open_threshold:
            closed += 1
            commands += int(command_gap_m < press_threshold_m)
            measured += int(gap < press_threshold_m)
    return {'sample_count': len(samples), 'closed_samples': closed,
            'closed_command_samples': commands, 'closed_measured_samples': measured,
            'passed': commands == 0 and measured == 0}
