"""学習動作の全区間で、閉眼中の指令・実測圧縮・残圧・流出を採点する。"""
PHYSICS_HZ = 240
OPEN_THRESHOLD = .95
PRESS_GAP_M = .0158
PRESSURE_EPSILON_KPA = 1e-9


def audit_fluid(samples, motion, *, expected_intervals):
    points = {round(s['time']*PHYSICS_HZ):s for s in samples}
    pressure = {}
    for row in motion:
        for time_key, bottle_key in (('start_time','bottle_before'), ('time','bottle_after')):
            pressure[round(row[time_key]*PHYSICS_HZ)] = row[bottle_key]['pressure_kpa']
    expected = set(range(expected_intervals+1))
    coverage = len(motion)==expected_intervals and set(points)==expected and set(pressure)==expected
    closed = [(tick,s) for tick,s in points.items() if s['open']<OPEN_THRESHOLD]
    command = sum(s['command_gap_m']<PRESS_GAP_M for _,s in closed)
    measured = sum(s['gap_m']<PRESS_GAP_M for _,s in closed)
    positive = sum(pressure[tick]>PRESSURE_EPSILON_KPA for tick,_ in closed) if coverage else None
    flow = sum(row['bottle_before']['reservoir_ml']-row['bottle_after']['reservoir_ml']
               for row in motion if points.get(round(row['time']*PHYSICS_HZ),{}).get('open',0.)<OPEN_THRESHOLD)
    return dict(coverage_complete=coverage,closed_command_samples=command,closed_measured_samples=measured,
                closed_positive_pressure_samples=positive,closed_flow_ml=flow,
                passed=bool(coverage and not command and not measured and positive==0 and flow==0.))
