"""線形まばたきの三角形を、複数の基準形状と移動上限で覆う診断用計算。"""
import numpy as np

ANCHORS = (0., .5, 1.)
MAX_ANCHOR_DISTANCE = .25
AFFINE_TOLERANCE_M = 1e-7
SAMPLE_OPENNESS = (1., .75, .5, .25, 0.)


def anchor_padding(open_vertices, half_vertices, closed_vertices):
    arrays = [np.asarray(v, dtype=float) for v in (open_vertices, half_vertices, closed_vertices)]
    a, h, b = arrays
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or any(v.shape != a.shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('まばたき頂点の対応が不正です')
    if np.max(np.linalg.norm(h - (a + b) / 2, axis=1)) > AFFINE_TOLERANCE_M:
        raise ValueError('線形補間を検証できません')
    # 任意の開度は基準開度から最大1/4。三角形内部も頂点移動上限の凸結合で覆える。
    return np.linalg.norm(a - b, axis=1) * MAX_ANCHOR_DISTANCE + AFFINE_TOLERANCE_M


def validated_anchor_meshes(blink):
    if blink['values'] != list(SAMPLE_OPENNESS) or blink['isBinary'] or blink['overrideBlink'] != 'none' or not blink['bonesUnchanged']:
        raise ValueError('連続まばたき包絡の前提を満たしません')
    if not blink['binds'] or any(kind != 'VRMExpressionMorphTargetBind' for kind in blink['binds']):
        raise ValueError('線形頂点変形以外のまばたきは扱いません')
    reference = [e for e in blink['weights'][0] if e['name'] != 'blink']
    for value, weights in zip(SAMPLE_OPENNESS, blink['weights'], strict=True):
        if [e for e in weights if e['name'] != 'blink'] != reference:
            raise ValueError('他の表情が変化しています')
        current = [e for e in weights if e['name'] == 'blink']
        if len(current) != 1 or current[0]['weight'] != 1-value:
            raise ValueError('開度と表情重みが一致しません')
    frames=blink['frames']; result=[]
    if len(frames) != len(SAMPLE_OPENNESS) or not frames[0] or any(len(f) != len(frames[0]) for f in frames):
        raise ValueError('形状数が一致しません')
    for index, mesh in enumerate(frames[0]):
        if any(f[index]['name'] != mesh['name'] or f[index]['indices'] != mesh['indices'] for f in frames):
            raise ValueError('三角形の対応が変化しています')
        vertices=[np.asarray(f[index]['vertices'],dtype=float).reshape(-1,3) for f in frames]
        padding=anchor_padding(vertices[0],vertices[2],vertices[4])
        for value,v in zip(SAMPLE_OPENNESS,vertices,strict=True):
            if v.shape != vertices[0].shape or not np.isfinite(v).all() or np.max(np.linalg.norm(v-(value*vertices[0]+(1-value)*vertices[4]),axis=1)) > AFFINE_TOLERANCE_M:
                raise ValueError('線形の頂点移動を確認できません')
        for anchor in (0,2,4): result.append((vertices[anchor],mesh['indices'],'person:'+mesh['name'],padding))
    return result
