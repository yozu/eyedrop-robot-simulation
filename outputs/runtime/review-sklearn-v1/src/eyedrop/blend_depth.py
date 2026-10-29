"""固定した二つの画像回帰を、人物共通の比率で奥行きだけ合成する。"""


class BlendDepthPosition:
    def __init__(self,baseline,candidate,weight):
        if not 0<weight<1:raise ValueError('合成比率は0と1の間が必要です')
        self.baseline=baseline
        self.candidate=candidate
        self.weight=weight

    def predict_image(self,features,content):
        point=self.baseline.predict_image(features,content).copy()
        alternative=self.candidate.predict_image(features,content)
        point[2]+=self.weight*(alternative[2]-point[2])
        return point
