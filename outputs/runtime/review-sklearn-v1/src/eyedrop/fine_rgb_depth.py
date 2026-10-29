"""高精細なRGB回帰へ奥行きだけを差し替える。"""
from .rgb_depth import image_vector as coarse_vector,DEPTH_AXIS

IMAGE_SIDE=64


def image_vector(content):return coarse_vector(content,side=IMAGE_SIDE)


class FineRGBDepthPosition:
    def __init__(self,baseline,regressor):
        self.baseline=baseline
        self.regressor=regressor

    def predict_image(self,features,content):
        point=self.baseline.predict_image(features,content).copy()
        point[DEPTH_AXIS]=self.regressor.predict(image_vector(content).reshape(1,-1))[0]
        return point
