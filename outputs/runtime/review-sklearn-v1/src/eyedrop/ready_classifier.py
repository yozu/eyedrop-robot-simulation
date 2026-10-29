"""準備状態の分類確率を、動作側へ渡す二値判定へ変換する。"""
import numpy as np

CLASS_THRESHOLD=.5


class ReadyClassifier:
    def __init__(self,model):
        self.model=model
        self.n_features_in_=model.n_features_in_

    def predict(self,features):
        return (np.asarray(self.model.predict(features))>=CLASS_THRESHOLD).astype(float)
