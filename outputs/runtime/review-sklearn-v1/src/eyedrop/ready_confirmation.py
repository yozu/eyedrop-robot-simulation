"""過去と現在の画像判定だけで、短い再開眼を押圧準備から除外する。"""
from .state_control import OPEN_THRESHOLD


class ReadyConfirmation:
    def __init__(self,frames):
        if frames<1:raise ValueError('確認フレーム数は1以上が必要です')
        self.frames=frames
        self.count=0

    def update(self,opening):
        self.count=self.count+1 if opening>=OPEN_THRESHOLD else 0
        if self.frames==1 or opening<OPEN_THRESHOLD:return opening
        return opening if self.count>=self.frames else 0.
