from dataclasses import dataclass
from .protocol import Reply


@dataclass(frozen=True)
class GateConfig:
    max_age: float = 0.3
    max_gap: float = 0.15
    stable_seconds: float = 0.1
    open_threshold: float = 0.9
    visible_threshold: float = 0.9


class Gate:
    """鮮度と識別子を検証し、一度だけ放出を許可する。"""

    def __init__(self, session, episode, model, config=None):
        self.session, self.episode, self.model = session, episode, model
        self.config = config or GateConfig()
        self.last = None
        self.open_since = None
        self.ready = False
        self.aborted = False
        self.released = False

    def disconnect(self):
        self.aborted = True
        self.ready = False
        self.open_since = None

    def observe(self, result: Reply, now: float, aligned: bool):
        if self.aborted or self.released:
            return False
        valid = (result.session == self.session and result.episode == self.episode
                 and result.model == self.model and result.mode != "stub"
                 and 0 <= now - result.captured <= self.config.max_age)
        if self.last is not None:
            valid = valid and result.frame > self.last.frame and result.captured > self.last.captured
        if not valid:
            self.disconnect()
            return False
        if self.last is not None and result.captured - self.last.captured > self.config.max_gap:
            self.open_since = None
        self.last = result
        good = (aligned and result.visible >= self.config.visible_threshold
                and result.eyelid[0] >= self.config.open_threshold)
        if not good:
            self.open_since = None
            self.ready = False
            return False
        if self.open_since is None:
            self.open_since = result.captured
        self.ready = result.captured - self.open_since >= self.config.stable_seconds
        return self.ready

    def release(self, now: float, aligned: bool = True):
        if self.last is None or not 0 <= now - self.last.captured <= self.config.max_age:
            self.disconnect()
        if self.aborted or self.released or not self.ready or not aligned:
            return False
        self.released = True
        return True
