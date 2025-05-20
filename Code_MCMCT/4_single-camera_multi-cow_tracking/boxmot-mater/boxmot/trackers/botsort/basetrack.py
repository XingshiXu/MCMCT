# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
from collections import OrderedDict

import numpy as np


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        """Retruns the end frame of the track"""
        return self.frame_id

    @staticmethod
    def next_id():
        """Increment and return the global track ID counter
        返回并增加 全局track ID counter.
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """使用提供的参数 激活Track"""
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
