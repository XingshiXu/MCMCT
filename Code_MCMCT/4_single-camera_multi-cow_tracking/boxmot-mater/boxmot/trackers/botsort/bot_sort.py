# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend    # 获取ReID功能
from boxmot.motion.cmc.sof import SOF    # 一种光流方法  用于运动补偿
from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator

print("bot_sort")

# class STrack00(BaseTrack):
#     shared_kalman = KalmanFilterXYWH()
#
#         # det是检测到的目标的坐标及相关信息，格式为[x1,y1,x2,y2, conf, cls, det_ind]
#         # feat 目标的特征向量
#         #  feat_history 保存历史特征的长度
#         # max_obs 保存历史观测的位置数量
#     def __init__(self, det, feat=None, feat_history=50, max_obs=50):
#         # wait activate
#         self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
#         self.conf = det[4]
#         self.cls = det[5]
#         self.det_ind = det[6]
#         self.max_obs=max_obs
#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False    # 是否已经激活该轨迹
#         self.cls_hist = []  # (cls id, freq)
#         self.update_cls(self.cls, self.conf)
#         self.history_observations = deque([], maxlen=self.max_obs)  # deque是一个”双段队列“。该代码 保持队列中仅存储最近的feat_history特征
#
#         self.tracklet_len = 0
#
#         self.smooth_feat = None    # 特征平滑
#         self.curr_feat = None
#         if feat is not None:
#             self.update_features(feat)
#         self.features = deque([], maxlen=feat_history)
#         self.alpha = 0.9
#
#     def update_features(self, feat):
#         feat /= np.linalg.norm(feat)
#         self.curr_feat = feat
#         if self.smooth_feat is None:
#             self.smooth_feat = feat
#         else: # 使用指数加权移动平均法（EWMA）平滑当前特征与历史特征
#             self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
#         self.features.append(feat)
#         self.smooth_feat /= np.linalg.norm(self.smooth_feat)
#
#     # 更新目标的类别cls和对应的置信度conf
#     def update_cls(self, cls, conf):
#         if len(self.cls_hist) > 0:
#             max_freq = 0
#             found = False
#             for c in self.cls_hist:
#                 if cls == c[0]:
#                     c[1] += conf
#                     found = True
#
#                 if c[1] > max_freq:
#                     max_freq = c[1]
#                     self.cls = c[0]
#             if not found:
#                 self.cls_hist.append([cls, conf])
#                 self.cls = cls
#         else:
#             self.cls_hist.append([cls, conf])
#             self.cls = cls
#
#     # 使用卡尔曼滤波器预测目标的下一步位置
#     def predict(self):
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[6] = 0
#             mean_state[7] = 0
#
#         self.mean, self.covariance = self.kalman_filter.predict(
#             mean_state, self.covariance
#         )
#
#     @staticmethod
#     def multi_predict(stracks):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#             for i, st in enumerate(stracks):
#                 if st.state != TrackState.Tracked:
#                     multi_mean[i][6] = 0
#                     multi_mean[i][7] = 0
#             multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
#                 multi_mean, multi_covariance
#             )
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov
#
#     @staticmethod
#     def multi_gmc(stracks, H=np.eye(2, 3)):
#         if len(stracks) > 0:
#             multi_mean = np.asarray([st.mean.copy() for st in stracks])
#             multi_covariance = np.asarray([st.covariance for st in stracks])
#
#             R = H[:2, :2]
#             R8x8 = np.kron(np.eye(4, dtype=float), R)
#             t = H[:2, 2]
#
#             for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#                 mean = R8x8.dot(mean)
#                 mean[:2] += t
#                 cov = R8x8.dot(cov).dot(R8x8.transpose())
#
#                 stracks[i].mean = mean
#                 stracks[i].covariance = cov
#
#     def activate(self, kalman_filter, frame_id):
#         """Start a new tracklet"""
#         self.kalman_filter = kalman_filter
#         self.id = self.next_id()
#
#         self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)
#
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         if frame_id == 1:
#             self.is_activated = True
#         self.frame_id = frame_id
#         self.start_frame = frame_id
#
#     # 重新激活一个轨迹 【可以选择是否分配新的ID(new_id-True时)】
#     def re_activate(self, new_track, frame_id, new_id=False):
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, new_track.xywh
#         )
#         if new_track.curr_feat is not None:
#             self.update_features(new_track.curr_feat)
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.frame_id = frame_id
#         if new_id:
#             self.id = self.next_id()
#         self.conf = new_track.conf
#         self.cls = new_track.cls
#         self.det_ind = new_track.det_ind
#
#         self.update_cls(new_track.cls, new_track.conf)
#
#     def update(self, new_track, frame_id):
#         """
#         Update a matched track
#         :type new_track: STrack
#         :type frame_id: int
#         :type update_feature: bool
#         :return:
#         """
#         self.frame_id = frame_id
#         self.tracklet_len += 1
#
#         self.history_observations.append(self.xyxy)
#
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, new_track.xywh
#         )
#
#         if new_track.curr_feat is not None:
#             self.update_features(new_track.curr_feat)
#
#         self.state = TrackState.Tracked
#         self.is_activated = True
#
#         self.conf = new_track.conf
#         self.cls = new_track.cls
#         self.det_ind = new_track.det_ind
#         self.update_cls(new_track.cls, new_track.conf)
#
#     @property
#     def xyxy(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         if self.mean is None:
#             ret = self.xywh.copy()  # (xc, yc, w, h)
#         else:
#             ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
#         ret = xywh2xyxy(ret)
#         return ret

"""HingsHsu修改（XuXingshi修改于2024.09）"""
"""改进主要设计以下几方面：
    1.方向性嵌入的存储与更新
    2.方向性判断
    3.嵌入匹配
    """
class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

        # det是检测到的目标的坐标及相关信息，格式为[x1,y1,x2,y2, conf, cls, det_ind]
        # feat 目标的特征向量
        #  feat_history 保存历史特征的长度
        # max_obs 保存历史观测的位置数量
    def __init__(self, det, feat=None, feat_history=50, max_obs=50):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs  ## 最多的观测XX数?
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False    # 是否已经激活该轨迹
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=self.max_obs)  # deque是一个”双段队列“。该代码 保持队列中仅存储最近的feat_history特征

        self.tracklet_len = 0

        self.feat_front = None # added by 许
        self.feat_back = None # added by 许
        self.feat_left = None # added by 许
        self.feat_right = None # added by 许

        self.smooth_feat = None    # 特征平滑
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9



    def update_features(self, feat, direction=None):
        """更新嵌入特征，根据方向更新不同的方向嵌入，同时更新最近特征
        param feat: 新的牛只表观特征
        param direction: 指定更新哪个方向的特征
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        # 更新最近的特征（原代码的smooth_feat逻辑）
        # self.smooth_feat = None########################
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else: # 使用指数加权移动平均法（EWMA）平滑当前特征与历史特征
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

        # 根据方向更新对应的嵌入：
        if feat is not None:
            if   direction == "front":
                self.feat_front = feat
            elif direction == "back":
                self.feat_back = feat
            elif direction == "left":
                self.feat_left = feat
            elif direction == "right":
                self.feat_right = feat

    """添加轨迹朝向判断程序"""
    #  基于连续的轨迹位移方向，在update或predict中进行朝向判断
    # def determine_direction_NO(self):
    #     """根据最近5帧的轨迹位移来确定奶牛运动的朝向
    #     返回值为 ‘front'、’back‘、'left'、'right'其中之一"""
    #     if len(self.history_observations) >= 5:# 计算最近5帧的x和y坐标的位移
    #         deltas = [self.history_observations[i+1][:2] - self.history_observations[i][:2] for i in range(-5, -1)]
    #         avg_delta = np.mean(deltas, axis=0)
    #
    #         # 判断主要方向
    #         if avg_delta[0] > 0: # 向右
    #             return "right"
    #         elif avg_delta[0] < 0: # 向右
    #             return "left"
    #         elif avg_delta[1] > 0: # local相机视角下的向下，即靠近摄像头（头朝向摄像头）
    #             return "front"
    #         elif avg_delta[1] < 0: # 向下
    #             return "back"
    #     return None
    def determine_direction(self):
        """根据最近5帧的轨迹位移来确定奶牛运动的朝向
        返回值为 'front'、'back'、'left'、'right' 其中之一"""
        if len(self.history_observations) >= 11:
            directions = []
            m = 10 # 连续m帧(超参数)
            for i in range(m):
            # for i in range(len(self.history_observations) - 1):
                delta = self.history_observations[i + 1][:2] - self.history_observations[i][:2]
                if self.conf > 0.85:  ####
                    if abs(delta[0]) > 1.2*abs(delta[1]):
                        if delta[0] > 0:
                            # directions.append("NA")
                            directions.append("right")
                        else:
                            # directions.append("NA")
                            directions.append("left")
                    else:
                        if delta[1] > 0:
                            # directions.append("NA")
                            directions.append("front")
                        else:
                            # directions.append("NA")
                            directions.append("back")
                else: directions.append("NA")###

            # Check if all directions are the same
            if len(set(directions)) == 1:
                return directions[0]

        return None

    # 更新目标的类别cls和对应的置信度conf
    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += conf
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, conf])
                self.cls = cls
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    # 使用卡尔曼滤波器预测目标的下一步位置
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    """在重识别过程中，在距离计算时比较四个方向的特征和最近较近。选择距离最小的那个"""
    def calc_embedding_distance(self, query_feat):
        """计算query_feat与各个方向嵌入和最近特征的距离
        返回最小距离"""
        distances = []

        if self.feat_front is not None:
            distances.append(np.linalg.norm(query_feat - self.feat_front))
        if self.feat_back is not None:
            distances.append(np.linalg.norm(query_feat - self.feat_back))
        if self.feat_left is not None:
            distances.append(np.linalg.norm(query_feat - self.feat_left))
        if self.feat_right is not None:
            distances.append(np.linalg.norm(query_feat - self.feat_right))
        if self.feat_recent is not None:
            distances.append(np.linalg.norm(query_feat - self.feat_recent))

        return min(distances) if distances else float('inf')

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    # 重新激活一个轨迹 【可以选择是否分配新的ID(new_id-True时)】
    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_id):
        """
        更新匹配的轨迹，并更新方向特征
        """
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)

        # 使用卡尔曼滤波器更新状态
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )

        # 根据轨迹方向更新对应的特征
        direction = self.determine_direction()
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, direction)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret

class BoTSORT(BaseTracker):

    def __init__(
        self,
        model_weights, # ReID的模型权重文件路径
        device,
        fp16,
        per_class=False,
        track_high_thresh: float = 0.5,  # 高置信度阈值
        track_low_thresh: float = 0.1,   # 低置信度阈值
        new_track_thresh: float = 0.6,   # 创建新轨迹的置信度阈值
        track_buffer: int = 30,          # 轨迹缓存帧数
        match_thresh: float = 0.8,      # 匹配步骤中使用的阈值
        proximity_thresh: float = 0.5,  # IOU距离 的阈值
        appearance_thresh: float = 0.25,# 外观嵌入匹配的阈值
        cmc_method: str = "sof",
        frame_rate=5,
        fuse_first_associate: bool = True,
        with_reid: bool = True, # 是否使用ReID模型进行外观特征匹配
    ):
        super().__init__()
        self.lost_stracks = []  # type: list[STrack] # 用于存储当前帧的活跃轨迹
        self.removed_stracks = []  # type: list[STrack]  # 用于存储当前帧需要删除的轨迹
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            ).model

        self.cmc = SOF()
        self.fuse_first_associate = fuse_first_associate

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:

        self.check_inputs(dets, img)

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        ### 将检测框分为两轮关联（根据置信度分为高、低检测框）
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        # Remove bad detections
        confs = dets[:, 4]
        # find second round association detections # 低置信度检测框
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        # find first round association detections # 高置信度检测框
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        """Extract embeddings 外观特征提取 """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_high = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_high = self.model.get_features(dets_first[:, 0:4], img)


        ### 将高置信度的检测框 dets_first 转换为 STrack 对象，用于后续的轨迹管理和关联
        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []



        """ Add newly detected tracklets to active_tracks"""
        unconfirmed = []                           # 未确认的轨迹
        active_tracks = []  # type: list[STrack]   # 已经被确认且在跟踪中的轨迹
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)   # active_tracks是激活的轨迹； lost_stracks是指前几帧中失去检测目标的轨迹

        # Predict the current location with KF                          # 卡尔曼滤波  预测当前轨迹下一帧的位置
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high conf detection boxes # 高置信度框 第一次关联
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
          ious_dists = fuse_score(ious_dists, detections)
        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0

            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
            print("ious_dists is {}".format(ious_dists))
            print("emb_dists is {}".format(emb_dists))
        else:
            dists = ious_dists
        # 使用匈牙利算法进行线性分配 确定轨迹和检测框之间的匹配关系
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        # 对匹配的轨迹进行更新
        #    如果轨迹正处于跟踪状态，直接更新位置
        #    如果是丢失的轨迹，重新激活该轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # 对未匹配的轨迹使用低置信度检测框进行第二次关联，采用IoU作为距离度量，并更新轨迹状态
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(dets_second, max_obs=self.max_obs) for dets_second in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        ious_dists = fuse_score(ious_dists, detections)
        
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.1)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 处理未确认轨迹和新轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # 更新丢失、移除轨迹： 如果丢失的帧数超过了max_time_lost,则将其标记为移除
        for track in self.lost_stracks:
            # if self.frame_count - track.end_frame > self.max_age:
            print(self.max_age)
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        # 返回当前帧的轨迹。输出当前帧的轨迹信息（包括轨迹的边界框坐标、ID、置信度、类别）
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        outputs_add_emb = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)

            output_emb = []  # xxs添加，由于保存姿态引导的嵌入以及最近的外观嵌入
            output_emb.extend(t.xyxy)
            output_emb.append(t.id)
            output_emb.append(t.conf)
            output_emb.append(t.cls)
            output_emb.append(t.det_ind)

            # 如果没有这一特征则 512维特征全部置零
            output_emb.extend(t.feat_front if t.feat_front is not None else np.zeros_like(t.smooth_feat))  # 增加方向引导的外观嵌入
            output_emb.extend(t.feat_back if t.feat_back is not None else np.zeros_like(t.smooth_feat))
            output_emb.extend(t.feat_right if t.feat_right is not None else np.zeros_like(t.smooth_feat))
            output_emb.extend(t.feat_left if t.feat_left is not None else np.zeros_like(t.smooth_feat))
            output_emb.extend(t.smooth_feat)

            outputs_add_emb.append(output_emb)

        outputs = np.asarray(outputs)
        outputs_add_emb = np.asarray(outputs_add_emb)
        return outputs, outputs_add_emb


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
