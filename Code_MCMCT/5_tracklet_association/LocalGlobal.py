import numpy as np
from sklearn.cluster import SpectralClustering
import os

def read_tracklet_data(file_path, camera_id):
    tracklets = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = line.strip().split(',')
            frame_index = int(data[0])
            track_id = int(data[1])
            x_world = float(data[4])
            y_world = float(data[5])
            embedding = np.array([float(v) for v in data[9:]])

            unique_track_id = f"cam{camera_id}_track{track_id}"

            if unique_track_id not in tracklets:
                tracklets[unique_track_id] = {
                    'frames': [],
                    'coordinates': [],
                    'embeddings': []
                }

            tracklets[unique_track_id]['frames'].append(frame_index)
            tracklets[unique_track_id]['coordinates'].append((x_world, y_world))
            tracklets[unique_track_id]['embeddings'].append(embedding)

    for track_id, info in tracklets.items():
        tracklets[track_id]['avg_embedding'] = np.mean(info['embeddings'], axis=0)
        tracklets[track_id]['start_frame'] = min(info['frames'])
        tracklets[track_id]['end_frame'] = max(info['frames'])
        tracklets[track_id]['avg_position'] = np.mean(info['coordinates'], axis=0)

    return tracklets

def compute_similarity_matrix(tracklets1, tracklets2, position_threshold=700):
    tracklet_ids = list(tracklets1.keys()) + list(tracklets2.keys())
    num_tracklets = len(tracklet_ids)

    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_tracklets, num_tracklets))

    for i in range(num_tracklets):
        for j in range(i + 1, num_tracklets):
            id1 = tracklet_ids[i]
            id2 = tracklet_ids[j]


            t1 = tracklets1[id1] if id1 in tracklets1 else tracklets2[id1]
            t2 = tracklets1[id2] if id2 in tracklets1 else tracklets2[id2]

            # 时间约束
            if (id1 in tracklets1 and id2 in tracklets1) or (id1 in tracklets2 and id2 in tracklets2):
                if not (t1['end_frame'] < t2['start_frame'] or t2['end_frame'] < t1['start_frame']):
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 100000
                    continue

            # 空间约束
            if id1 in tracklets1 and id2 in tracklets2:
                position_distance = np.linalg.norm(t1['avg_position'] - t2['avg_position'])
                if position_distance > position_threshold:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 100000
                    continue

            # 外观特征
            appearance_distance = np.linalg.norm(t1['avg_embedding'] - t2['avg_embedding'])
            similarity_matrix[i, j] = similarity_matrix[j, i] = np.exp(-appearance_distance)

    return similarity_matrix, tracklet_ids

def tracklet_to_global_id(file1, file2, num_clusters=6, position_threshold=700):
    tracklets1 = read_tracklet_data(file1, camera_id=1)
    tracklets2 = read_tracklet_data(file2, camera_id=2)

    # 构建相似度矩阵
    similarity_matrix, tracklet_ids = compute_similarity_matrix(tracklets1, tracklets2, position_threshold)

    # 执行谱聚类
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = clustering.fit_predict(similarity_matrix)

    # 输出tracklet ID到global ID的映射
    tracklet_to_global = {}
    for i, tracklet_id in enumerate(tracklet_ids):
        tracklet_to_global[tracklet_id] = labels[i]

    return tracklet_to_global

file1 = 'sample_testResults/20240425_6_04_D2_emb_added_WorldADD.txt'
file2 = 'sample_testResults/20240425_6_04_D3_emb_added_WorldADD.txt'

# 聚合tracklet
tracklet_to_global = tracklet_to_global_id(file1, file2)

# 打印结果
for tracklet_id, global_id in tracklet_to_global.items():
    print(f"Tracklet {tracklet_id} -> Global ID {global_id}")
