import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math
from tqdm import tqdm
import random
import copy
from collections import defaultdict


# ============= 实用函数 =============

def check_nan(tensor, name=""):
    """检查张量是否包含NaN值"""
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含NaN值!")
        return True
    return False


def safe_normalize(tensor, dim=-1, eps=1e-8):
    """安全的向量归一化，防止零向量"""
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True).clamp(min=eps)
    return tensor / norm


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_label_csv():
    """加载并正确解析label.csv文件"""
    # 尝试不同的分隔符读取文件
    separators = ['\t', ';', '|']
    csv_path = 'D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/label_fixed.csv'

    # 首先读取原始文本，检查实际内容
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(5)]
        print("CSV文件前5行内容:")
        for line in first_lines:
            print(line.strip())
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 尝试不同的分隔符读取
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            # 如果成功解析并有多列，返回结果
            if len(df.columns) > 1:
                print(f"成功使用分隔符 '{sep}' 读取CSV，识别到{len(df.columns)}列")
                print(f"列名: {df.columns.tolist()}")
                return df
        except Exception:
            continue

    # 如果常规方法失败，尝试自定义解析方法
    try:
        # 读取原始文本
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 检查第一行是否包含标题
        header = lines[0].strip()
        if '\t' in header:
            sep = '\t'
        elif ';' in header:
            sep = ';'
        else:
            # 假设空格是分隔符，但需要处理标题中的空格
            parts = header.split()
            if len(parts) >= 7:  # 假设至少有7个属性
                # 创建自定义列名
                cols = ['user', 'path', 'echo_id', 'rating', 'mood_valence', 'mood_arousal', 'music_valence']

                # 处理数据行
                data = []
                for line in lines[1:]:  # 跳过标题行
                    values = line.strip().split()
                    if len(values) >= 7:
                        record = {cols[i]: values[i] for i in range(min(len(cols), len(values)))}
                        data.append(record)

                # 创建DataFrame
                df = pd.DataFrame(data)
                print(f"使用自定义解析方法成功读取CSV，识别到{len(df.columns)}列")
                print(f"列名: {df.columns.tolist()}")
                return df

        # 如果找到了分隔符，使用该分隔符解析
        if sep:
            cols = header.split(sep)
            data = []
            for line in lines[1:]:  # 跳过标题行
                values = line.strip().split(sep)
                if len(values) == len(cols):
                    record = {cols[i]: values[i] for i in range(len(cols))}
                    data.append(record)

            # 创建DataFrame
            df = pd.DataFrame(data)
            print(f"使用分隔符 '{sep}' 成功读取CSV，识别到{len(df.columns)}列")
            print(f"列名: {df.columns.tolist()}")
            return df

    except Exception as e:
        print(f"自定义解析CSV时出错: {e}")

    # 如果所有方法都失败，使用pandas的更灵活的解析器
    try:
        df = pd.read_csv(csv_path, engine='python')
        print(f"使用pandas灵活解析器读取CSV，识别到{len(df.columns)}列")
        print(f"列名: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"使用pandas灵活解析器时出错: {e}")

    # 如果一切都失败了，返回None
    print("无法正确解析CSV文件")
    return None


# ============= 增强的数据加载和预处理 =============

def load_eeg_data(load_metadata=True):
    """加载EEG数据文件并进行高级预处理"""
    print("加载EEG数据文件...")

    # 加载标签和特征数据
    label = np.load('D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/label_fixed.npy', allow_pickle=True)
    data = np.load('D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/data.npy')
    data_withpsd = np.load('D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/data_withpsd.npy')
    ratings = np.load('D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/rating.npy')

    print(f"数据加载完成: {data.shape[0]}个样本, {data.shape[1]}个基础特征, {data_withpsd.shape[1]}个PSD特征")

    # 检查标签类型并转换为可用的整数数组
    print(f"标签类型: {label.dtype}, 形状: {label.shape}")

    # 确保标签是一维数组
    if len(label.shape) > 1:
        print("标签不是一维数组，展平中...")
        label = label.flatten()

    # 转换标签为整数
    try:
        if np.issubdtype(label.dtype, np.floating):
            print("将标签从浮点数转换为整数...")
            label = label.astype(int)
        elif not np.issubdtype(label.dtype, np.integer):
            print(f"标签不是数值类型，创建数值映射...")
            unique_labels = np.unique(label)
            label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
            label = np.array([label_to_idx[lbl] for lbl in label])

        print(f"标签分布: {np.bincount(label)}")
    except Exception as e:
        print(f"标签处理错误: {e}")
        # 如果无法转换，创建默认标签
        print("创建默认标签...")
        label = np.zeros(len(data), dtype=int)

    try:
        # 手动解析CSV文件
        csv_path = 'D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/label_fixed.csv'

        # 读取文件内容
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().strip('"') for line in f.readlines()]

        print(f"CSV文件总行数: {len(lines)}")

        # 分割标题和数据
        header = lines[0].split('\t')
        print(f"正确解析的列名: {header}")

        # 创建数据列表
        rows = []
        for line in lines[1:]:
            values = line.split('\t')
            if len(values) == len(header):
                rows.append(values)

        # 创建DataFrame
        df = pd.DataFrame(rows, columns=header)
        print(f"成功创建DataFrame，包含{len(df.columns)}列和{len(df)}行")
        print(f"原始数据行数: {len(label)}")

        # 检查行数匹配情况
        if len(df) != len(label):
            print(f"警告: CSV行数({len(df)})与标签数量({len(label)})不匹配")

        # 提取用户ID和评分
        if 'user' in df.columns and 'rating' in df.columns:
            user_ids = df['user'].values
            ratings = df['rating'].values

            # 将评分转换为数值类型
            ratings = np.array([float(r) for r in ratings])

            # 创建基于评分的分类标签: 0=低(1-2), 1=中(3), 2=高(4-5)
            rating_labels = np.zeros_like(ratings, dtype=int)
            rating_labels[(ratings == 3)] = 1  # 中等评分
            rating_labels[(ratings >= 4)] = 2  # 高评分

            print(f"创建了基于评分的标签，分布: {np.bincount(rating_labels)}")

            # 扩展标签以匹配原始数据长度
            if len(rating_labels) != len(data):
                print(f"调整标签长度从{len(rating_labels)}到{len(data)}")
                extended_labels = np.zeros(len(data), dtype=rating_labels.dtype)
                # 复制已有标签
                extended_labels[:len(rating_labels)] = rating_labels
                # 对于额外的样本，使用最常见的标签值或最后一个标签
                if len(rating_labels) > 0:
                    most_common_label = np.argmax(np.bincount(rating_labels))
                    extended_labels[len(rating_labels):] = most_common_label

                label = extended_labels
            else:
                label = rating_labels

            # 创建用户映射
            unique_users = np.unique(user_ids)
            user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
            user_mapping = np.array([user_to_idx[uid] for uid in user_ids])
            print(f"已创建用户映射，共有{len(unique_users)}名真实用户")
        else:
            print(f"无法找到用户ID或评分列，可用列: {df.columns.tolist()}")
            user_mapping = np.arange(len(label))
    except Exception as e:
        print(f"处理CSV时出错: {e}")
        import traceback
        traceback.print_exc()
        user_mapping = np.arange(len(label))

    # 更好的特征处理：使用RobustScaler对异常值不敏感
    scaler = RobustScaler()
    data = scaler.fit_transform(data)
    data_withpsd = scaler.fit_transform(data_withpsd)

    # 限制极端值，但范围更大，保留更多信息
    data = np.clip(data, -10, 10)
    data_withpsd = np.clip(data_withpsd, -10, 10)

    # 特征归一化，使其平均值接近0，方差为1（增强稳定性）
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_withpsd = scaler.fit_transform(data_withpsd)

    print("数据预处理完成")

    return label, data, data_withpsd, ratings, user_mapping


def extract_eeg_bands(eeg_data, bands=5):
    """
    从EEG特征中提取频带特征

    参数:
    eeg_data: 只包含EEG特征的数据
    bands: 分析的频带数量
    """
    feature_dim = eeg_data.shape[1]
    features_per_band = feature_dim // bands

    band_features = []
    for i in range(bands):
        start_idx = i * features_per_band
        end_idx = start_idx + features_per_band
        if i == bands - 1:  # 最后一个频带可能包含剩余的所有特征
            end_idx = feature_dim

        band_data = eeg_data[:, start_idx:end_idx]

        # 提取每个频带的统计特征
        band_mean = np.mean(band_data, axis=1, keepdims=True)
        band_std = np.std(band_data, axis=1, keepdims=True)
        band_max = np.max(band_data, axis=1, keepdims=True)
        band_min = np.min(band_data, axis=1, keepdims=True)
        # 添加更多高级统计特征
        band_median = np.median(band_data, axis=1, keepdims=True)
        band_skewness = np.zeros_like(band_mean)  # 偏度
        band_kurtosis = np.zeros_like(band_mean)  # 峰度

        # 计算偏度和峰度（避免除以零）
        for j in range(len(band_data)):
            if band_std[j] > 0:
                # 计算偏度
                diff = band_data[j] - band_mean[j]
                skew = np.mean((diff ** 3) / (band_std[j] ** 3 + 1e-10))
                band_skewness[j] = skew

                # 计算峰度
                kurt = np.mean((diff ** 4) / (band_std[j] ** 4 + 1e-10)) - 3  # 减3使得正态分布的峰度为0
                band_kurtosis[j] = kurt

        # 合并统计特征
        band_stats = np.concatenate([
            band_mean, band_std, band_max, band_min,
            band_median, band_skewness, band_kurtosis
        ], axis=1)
        band_features.append(band_stats)

    # 合并所有频带的特征
    enhanced_features = np.concatenate(band_features, axis=1)

    # 再次标准化
    scaler = StandardScaler()
    enhanced_features = scaler.fit_transform(enhanced_features)

    return enhanced_features


def create_recommendation_dataset(label, data_withpsd, ratings=None,
                                  create_interactions=False,
                                  user_features_range=(0, 2),
                                  music_features_range=(2, 27),
                                  eeg_features_range=(27, 52),
                                  num_music_items=300,
                                  user_mapping=None,
                                  augment_factor=0):  # 添加数据增强系数
    """
    创建推荐系统数据集，基于正确的特征范围

    参数:
        label: 用户类别标签
        data_withpsd: 包含用户特征、音乐特征和EEG特征的组合数据
        ratings: 真实用户-音乐评分数据
        create_interactions: 是否创建合成用户-音乐交互记录
        user_features_range: 用户特征的索引范围，如(0, 2)表示索引0-1
        music_features_range: 音乐特征的索引范围，如(2, 27)表示索引2-26
        eeg_features_range: EEG特征的索引范围，如(27, 52)表示索引27-51
        num_music_items: 当生成合成特征时，生成的音乐项目数量
        user_mapping: 记录到真实用户ID的映射数组
        augment_factor: 数据增强因子，表示要生成多少倍于原始数据的合成数据
    """
    num_records = len(data_withpsd)

    # 确保user_mapping长度与数据长度匹配
    if user_mapping is not None:
        if len(user_mapping) < num_records:
            print(f"警告: user_mapping长度({len(user_mapping)})小于记录数({num_records})，将扩展映射")
            # 扩展user_mapping到正确长度
            extended_mapping = np.zeros(num_records, dtype=user_mapping.dtype)
            extended_mapping[:len(user_mapping)] = user_mapping
            # 对于缺失的映射使用最后一个有效的用户ID
            if len(user_mapping) > 0:
                extended_mapping[len(user_mapping):] = user_mapping[-1]
            user_mapping = extended_mapping
    else:
        # 如果没有用户映射，则每条记录视为一个独立用户
        user_mapping = np.arange(num_records)

    # 确定有多少真实用户
    unique_user_ids = np.unique(user_mapping)
    num_users = len(unique_user_ids)
    print(f"检测到{num_users}名真实用户，{num_records}条记录")

    # 从data_withpsd提取不同类型的特征
    print(f"从data_withpsd中提取特征:")
    print(f"  - 用户特征: 索引{user_features_range[0]}-{user_features_range[1] - 1}")
    print(f"  - 音乐特征: 索引{music_features_range[0]}-{music_features_range[1] - 1}")
    print(f"  - EEG特征: 索引{eeg_features_range[0]}-{eeg_features_range[1] - 1}")

    user_features = data_withpsd[:, user_features_range[0]:user_features_range[1]]
    music_features_all = data_withpsd[:, music_features_range[0]:music_features_range[1]]
    eeg_features = data_withpsd[:, eeg_features_range[0]:eeg_features_range[1]]

    # 创建用户EEG数据字典，考虑用户映射
    eeg_data = defaultdict(list)

    # 按真实用户ID组织EEG特征
    for i in range(num_records):
        user_id = user_mapping[i]
        eeg_data[user_id].append(eeg_features[i])

    # 为每个用户创建聚合的EEG特征
    aggregated_eeg_data = {}
    for user_id, features_list in eeg_data.items():
        # 可以尝试不同的聚合方法：平均、最大值、最小值等
        aggregated_eeg_data[user_id] = np.mean(features_list, axis=0)

    # 增强的EEG特征
    enhanced_features = extract_eeg_bands(eeg_features, bands=5)
    enhanced_eeg_data = defaultdict(list)

    # 按真实用户ID组织增强EEG特征
    for i in range(num_records):
        user_id = user_mapping[i]
        enhanced_eeg_data[user_id].append(enhanced_features[i])

    # 聚合增强的EEG特征
    aggregated_enhanced_eeg = {}
    for user_id, features_list in enhanced_eeg_data.items():
        aggregated_enhanced_eeg[user_id] = np.mean(features_list, axis=0)

    # 创建音乐特征
    music_features = {}
    for i in range(num_records):
        music_features[i] = music_features_all[i]

    # 为音乐分配类别标签
    music_class_labels = {}
    for item_id in range(num_records):
        # 安全访问标签，防止索引越界
        if item_id < len(label):
            music_class_labels[item_id] = label[item_id]
        else:
            # 对于超出标签范围的项目，使用默认值或最后一个有效标签
            music_class_labels[item_id] = label[-1] if len(label) > 0 else 0

    # 处理用户-音乐交互
    interactions = []

    # 使用真实评分数据创建交互（如果存在）
    if ratings is not None and not create_interactions:
        print("使用真实评分数据创建交互记录...")

        # 使用真实评分创建交互，考虑用户映射
        for i, rating in enumerate(ratings):
            if i < num_records and rating > 0:  # 只包含有效用户和评分>0的记录
                user_id = user_mapping[i]  # 使用映射的用户ID
                item_id = i  # 假设用户i对应的音乐也是i
                interactions.append((user_id, item_id, rating))

        if interactions:
            print(f"从真实评分创建了{len(interactions)}个用户-音乐交互记录")
        else:
            print("没有从真实评分创建交互记录")

    # 如果需要创建合成交互
    elif create_interactions:
        print("创建合成用户-项目交互...")

        # 对每个用户，创建与其类别匹配和不匹配的交互
        for user_id in range(num_users):
            # 获取该用户的所有记录索引
            user_records = np.where(user_mapping == user_id)[0]
            if len(user_records) == 0:
                continue

            # 使用第一条记录的类别作为用户类别
            sample_record = user_records[0]
            user_class = label[sample_record]

            # 每个用户的交互数量 - 使用更多样化的设置
            num_interactions = np.random.randint(15, 20)

            # 为每个类别分配比例
            class_weights = {}
            for cls in np.unique(label):
                if cls == user_class:
                    # 用户自己的类别获得更高权重
                    class_weights[cls] = 0.5  # 50%是同类
                else:
                    # 其他类平分剩余的50%
                    class_weights[cls] = 0.5 / (len(np.unique(label)) - 1)

            # 按类别选择项目
            selected_items = []

            for cls, weight in class_weights.items():
                # 获取该类别的所有项目
                class_items = [item_id for item_id, item_class in music_class_labels.items()
                               if item_class == cls]

                if class_items:
                    # 要选择的项目数
                    num_to_select = max(1, int(num_interactions * weight))
                    # 随机选择不重复的项目
                    num_to_select = min(num_to_select, len(class_items))
                    chosen_items = np.random.choice(class_items, num_to_select, replace=False)
                    selected_items.extend(chosen_items)

            # 如果需要，添加随机项目
            remaining = num_interactions - len(selected_items)
            if remaining > 0:
                remaining_items = [i for i in music_features.keys() if i not in selected_items]
                if remaining_items:
                    additional_items = np.random.choice(remaining_items,
                                                        min(remaining, len(remaining_items)),
                                                        replace=False)
                    selected_items.extend(additional_items)

            # 生成更多样化的评分
            for item_id in selected_items:
                item_class = music_class_labels[item_id]

                if item_class == user_class:
                    # 同类别 - 使用更多样化的评分分布
                    if np.random.random() < 0.7:  # 70%概率高评分
                        rating = np.random.uniform(3.5, 5.0)
                    else:  # 30%概率低评分
                        rating = np.random.uniform(1.0, 3.5)
                else:
                    # 不同类别 - 使用更多样化的评分分布
                    if np.random.random() < 0.3:  # 30%概率高评分
                        rating = np.random.uniform(3.5, 5.0)
                    else:  # 70%概率低评分
                        rating = np.random.uniform(1.0, 3.5)

                # 四舍五入到最近的0.5
                rating = round(rating * 2) / 2

                # 确保评分在[1,5]范围内
                rating = max(1, min(5, rating))

                interactions.append((user_id, item_id, rating))

        print(f"创建了{len(interactions)}个原始用户-音乐交互记录")

        # 数据增强：为每个用户创建更多的合成交互
        if augment_factor > 1:
            print(f"开始进行数据增强，增强因子: {augment_factor}")
            synthetic_interactions = []

            # 对每个用户，创建额外的合成交互
            user_interactions = defaultdict(list)

            # 按用户分组原始交互
            for user_id, item_id, rating in interactions:
                user_interactions[user_id].append((item_id, rating))

            # 对每个用户进行增强
            for user_id, user_items in user_interactions.items():
                # 获取该用户的特征
                user_records = np.where(user_mapping == user_id)[0]
                if len(user_records) == 0:
                    continue

                # 用户的类别
                user_class = label[user_records[0]]

                # 生成合成交互
                for _ in range(augment_factor - 1):  # 已有1倍原始数据
                    # 随机选择N个真实交互记录进行混合
                    num_samples = min(len(user_items), np.random.randint(3, max(4, len(user_items) // 2)))
                    samples = random.sample(user_items, num_samples)

                    # 对评分加入噪声
                    for item_id, rating in samples:
                        # 添加少量噪声到评分
                        noise = np.random.normal(0, 0.2)  # 标准差0.2的高斯噪声
                        new_rating = rating + noise
                        new_rating = round(new_rating * 2) / 2  # 四舍五入到最近的0.5
                        new_rating = max(1, min(5, new_rating))  # 限制在[1,5]范围

                        # 添加合成交互
                        synthetic_interactions.append((user_id, item_id, new_rating))

                    # 增加一些新物品的交互
                    new_item_count = np.random.randint(1, 5)
                    existing_items = [item for item, _ in samples]
                    new_items = [item for item in range(num_records) if item not in existing_items]

                    if new_items:
                        selected_new_items = np.random.choice(new_items,
                                                              min(len(new_items), new_item_count),
                                                              replace=False)

                        for item_id in selected_new_items:
                            item_class = music_class_labels.get(item_id, user_class)  # 默认为用户类别

                            if item_class == user_class:
                                # 同类别
                                if np.random.random() < 0.8:  # 80%概率高评分
                                    rating = np.random.uniform(4.0, 5.0)
                                else:
                                    rating = np.random.uniform(1.0, 4.0)
                            else:
                                # 不同类别
                                if np.random.random() < 0.2:  # 20%概率高评分
                                    rating = np.random.uniform(4.0, 5.0)
                                else:
                                    rating = np.random.uniform(1.0, 4.0)

                            # 四舍五入并限制范围
                            rating = round(rating * 2) / 2
                            rating = max(1, min(5, rating))

                            # 添加合成交互
                            synthetic_interactions.append((user_id, item_id, rating))

            # 合并原始交互和合成交互
            interactions.extend(synthetic_interactions)
            print(f"数据增强后，共有{len(interactions)}个用户-音乐交互记录")

    else:
        print("未创建用户-音乐交互记录")

    # 如果有交互记录，分析数据分布
    if interactions:
        ratings_list = [r for _, _, r in interactions]
        if ratings_list:
            rating_bins = np.bincount([int(r) for r in ratings_list], minlength=6)[1:]
            print(f"评分分布: {rating_bins}")

            # 分析类别匹配
            matching_classes = sum(1 for u, i, r in interactions
                                   if label[np.where(user_mapping == u)[0][0] if len(
                np.where(user_mapping == u)[0]) > 0 else 0] == music_class_labels[i] and r >= 4)
            total_high_ratings = sum(1 for _, _, r in interactions if r >= 4)
            match_ratio = matching_classes / total_high_ratings if total_high_ratings > 0 else 0
            print(f"高评分中类别匹配比例: {match_ratio:.2f}")

    return eeg_data, aggregated_enhanced_eeg, music_features, interactions, music_class_labels, user_mapping


# ============= 改进后的元学习模块 =============

class MAMLParams:
    """Model-Agnostic Meta-Learning参数类"""

    def __init__(self, inner_lr=0.01, n_inner_steps=1, adaptation_steps=1, first_order=True):
        self.inner_lr = inner_lr  # 内层学习率
        self.n_inner_steps = n_inner_steps  # 内层更新步数
        self.adaptation_steps = adaptation_steps  # 适应步数
        self.first_order = first_order  # 是否使用一阶近似


class MAMLLayer(nn.Module):
    """MAML兼容层，支持快速参数复制和更新"""

    def __init__(self):
        super(MAMLLayer, self).__init__()

    def clone_parameters(self):
        """复制当前层参数"""
        cloned = {}
        for name, param in self.named_parameters():
            cloned[name] = param.clone()
        return cloned

    def update_parameters(self, cloned_params, grads, lr):
        """使用梯度更新克隆的参数"""
        updated = {}
        for name, param in cloned_params.items():
            if name in grads:
                updated[name] = param - lr * grads[name]
            else:
                updated[name] = param
        return updated

    def forward_with_params(self, x, params=None):
        """用自定义参数计算前向传播"""
        raise NotImplementedError("子类必须实现forward_with_params方法")


class MAMLLinear(MAMLLayer):
    """支持MAML的线性层"""

    def __init__(self, in_features, out_features, bias=True):
        super(MAMLLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def forward_with_params(self, x, params=None):
        """使用给定参数进行前向传播"""
        if params is None:
            return self.forward(x)

        weight = params.get('weight', self.weight)
        bias = params.get('bias', self.bias)
        return F.linear(x, weight, bias)


class MAMLSequential(MAMLLayer):
    """支持MAML的Sequential层"""

    def __init__(self, *layers):
        super(MAMLSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_with_params(self, x, params=None):
        """使用给定参数进行前向传播"""
        if params is None:
            return self.forward(x)

        curr_x = x
        for i, layer in enumerate(self.layers):
            layer_params = {k.replace(f'{i}.', ''): v for k, v in params.items()
                            if k.startswith(f'{i}.')}
            if hasattr(layer, 'forward_with_params'):
                curr_x = layer.forward_with_params(curr_x, layer_params)
            else:
                curr_x = layer(curr_x)
        return curr_x


# ============= 增强的数据集和加载器 =============

class EnhancedEEGDataset(Dataset):
    """增强的EEG推荐系统数据集，包含原始和增强特征，支持用户关联和少样本学习"""

    def __init__(self, eeg_data, enhanced_eeg_data, content_features, interactions,
                 user_mapping=None, augment=True, support_query_split=False, k_shot=5):
        self.eeg_data = eeg_data  # 原始EEG特征，按用户组织
        self.enhanced_eeg_data = enhanced_eeg_data  # 增强的EEG特征，按用户组织
        self.content_features = content_features  # 内容特征
        self.interactions = interactions  # (user_id, content_id, rating)
        self.user_mapping = user_mapping  # 记录索引到用户ID的映射
        self.augment = augment  # 是否应用数据增强

        # 少样本学习相关
        self.support_query_split = support_query_split  # 是否划分支持集和查询集
        self.k_shot = k_shot  # 每个用户的支持集样本数
        self.user_interactions = self._group_by_user()  # 按用户分组的交互

        # 验证数据集
        self._validate_dataset()

    def _group_by_user(self):
        """将交互按用户分组"""
        user_interactions = defaultdict(list)
        for idx, (user_id, content_id, rating) in enumerate(self.interactions):
            user_interactions[user_id].append((idx, content_id, rating))
        return user_interactions

    def _validate_dataset(self):
        """验证数据集的完整性，确保所有引用的键都存在"""
        missing_users = []
        missing_contents = []
        valid_interactions = []

        for user_id, content_id, rating in self.interactions:
            if user_id not in self.eeg_data or user_id not in self.enhanced_eeg_data:
                missing_users.append(user_id)
                continue

            if content_id not in self.content_features:
                missing_contents.append(content_id)
                continue

            valid_interactions.append((user_id, content_id, rating))

        if missing_users:
            print(f"警告: {len(missing_users)}个用户ID不在EEG数据中，这些交互将被忽略")

        if missing_contents:
            print(f"警告: {len(missing_contents)}个内容ID不在音乐特征中，这些交互将被忽略")

        if len(valid_interactions) < len(self.interactions):
            print(f"过滤后的交互数量: {len(valid_interactions)} (原始: {len(self.interactions)})")
            self.interactions = valid_interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        if not self.support_query_split:
            return self._get_regular_item(idx)
        else:
            return self._get_meta_learning_item(idx)

    def _get_regular_item(self, idx):
        """获取常规数据项"""
        user_id, content_id, rating = self.interactions[idx]

        # 获取用户的EEG数据和增强特征
        if isinstance(self.eeg_data[user_id], list):
            # 如果有多条记录，随机选择一条
            record_idx = np.random.randint(0, len(self.eeg_data[user_id]))
            eeg_features = self.eeg_data[user_id][record_idx]
        else:
            # 使用聚合特征
            eeg_features = self.eeg_data[user_id]

        if isinstance(self.enhanced_eeg_data[user_id], list):
            # 如果有多条记录，随机选择一条
            record_idx = np.random.randint(0, len(self.enhanced_eeg_data[user_id]))
            enhanced_eeg = self.enhanced_eeg_data[user_id][record_idx]
        else:
            # 使用聚合特征
            enhanced_eeg = self.enhanced_eeg_data[user_id]

        # 获取内容特征
        content_feat = self.content_features[content_id]

        # 数据增强
        if self.augment and np.random.random() < 0.7:  # 增加到70%的概率应用增强
            # 为EEG特征添加小的高斯噪声
            eeg_features = eeg_features + np.random.normal(0, 0.05, eeg_features.shape)
            # 为内容特征添加小的高斯噪声
            content_feat = content_feat + np.random.normal(0, 0.07, content_feat.shape)
            # 为增强的EEG特征添加噪声
            enhanced_eeg = enhanced_eeg + np.random.normal(0, 0.07, enhanced_eeg.shape)

            # 添加特征掩码（随机将一小部分特征置为0）
            if np.random.random() < 0.3:
                # 随机选择约10%的特征进行掩码
                mask_eeg = np.random.random(eeg_features.shape) > 0.1
                mask_enhanced = np.random.random(enhanced_eeg.shape) > 0.1
                mask_content = np.random.random(content_feat.shape) > 0.1

                eeg_features = eeg_features * mask_eeg
                enhanced_eeg = enhanced_eeg * mask_enhanced
                content_feat = content_feat * mask_content

        # 将评分归一化到[0,1]
        normalized_rating = rating / 5.0

        return {
            'user_id': user_id,
            'content_id': content_id,
            'eeg_features': torch.FloatTensor(eeg_features),
            'enhanced_eeg': torch.FloatTensor(enhanced_eeg),
            'content_features': torch.FloatTensor(content_feat),
            'rating': torch.FloatTensor([normalized_rating])
        }

    def _get_meta_learning_item(self, idx):
        """获取用于元学习的数据项（支持集和查询集）"""
        # 对于元学习，idx是用户ID的索引
        user_ids = list(self.user_interactions.keys())
        if idx >= len(user_ids):  # 防止索引越界
            idx = idx % len(user_ids)

        user_id = user_ids[idx]

        # 获取该用户的交互记录
        user_interactions = self.user_interactions[user_id]

        # 如果交互记录不足，使用所有记录
        if len(user_interactions) <= self.k_shot:
            support_indices = list(range(len(user_interactions)))
            query_indices = support_indices  # 如果数据不足，查询集与支持集相同
        else:
            # 随机抽样k个样本作为支持集
            support_indices = np.random.choice(
                range(len(user_interactions)),
                self.k_shot,
                replace=False
            ).tolist()

            # 其余样本作为查询集（查询集最多取10个样本)
            remaining_indices = [i for i in range(len(user_interactions)) if i not in support_indices]
            query_indices = np.random.choice(
                remaining_indices,
                min(10, len(remaining_indices)),
                replace=False
            ).tolist() if remaining_indices else []

        # 构建支持集
        support_set = []
        for idx in support_indices:
            _, content_id, rating = user_interactions[idx]
            item = self._prepare_interaction_data(user_id, content_id, rating)
            support_set.append(item)

        # 构建查询集
        query_set = []
        for idx in query_indices:
            _, content_id, rating = user_interactions[idx]
            item = self._prepare_interaction_data(user_id, content_id, rating)
            query_set.append(item)

        return {
            'user_id': user_id,
            'support': support_set,
            'query': query_set
        }

    def _prepare_interaction_data(self, user_id, content_id, rating):
        """准备单个交互的数据"""
        # 获取用户的EEG数据和增强特征
        if isinstance(self.eeg_data[user_id], list):
            record_idx = np.random.randint(0, len(self.eeg_data[user_id]))
            eeg_features = self.eeg_data[user_id][record_idx]
        else:
            eeg_features = self.eeg_data[user_id]

        if isinstance(self.enhanced_eeg_data[user_id], list):
            record_idx = np.random.randint(0, len(self.enhanced_eeg_data[user_id]))
            enhanced_eeg = self.enhanced_eeg_data[user_id][record_idx]
        else:
            enhanced_eeg = self.enhanced_eeg_data[user_id]

        # 获取内容特征
        content_feat = self.content_features[content_id]

        # 应用数据增强
        if self.augment and np.random.random() < 0.5:
            eeg_features = eeg_features + np.random.normal(0, 0.05, eeg_features.shape)
            content_feat = content_feat + np.random.normal(0, 0.07, content_feat.shape)
            enhanced_eeg = enhanced_eeg + np.random.normal(0, 0.07, enhanced_eeg.shape)

        # 将评分归一化到[0,1]
        normalized_rating = rating / 5.0

        return {
            'user_id': user_id,
            'content_id': content_id,
            'eeg_features': torch.FloatTensor(eeg_features),
            'enhanced_eeg': torch.FloatTensor(enhanced_eeg),
            'content_features': torch.FloatTensor(content_feat),
            'rating': torch.FloatTensor([normalized_rating])
        }

    def get_meta_batch(self, batch_size, k_shot=5, test=False):
        """获取用于元学习的批次"""
        user_ids = list(self.user_interactions.keys())

        # 过滤掉交互记录不足的用户
        valid_users = [uid for uid in user_ids if len(self.user_interactions[uid]) > k_shot]

        if len(valid_users) == 0:
            print("警告: 没有足够的交互记录来构建元学习批次")
            return []

        if len(valid_users) < batch_size:
            # 如果有效用户数量不足，可以重复采样
            selected_users = np.random.choice(valid_users, batch_size, replace=True)
        else:
            selected_users = np.random.choice(valid_users, batch_size, replace=False)

        batch = []
        for user_id in selected_users:
            user_interactions = self.user_interactions[user_id]

            # 确保用户至少有k_shot + 1条交互记录
            if len(user_interactions) <= k_shot:
                continue

            # 划分支持集和查询集
            indices = list(range(len(user_interactions)))
            support_indices = np.random.choice(indices, k_shot, replace=False)
            query_indices = [i for i in indices if i not in support_indices]

            # 对于测试，使用所有的查询样本；对于训练，只使用一部分
            if not test:
                if len(query_indices) > 10:  # 限制查询样本数量
                    query_indices = np.random.choice(query_indices, 10, replace=False)
            else:
                if len(query_indices) > 20:  # 测试时可以使用更多样本
                    query_indices = np.random.choice(query_indices, 20, replace=False)

            # 构建支持集
            support_set = []
            for idx in support_indices:
                _, content_id, rating = user_interactions[idx]
                item = self._prepare_interaction_data(user_id, content_id, rating)
                support_set.append(item)

            # 构建查询集
            query_set = []
            for idx in query_indices:
                _, content_id, rating = user_interactions[idx]
                item = self._prepare_interaction_data(user_id, content_id, rating)
                query_set.append(item)

            task = {
                'user_id': user_id,
                'support': support_set,
                'query': query_set
            }
            batch.append(task)

        return batch


# ============= 优化的EEG特征提取器 =============

class MultiHeadAttention(nn.Module):
    """简化但有效的多头注意力机制"""

    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "隐藏维度必须能被头数整除"

        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # 投影查询、键、值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑以便多头处理
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力权重计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # 最终线性投影
        output = self.out_proj(attn_output)

        return output

    def forward_with_params(self, x, params=None):
        """使用给定参数进行前向传播"""
        if params is None:
            return self.forward(x)

        batch_size, seq_len, hidden_dim = x.shape

        # 使用自定义参数进行投影
        q_weight = params.get('q_proj.weight', self.q_proj.weight)
        q_bias = params.get('q_proj.bias', self.q_proj.bias)
        k_weight = params.get('k_proj.weight', self.k_proj.weight)
        k_bias = params.get('k_proj.bias', self.k_proj.bias)
        v_weight = params.get('v_proj.weight', self.v_proj.weight)
        v_bias = params.get('v_proj.bias', self.v_proj.bias)
        out_weight = params.get('out_proj.weight', self.out_proj.weight)
        out_bias = params.get('out_proj.bias', self.out_proj.bias)

        # 投影查询、键、值
        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight, k_bias)
        v = F.linear(x, v_weight, v_bias)

        # 重塑以便多头处理
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力权重计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # 最终线性投影
        output = F.linear(attn_output, out_weight, out_bias)

        return output


class EEGFeatureExtractor(MAMLLayer):
    """为EEG数据优化的特征提取器，使用多头注意力和残差连接，支持MAML"""

    def __init__(self, input_dim, enhanced_dim, hidden_dim=128, output_dim=64, dropout=0.5):
        super(EEGFeatureExtractor, self).__init__()

        # 原始EEG特征处理路径
        self.eeg_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 增强EEG特征处理路径
        self.enhanced_path = nn.Sequential(
            nn.Linear(enhanced_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批归一化
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # 增加一层
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 最终映射到输出维度
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重
        self._init_weights()

        # 保存维度信息（用于元学习）
        self.input_dim = input_dim
        self.enhanced_dim = enhanced_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def _init_weights(self):
        """使用较大的初始化范围以避免梯度消失"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, eeg, enhanced_eeg):
        # 处理原始和增强特征
        eeg_features = self.eeg_path(eeg)

        # 检查批次维度是否存在
        if enhanced_eeg.dim() == 1:
            enhanced_eeg = enhanced_eeg.unsqueeze(0)  # 添加批次维度

        enhanced_features = self.enhanced_path(enhanced_eeg)

        # 融合特征
        combined = torch.cat([eeg_features, enhanced_features], dim=1)
        fused = self.fusion(combined)

        # 添加序列维度 (将特征视为长度为1的序列)
        x = fused.unsqueeze(1)

        # 自注意力处理 (残差连接)
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # 前馈网络 (残差连接)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        # 移除序列维度并映射到输出维度
        x = x.squeeze(1)
        output = self.output_proj(x)

        return output

    def forward_with_params(self, eeg, enhanced_eeg, params=None):
        """使用给定参数的前向传播"""
        if params is None:
            return self.forward(eeg, enhanced_eeg)

        # 这里需要对每个层使用自定义参数进行前向传播
        # 由于层太多，这里简化处理，只实现最关键的部分

        # 以下是一个简化示例，实际实现可能需要更详细的处理

        # 处理原始EEG特征
        eeg_features = eeg
        for i, layer in enumerate(self.eeg_path):
            layer_name = f'eeg_path.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                eeg_features = F.linear(eeg_features, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                eeg_features = F.layer_norm(eeg_features, layer.normalized_shape, weight, bias)
            elif isinstance(layer, nn.GELU):
                eeg_features = F.gelu(eeg_features)
            elif isinstance(layer, nn.Dropout) and self.training:
                eeg_features = F.dropout(eeg_features, layer.p)

        # 处理增强EEG特征
        # 检查批次维度是否存在
        if enhanced_eeg.dim() == 1:
            enhanced_eeg = enhanced_eeg.unsqueeze(0)  # 添加批次维度

        enhanced_features = enhanced_eeg
        for i, layer in enumerate(self.enhanced_path):
            layer_name = f'enhanced_path.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                enhanced_features = F.linear(enhanced_features, weight, bias)
            elif isinstance(layer, nn.BatchNorm1d):
                # BatchNorm较复杂，简化处理
                enhanced_features = layer(enhanced_features)
            elif isinstance(layer, nn.GELU):
                enhanced_features = F.gelu(enhanced_features)

        # 融合特征
        combined = torch.cat([eeg_features, enhanced_features], dim=1)

        # 融合层
        fused = combined
        for i, layer in enumerate(self.fusion):
            layer_name = f'fusion.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                fused = F.linear(fused, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                fused = F.layer_norm(fused, layer.normalized_shape, weight, bias)
            elif isinstance(layer, nn.GELU):
                fused = F.gelu(fused)
            elif isinstance(layer, nn.Dropout) and self.training:
                fused = F.dropout(fused, layer.p)

        # 添加序列维度
        x = fused.unsqueeze(1)

        # 自注意力 (残差连接)
        attention_params = {k.replace('attention.', ''): v for k, v in params.items()
                            if k.startswith('attention.')}
        attn_output = self.attention.forward_with_params(x, attention_params)

        # 第一个层归一化
        norm1_weight = params.get('norm1.weight', self.norm1.weight)
        norm1_bias = params.get('norm1.bias', self.norm1.bias)
        x = F.layer_norm(x + attn_output, self.norm1.normalized_shape, norm1_weight, norm1_bias)

        # 前馈网络
        ff_output = x
        for i, layer in enumerate(self.feed_forward):
            layer_name = f'feed_forward.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                ff_output = F.linear(ff_output, weight, bias)
            elif isinstance(layer, nn.GELU):
                ff_output = F.gelu(ff_output)
            elif isinstance(layer, nn.Dropout) and self.training:
                ff_output = F.dropout(ff_output, layer.p)

        # 第二个层归一化
        norm2_weight = params.get('norm2.weight', self.norm2.weight)
        norm2_bias = params.get('norm2.bias', self.norm2.bias)
        x = F.layer_norm(x + ff_output, self.norm2.normalized_shape, norm2_weight, norm2_bias)

        # 移除序列维度
        x = x.squeeze(1)

        # 输出投影
        output = x
        for i, layer in enumerate(self.output_proj):
            layer_name = f'output_proj.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                output = F.linear(output, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                output = F.layer_norm(output, layer.normalized_shape, weight, bias)

        return output


class ContentEncoder(MAMLLayer):
    """内容特征编码器，将音乐特征映射到匹配空间，支持MAML"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=64, dropout=0.1):
        super(ContentEncoder, self).__init__()

        # 内容编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重
        self._init_weights()

        # 保存维度信息
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.encoder(x)

    def forward_with_params(self, x, params=None):
        """使用给定参数的前向传播"""
        if params is None:
            return self.forward(x)

        # 处理内容特征
        output = x
        for i, layer in enumerate(self.encoder):
            layer_name = f'encoder.{i}'
            if isinstance(layer, nn.Linear):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                output = F.linear(output, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = params.get(f'{layer_name}.weight', layer.weight)
                bias = params.get(f'{layer_name}.bias', layer.bias)
                output = F.layer_norm(output, layer.normalized_shape, weight, bias)
            elif isinstance(layer, nn.GELU):
                output = F.gelu(output)
            elif isinstance(layer, nn.Dropout) and self.training:
                output = F.dropout(output, layer.p)

        return output


# ============= 对比学习模块 =============

class ContrastiveLearningModule(nn.Module):
    """对比学习模块，用于学习更好的特征表示"""

    def __init__(self, feature_dim, temperature=0.07):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
        """
        计算归一化投影特征

        参数:
            features: 形状为[batch_size, feature_dim]的特征向量

        返回:
            归一化的投影特征
        """
        # 投影
        projections = self.projection(features)

        # L2归一化
        projections = F.normalize(projections, p=2, dim=1)

        return projections

    def calculate_contrastive_loss(self, anchor_proj, positive_proj, negative_projs=None):
        """
        计算对比损失

        参数:
            anchor_proj: 锚点样本的投影 [batch_size, feature_dim]
            positive_proj: 正样本的投影 [batch_size, feature_dim]
            negative_projs: 负样本的投影列表，每个形状为[batch_size, feature_dim]

        返回:
            对比损失
        """
        batch_size = anchor_proj.shape[0]

        # 如果没有显式提供负样本，将同一批次中的其他样本视为负样本
        if negative_projs is None:
            # 计算相似度矩阵 [batch_size, batch_size]
            sim_matrix = torch.matmul(anchor_proj, positive_proj.T) / self.temperature

            # 正样本对的相似度是对角线元素
            positive_samples = torch.diagonal(sim_matrix)

            # 创建标签: [batch_size]
            labels = torch.arange(batch_size, device=anchor_proj.device)

            # 交叉熵损失
            loss = F.cross_entropy(sim_matrix, labels)
        else:
            # 如果有明确的负样本，计算InfoNCE损失
            # 正样本相似度 [batch_size, 1]
            l_pos = torch.sum(anchor_proj * positive_proj, dim=1, keepdim=True)

            # 负样本相似度 [batch_size, num_negatives]
            l_neg_list = []
            for neg_proj in negative_projs:
                l_neg = torch.matmul(anchor_proj, neg_proj.T)
                l_neg_list.append(l_neg)

            l_neg = torch.cat(l_neg_list, dim=1) if l_neg_list else torch.empty((batch_size, 0),
                                                                                device=anchor_proj.device)

            # Logits: [batch_size, 1+num_negatives]
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

            # 标签: 第一列是正样本 (索引0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_proj.device)

            # 交叉熵损失
            loss = F.cross_entropy(logits, labels)

        return loss


# ============= 增强的推荐系统模型 =============

class EnhancedEEGRecommender(nn.Module):
    """增强的EEG推荐系统，使用注意力机制、多任务学习和元学习支持"""

    def __init__(self, eeg_dim, enhanced_dim, content_dim, feature_dim=64, dropout=0.1,
                 use_contrastive=True, maml_params=None):
        super(EnhancedEEGRecommender, self).__init__()

        # EEG特征提取器
        self.eeg_extractor = EEGFeatureExtractor(
            input_dim=eeg_dim,
            enhanced_dim=enhanced_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )

        # 内容编码器
        self.content_encoder = ContentEncoder(
            input_dim=content_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )

        # 双线性层 - 用于捕获交叉特征
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, 1)

        # 评分预测器
        self.rating_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, feature_dim),  # +2 for similarity and bilinear
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # 输出范围[0,1]
        )

        # 类别匹配预测器
        self.match_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # 输出范围[0,1]
        )

        # 对比学习模块
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_module = ContrastiveLearningModule(feature_dim)

        # 元学习参数
        self.maml_params = maml_params

        # 维度信息
        self.eeg_dim = eeg_dim
        self.enhanced_dim = enhanced_dim
        self.content_dim = content_dim
        self.feature_dim = feature_dim

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, eeg_data, enhanced_eeg, content_data):
        # 提取特征
        eeg_features = self.eeg_extractor(eeg_data, enhanced_eeg)
        content_features = self.content_encoder(content_data)

        # 计算余弦相似度
        similarity = F.cosine_similarity(eeg_features, content_features, dim=1, eps=1e-8).unsqueeze(1)

        # 双线性交互
        bilinear_score = self.bilinear(eeg_features, content_features)

        # 联合输入进行评分预测（现在包括相似度和双线性分数）
        combined_features = torch.cat([eeg_features, content_features, similarity, bilinear_score], dim=1)
        rating = self.rating_predictor(combined_features)

        # 类别匹配预测
        match_features = torch.cat([eeg_features, content_features], dim=1)
        match_score = self.match_predictor(match_features)

        return rating, match_score, eeg_features, content_features

    def process_batch(self, batch, device):
        """处理数据批次"""
        if isinstance(batch, dict):  # 单个样本
            eeg_features = batch['eeg_features'].to(device)
            enhanced_eeg = batch['enhanced_eeg'].to(device)
            content_features = batch['content_features'].to(device)
            ratings = batch['rating'].to(device)

            return eeg_features, enhanced_eeg, content_features, ratings
        else:  # 批次
            eeg_features = torch.stack([item['eeg_features'] for item in batch]).to(device)
            enhanced_eeg = torch.stack([item['enhanced_eeg'] for item in batch]).to(device)
            content_features = torch.stack([item['content_features'] for item in batch]).to(device)
            ratings = torch.stack([item['rating'] for item in batch]).to(device)

            return eeg_features, enhanced_eeg, content_features, ratings

    def meta_learn(self, tasks, device, inner_opt=None):
        """
        Perform meta-learning process

        Args:
            tasks: List of tasks, each with support and query sets
            device: Computation device
            inner_opt: Inner loop optimizer

        Returns:
            Loss on query sets
        """
        if self.maml_params is None:
            raise ValueError("Meta-learning parameters not set")

        meta_batch_size = len(tasks)
        if meta_batch_size == 0:
            return torch.tensor(0.0, device=device)  # Empty batch, return zero loss

        outer_loss = 0.0

        for task in tasks:
            # Clone model parameters
            fast_weights = self.clone_params()

            # Support set
            support = task['support']
            if not support:  # Empty support set, skip this task
                continue

            # Inner loop updates
            for _ in range(self.maml_params.n_inner_steps):
                # Process support set in batches if possible
                support_loss = 0.0

                # Try to create a batch from support set to avoid BatchNorm issues
                if len(support) > 1:
                    # Create batches
                    eeg_batch = torch.stack([s['eeg_features'] for s in support]).to(device)
                    enhanced_batch = torch.stack([s['enhanced_eeg'] for s in support]).to(device)
                    content_batch = torch.stack([s['content_features'] for s in support]).to(device)
                    rating_batch = torch.stack([s['rating'] for s in support]).to(device)

                    # Calculate predictions for the batch
                    rating_pred, _, _, _ = self.forward_with_params(
                        eeg_batch, enhanced_batch, content_batch, fast_weights
                    )

                    # Calculate loss
                    support_loss = F.mse_loss(rating_pred, rating_batch)
                else:
                    # If only one sample, process it individually
                    sample = support[0]
                    eeg_features, enhanced_eeg, content_features, ratings = self.process_batch(sample, device)

                    # Calculate predictions
                    rating_pred, _, _, _ = self.forward_with_params(
                        eeg_features, enhanced_eeg, content_features, fast_weights
                    )

                    # Calculate loss
                    support_loss = F.mse_loss(rating_pred, ratings)

                # Calculate gradients and update parameters
                grads = torch.autograd.grad(
                    support_loss,
                    [p for p in fast_weights.values() if p.requires_grad],
                    create_graph=True,
                    allow_unused=True
                )

                # Update parameters with gradients
                fast_weights = self.update_params(fast_weights, grads, self.maml_params.inner_lr)

            # Query set
            query = task['query']
            if not query:  # Empty query set, skip this task
                continue

            task_loss = 0.0

            # Similarly, process query set in batches if possible
            if len(query) > 1:
                # Create batches
                eeg_batch = torch.stack([s['eeg_features'] for s in query]).to(device)
                enhanced_batch = torch.stack([s['enhanced_eeg'] for s in query]).to(device)
                content_batch = torch.stack([s['content_features'] for s in query]).to(device)
                rating_batch = torch.stack([s['rating'] for s in query]).to(device)

                # Calculate predictions with updated parameters
                rating_pred, _, _, _ = self.forward_with_params(
                    eeg_batch, enhanced_batch, content_batch, fast_weights
                )

                # Calculate loss
                task_loss = F.mse_loss(rating_pred, rating_batch)
            else:
                # If only one sample, process it individually
                for sample in query:
                    eeg_features, enhanced_eeg, content_features, ratings = self.process_batch(sample, device)

                    # Calculate predictions with updated parameters
                    rating_pred, _, _, _ = self.forward_with_params(
                        eeg_features, enhanced_eeg, content_features, fast_weights
                    )

                    # Calculate loss
                    loss = F.mse_loss(rating_pred, ratings)
                    task_loss += loss

                task_loss /= len(query)

            outer_loss += task_loss

        outer_loss /= max(1, meta_batch_size)  # Avoid division by zero
        return outer_loss

    def clone_params(self):
        """克隆模型参数"""
        return {name: param.clone() for name, param in self.named_parameters()}

    def update_params(self, params, grads, lr):
        """使用梯度更新参数"""
        updated = {}
        i = 0
        for name, param in params.items():
            if param.requires_grad:
                if i < len(grads) and grads[i] is not None:
                    updated[name] = param - lr * grads[i]
                else:
                    updated[name] = param
                i += 1
            else:
                updated[name] = param
        return updated

    def forward_with_params(self, eeg_data, enhanced_eeg, content_data, params=None):
        """使用自定义参数计算前向传播"""
        if params is None:
            return self.forward(eeg_data, enhanced_eeg, content_data)

        # 提取特征
        eeg_extractor_params = {k: v for k, v in params.items() if k.startswith('eeg_extractor.')}
        content_encoder_params = {k: v for k, v in params.items() if k.startswith('content_encoder.')}

        # 移除前缀
        eeg_params = {k.replace('eeg_extractor.', ''): v for k, v in eeg_extractor_params.items()}
        content_params = {k.replace('content_encoder.', ''): v for k, v in content_encoder_params.items()}

        # 使用自定义参数进行前向传播
        eeg_features = self.eeg_extractor.forward_with_params(eeg_data, enhanced_eeg, eeg_params)
        content_features = self.content_encoder.forward_with_params(content_data, content_params)

        # 计算余弦相似度
        similarity = F.cosine_similarity(eeg_features, content_features, dim=1, eps=1e-8).unsqueeze(1)

        # 双线性交互
        bilinear_params = {k.replace('bilinear.', ''): v for k, v in params.items() if k.startswith('bilinear.')}
        bilinear_weight = bilinear_params.get('weight', self.bilinear.weight)
        bilinear_bias = bilinear_params.get('bias', self.bilinear.bias)
        bilinear_score = F.bilinear(eeg_features, content_features, bilinear_weight, bilinear_bias)

        # 评分预测
        combined_features = torch.cat([eeg_features, content_features, similarity, bilinear_score], dim=1)
        rating_params = {k.replace('rating_predictor.', ''): v for k, v in params.items()
                         if k.startswith('rating_predictor.')}

        # 使用评分预测器的自定义参数
        rating = combined_features
        for i, layer in enumerate(self.rating_predictor):
            layer_name = f'{i}'
            if isinstance(layer, nn.Linear):
                weight = rating_params.get(f'{layer_name}.weight', layer.weight)
                bias = rating_params.get(f'{layer_name}.bias', layer.bias)
                rating = F.linear(rating, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = rating_params.get(f'{layer_name}.weight', layer.weight)
                bias = rating_params.get(f'{layer_name}.bias', layer.bias)
                rating = F.layer_norm(rating, layer.normalized_shape, weight, bias)
            elif isinstance(layer, nn.GELU):
                rating = F.gelu(rating)
            elif isinstance(layer, nn.Dropout) and self.training:
                rating = F.dropout(rating, layer.p)
            elif isinstance(layer, nn.Sigmoid):
                rating = torch.sigmoid(rating)

        # 类别匹配预测
        match_features = torch.cat([eeg_features, content_features], dim=1)
        match_params = {k.replace('match_predictor.', ''): v for k, v in params.items()
                        if k.startswith('match_predictor.')}

        # 使用匹配预测器的自定义参数
        match_score = match_features
        for i, layer in enumerate(self.match_predictor):
            layer_name = f'{i}'
            if isinstance(layer, nn.Linear):
                weight = match_params.get(f'{layer_name}.weight', layer.weight)
                bias = match_params.get(f'{layer_name}.bias', layer.bias)
                match_score = F.linear(match_score, weight, bias)
            elif isinstance(layer, nn.LayerNorm):
                weight = match_params.get(f'{layer_name}.weight', layer.weight)
                bias = match_params.get(f'{layer_name}.bias', layer.bias)
                match_score = F.layer_norm(match_score, layer.normalized_shape, weight, bias)
            elif isinstance(layer, nn.GELU):
                match_score = F.gelu(match_score)
            elif isinstance(layer, nn.Dropout) and self.training:
                match_score = F.dropout(match_score, layer.p)
            elif isinstance(layer, nn.Sigmoid):
                match_score = torch.sigmoid(match_score)

        return rating, match_score, eeg_features, content_features

    def recommend(self, eeg_data, enhanced_eeg, content_catalog, top_k=5, diversity_weight=0.2, device=None,
                  user_params=None):
        """生成个性化推荐，考虑评分和多样性"""
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        scores = []

        # 确保数据在正确设备上
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.FloatTensor(eeg_data).to(device)
        else:
            eeg_data = eeg_data.to(device)

        if not isinstance(enhanced_eeg, torch.Tensor):
            enhanced_eeg = torch.FloatTensor(enhanced_eeg).to(device)
        else:
            enhanced_eeg = enhanced_eeg.to(device)

        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data.unsqueeze(0)

        if len(enhanced_eeg.shape) == 1:
            enhanced_eeg = enhanced_eeg.unsqueeze(0)

        # 获取EEG特征表示
        with torch.no_grad():
            if user_params is not None:
                # 使用个性化参数提取特征
                eeg_params = {k.replace('eeg_extractor.', ''): v for k, v in user_params.items()
                              if k.startswith('eeg_extractor.')}
                user_embedding = self.eeg_extractor.forward_with_params(eeg_data, enhanced_eeg, eeg_params)
            else:
                # 使用通用参数提取特征
                user_embedding = self.eeg_extractor(eeg_data, enhanced_eeg)

        # 为每个内容项计算分数
        item_scores = []
        item_embeddings = {}

        # In the recommend method of EnhancedEEGRecommender class (continuing from where it was cut off)
        for item_id, item_features_np in content_catalog.items():
            # Convert to tensor
            if not isinstance(item_features_np, torch.Tensor):
                item_features_tensor = torch.FloatTensor(item_features_np).to(device)
            else:
                item_features_tensor = item_features_np.to(device)

            if len(item_features_tensor.shape) == 1:
                item_features_tensor = item_features_tensor.unsqueeze(0)

            # Get content embedding
            with torch.no_grad():
                if user_params is not None:
                    # Use personalized parameters
                    content_params = {k.replace('content_encoder.', ''): v for k, v in user_params.items()
                                      if k.startswith('content_encoder.')}
                    content_embedding = self.content_encoder.forward_with_params(item_features_tensor, content_params)
                else:
                    content_embedding = self.content_encoder(item_features_tensor)

                # Calculate similarity
                similarity = F.cosine_similarity(user_embedding, content_embedding, dim=1, eps=1e-8).item()

                # Calculate bilinear score
                bilinear_score = self.bilinear(user_embedding, content_embedding).item()

                # Combine features for prediction
                combined = torch.cat([user_embedding, content_embedding,
                                      torch.tensor([[similarity]], device=device),
                                      torch.tensor([[bilinear_score]], device=device)], dim=1)

                # Predict rating
                rating = self.rating_predictor(combined).item()

            # Store results
            item_scores.append((item_id, rating, similarity))
            item_embeddings[item_id] = content_embedding.cpu().numpy()

        # Sort by predicted rating
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Add diversity - select items that are different from each other
        if diversity_weight > 0 and len(item_scores) > top_k:
            selected_items = [item_scores[0]]  # Start with highest rated item
            item_scores = item_scores[1:]

            while len(selected_items) < top_k and item_scores:
                # Calculate diversity score for each remaining item
                max_diversity_score = -1
                best_item_idx = 0

                for i, (item_id, rating, similarity) in enumerate(item_scores):
                    # Calculate average similarity to already selected items
                    avg_sim = 0
                    for sel_id, sel_rating, sel_sim in selected_items:
                        # Get embeddings for both items
                        sel_embedding = item_embeddings[sel_id]
                        cur_embedding = item_embeddings[item_id]

                        # Calculate cosine similarity
                        sim = np.dot(sel_embedding.flatten(), cur_embedding.flatten()) / (
                                np.linalg.norm(sel_embedding) * np.linalg.norm(cur_embedding) + 1e-8)
                        avg_sim += sim

                    avg_sim /= len(selected_items)

                    # Combined score: rating and diversity
                    diversity = 1 - avg_sim  # Higher value means more diverse
                    combined_score = (1 - diversity_weight) * rating + diversity_weight * diversity

                    if combined_score > max_diversity_score:
                        max_diversity_score = combined_score
                        best_item_idx = i

                # Add the best item to selected list
                selected_items.append(item_scores[best_item_idx])
                item_scores.pop(best_item_idx)

            recommendations = selected_items
        else:
            # Just take top-k by rating
            recommendations = item_scores[:top_k]

        return recommendations


# ============= 训练和评估函数 =============

def train_model(model, train_loader, val_loader=None,
                epochs=10, lr=0.001, meta_batch_size=4, device='cuda',
                early_stopping_patience=5, contrastive_weight=0.2,
                meta_weight=0.3, verbose=True):
    """
    训练EEG推荐系统模型

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        meta_batch_size: 元学习批次大小
        device: 计算设备
        early_stopping_patience: 早停耐心值
        contrastive_weight: 对比学习损失权重
        meta_weight: 元学习损失权重
        verbose: 是否打印详细信息
    """
    # 将模型移至设备
    model.to(device)

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=verbose
    )

    # 早停
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # 跟踪指标
    train_losses = []
    val_losses = []
    rating_losses = []
    meta_losses = []
    contrastive_losses = []

    for epoch in range(epochs):
        # 训练模式
        model.train()
        epoch_loss = 0.0
        epoch_rating_loss = 0.0
        epoch_meta_loss = 0.0
        epoch_contrastive_loss = 0.0

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
            pbar = tqdm(total=len(train_loader))

        # 正常训练
        for batch in train_loader:
            optimizer.zero_grad()

            # 处理数据
            eeg_features = batch['eeg_features'].to(device)
            enhanced_eeg = batch['enhanced_eeg'].to(device)
            content_features = batch['content_features'].to(device)
            ratings = batch['rating'].to(device)

            # 前向传播
            pred_ratings, pred_match, eeg_emb, content_emb = model(
                eeg_features, enhanced_eeg, content_features
            )

            # 评分损失 (MSE)
            rating_loss = F.mse_loss(pred_ratings, ratings)

            # 对比学习损失 (可选)
            contrastive_loss = torch.tensor(0.0, device=device)
            if model.use_contrastive:
                # 为正样本对和负样本对创建投影
                eeg_proj = model.contrastive_module(eeg_emb)
                content_proj = model.contrastive_module(content_emb)

                # 为负样本创建索引
                batch_size = eeg_emb.shape[0]
                neg_indices = torch.randperm(batch_size)

                # 创建负样本投影
                neg_content_proj = content_proj[neg_indices]

                # 计算对比损失
                contrastive_loss = model.contrastive_module.calculate_contrastive_loss(
                    eeg_proj, content_proj, [neg_content_proj]
                )

            # 元学习损失 (可选)
            meta_loss = torch.tensor(0.0, device=device)
            if model.maml_params is not None and epoch > 0:  # 从第二个epoch开始使用元学习
                # 创建元学习批次
                meta_batch = train_loader.dataset.get_meta_batch(meta_batch_size)
                if meta_batch:
                    meta_loss = model.meta_learn(meta_batch, device)

            # 总损失
            loss = rating_loss + contrastive_weight * contrastive_loss + meta_weight * meta_loss

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()
            epoch_rating_loss += rating_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_meta_loss += meta_loss.item()

            if verbose:
                pbar.update(1)

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_rating_loss = epoch_rating_loss / len(train_loader)
        avg_contrastive_loss = epoch_contrastive_loss / len(train_loader)
        avg_meta_loss = epoch_meta_loss / len(train_loader)

        train_losses.append(avg_loss)
        rating_losses.append(avg_rating_loss)
        contrastive_losses.append(avg_contrastive_loss)
        meta_losses.append(avg_meta_loss)

        if verbose:
            pbar.close()
            print(f"训练损失: {avg_loss:.4f} (评分: {avg_rating_loss:.4f}, "
                  f"对比: {avg_contrastive_loss:.4f}, 元学习: {avg_meta_loss:.4f})")

        # 验证
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, device, verbose=False)
            val_losses.append(val_loss)

            if verbose:
                print(f"验证损失: {val_loss:.4f}")

            # 学习率调度
            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"早停触发于epoch {epoch + 1}")
                    break
        else:
            # 如果没有验证集，使用训练损失进行调度
            scheduler.step(avg_loss)

            # 保存最佳模型
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_model_state = model.state_dict().copy()

    # 恢复最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 返回训练和验证损失
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rating_losses': rating_losses,
        'contrastive_losses': contrastive_losses,
        'meta_losses': meta_losses
    }


def evaluate_model(model, data_loader, device, verbose=True):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            # 处理数据
            eeg_features = batch['eeg_features'].to(device)
            enhanced_eeg = batch['enhanced_eeg'].to(device)
            content_features = batch['content_features'].to(device)
            ratings = batch['rating'].to(device)

            # 前向传播
            pred_ratings, _, _, _ = model(eeg_features, enhanced_eeg, content_features)

            # 计算损失
            loss = F.mse_loss(pred_ratings, ratings)
            total_loss += loss.item()

            # 收集预测和目标
            all_preds.extend(pred_ratings.cpu().numpy().flatten())
            all_targets.extend(ratings.cpu().numpy().flatten())
            scaled_preds = np.array(all_preds) * 3.5  # 恢复到1-5尺度
            scaled_targets = np.array(all_targets) * 3.5


    # 计算指标
    mse = mean_squared_error(scaled_targets, scaled_preds)
    mae = mean_absolute_error(scaled_targets, scaled_preds)

    # 将预测和目标转换为二分类 (高评分 vs 低评分)
    binary_preds = [1 if p >= 0.6 else 0 for p in all_preds]  # 0.6对应3/5评分
    binary_targets = [1 if t >= 0.6 else 0 for t in all_targets]

    # 计算分类指标
    precision = precision_score(binary_targets, binary_preds, zero_division=0)
    recall = recall_score(binary_targets, binary_preds, zero_division=0)
    f1 = f1_score(binary_targets, binary_preds, zero_division=0)

    if verbose:
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"分类 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_and_evaluate_full_system(eeg_data_path='D:/MUSIC-REC-EEG/EEG_music-main/code/cut_data/',
                                   batch_size=64,
                                   num_epochs=50,
                                   feature_dim=64,
                                   learning_rate=0.001,
                                   early_stopping_patience=7,
                                   use_contrastive=False,
                                   use_meta_learning=True,
                                   test_size=0.2,
                                   random_state=42,
                                   device=None):
    """
    训练和评估完整的EEG音乐推荐系统

    参数:
        eeg_data_path: EEG数据路径
        batch_size: 批次大小
        num_epochs: 训练轮数
        feature_dim: 特征维度
        learning_rate: 学习率
        early_stopping_patience: 早停耐心值
        use_contrastive: 是否使用对比学习
        use_meta_learning: 是否使用元学习
        test_size: 测试集大小
        random_state: 随机种子
        device: 计算设备
    """
    # 设置随机种子
    set_seed(random_state)

    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载EEG数据...")
    label, data, data_withpsd, ratings, user_mapping = load_eeg_data()

    print(f"数据加载完成: {len(data)}条记录")
    print(f"EEG特征维度: {data.shape[1]}")
    print(f"带PSD的EEG特征维度: {data_withpsd.shape[1]}")

    # 创建推荐数据集
    print("创建推荐数据集...")
    eeg_data, enhanced_eeg, music_features, interactions, music_class_labels, user_mapping = create_recommendation_dataset(
        label, data_withpsd, ratings, create_interactions=True, user_mapping=user_mapping
    )

    print(f"创建了{len(interactions)}个用户-音乐交互")

    # 划分训练集和测试集 - 按用户分组确保同一用户的数据不会同时出现在训练集和测试集中
    unique_users = np.unique([u for u, _, _ in interactions])
    train_users, test_users = train_test_split(
        unique_users, test_size=test_size, random_state=random_state
    )

    print(f"训练用户: {len(train_users)}, 测试用户: {len(test_users)}")

    # 按用户分组交互
    train_interactions = [inter for inter in interactions if inter[0] in train_users]
    test_interactions = [inter for inter in interactions if inter[0] in test_users]

    print(f"训练交互: {len(train_interactions)}, 测试交互: {len(test_interactions)}")

    # 创建数据集
    train_dataset = EnhancedEEGDataset(
        eeg_data, enhanced_eeg, music_features, train_interactions,
        user_mapping=user_mapping, augment=True, support_query_split=False
    )

    test_dataset = EnhancedEEGDataset(
        eeg_data, enhanced_eeg, music_features, test_interactions,
        user_mapping=user_mapping, augment=False, support_query_split=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 确定特征维度
    sample = next(iter(train_loader))
    eeg_dim = sample['eeg_features'].shape[1]
    enhanced_dim = sample['enhanced_eeg'].shape[1]
    content_dim = sample['content_features'].shape[1]

    print(f"特征维度 - EEG: {eeg_dim}, 增强EEG: {enhanced_dim}, 内容: {content_dim}")

    # 创建MAML参数
    maml_params = MAMLParams(inner_lr=0.05, n_inner_steps=1) if use_meta_learning else None

    # 创建模型
    model = EnhancedEEGRecommender(
        eeg_dim=eeg_dim,
        enhanced_dim=enhanced_dim,
        content_dim=content_dim,
        feature_dim=feature_dim,
        dropout=0.2,
        use_contrastive=use_contrastive,
        maml_params=maml_params
    )

    print(f"模型创建完成，总参数数量: {sum(p.numel() for p in model.parameters())}")

    # 训练模型
    print("开始训练...")
    train_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        lr=learning_rate,
        meta_batch_size=8,
        device=device,
        early_stopping_patience=early_stopping_patience,
        contrastive_weight=0.3 if use_contrastive else 0.0,
        meta_weight=0.4 if use_meta_learning else 0.0,
        verbose=True
    )

    # 评估模型
    print("评估模型性能...")
    test_loss = evaluate_model(model, test_loader, device, verbose=True)
    print(f"测试损失: {test_loss:.4f}")

    # 可视化训练历史
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_history['train_losses'], label='训练损失')
    if train_history['val_losses']:
        plt.plot(train_history['val_losses'], label='验证损失')
    plt.title('总损失')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_history['rating_losses'], label='评分损失')
    plt.title('评分损失')
    plt.legend()

    if use_contrastive:
        plt.subplot(2, 2, 3)
        plt.plot(train_history['contrastive_losses'], label='对比损失')
        plt.title('对比学习损失')
        plt.legend()

    if use_meta_learning:
        plt.subplot(2, 2, 4)
        plt.plot(train_history['meta_losses'], label='元学习损失')
        plt.title('元学习损失')
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # 为每个测试用户生成推荐并评估
    print("为测试用户生成推荐...")

    # 用户推荐结果评估
    user_recommendations = {}
    recommendation_hits = []
    recommendation_ndcg = []

    for user_id in test_users:
        # 获取用户的EEG数据
        if isinstance(eeg_data[user_id], list):
            user_eeg = eeg_data[user_id][0]  # 使用第一条EEG记录
        else:
            user_eeg = eeg_data[user_id]

        if isinstance(enhanced_eeg[user_id], list):
            user_enhanced = enhanced_eeg[user_id][0]
        else:
            user_enhanced = enhanced_eeg[user_id]

        # 获取该用户在测试集中的交互记录
        user_test_interactions = [inter for inter in test_interactions if inter[0] == user_id]

        # 创建一个包含所有音乐项目的内容目录
        content_catalog = {}
        for _, item_id, _ in user_test_interactions:
            content_catalog[item_id] = music_features[item_id]

        # 添加一些随机项目以扩展推荐范围
        random_items = np.random.choice(
            [i for i in music_features.keys() if i not in content_catalog],
            min(50, len(music_features) - len(content_catalog)),
            replace=False
        )

        for item_id in random_items:
            content_catalog[item_id] = music_features[item_id]

        # 生成推荐
        recommendations = model.recommend(
            user_eeg, user_enhanced, content_catalog,
            top_k=10, diversity_weight=0.1, device=device
        )

        # 存储推荐结果
        user_recommendations[user_id] = [(item_id, score) for item_id, score, _ in recommendations]

        # 计算命中率 - 检查用户喜欢的项目是否在推荐中
        liked_items = [item_id for _, item_id, rating in user_test_interactions if rating >= 0.6]
        recommended_items = [item_id for item_id, _, _ in recommendations]

        hits = len(set(liked_items) & set(recommended_items))
        hit_rate = hits / len(liked_items) if liked_items else 0
        recommendation_hits.append(hit_rate)

        # 计算NDCG
        # 创建相关性列表 (1表示项目在用户喜欢的项目中)
        relevance = [1 if item_id in liked_items else 0 for item_id, _, _ in recommendations]

        # 计算DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

        # 计算理想DCG (所有相关项目排在前面)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        # 计算NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        recommendation_ndcg.append(ndcg)

    # 打印推荐性能
    avg_hit_rate = np.mean(recommendation_hits)
    avg_ndcg = np.mean(recommendation_ndcg)

    print(f"推荐评估结果:")
    print(f"平均命中率: {avg_hit_rate:.4f}")
    print(f"平均NDCG: {avg_ndcg:.4f}")

    # 返回训练后的模型和评估结果
    return model, {
        'test_loss': test_loss,
        'hit_rate': avg_hit_rate,
        'ndcg': avg_ndcg,
        'user_recommendations': user_recommendations,
        'train_history': train_history
    }


# ============= 主函数 =============

def main():
    # 设置随机种子以确保可重复性
    set_seed(42)

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载EEG数据...")
    label, data, data_withpsd, ratings, user_mapping = load_eeg_data()

    print(f"数据加载完成: {len(data)}条记录")
    print(f"EEG特征维度: {data.shape[1]}")
    print(f"带PSD的EEG特征维度: {data_withpsd.shape[1]}")

    # 训练和评估
    model, results = train_and_evaluate_full_system(
        batch_size=32,
        num_epochs=20,
        feature_dim=64,
        learning_rate=0.001,
        use_contrastive=True,
        use_meta_learning=True,
        device=device
    )

    # 打印结果
    print("\n最终评估结果:")
    print(f"测试损失: {results['test_loss']:.4f}")
    print(f"命中率: {results['hit_rate']:.4f}")
    print(f"NDCG: {results['ndcg']:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'eeg_recommender_model.pth')
    print("模型已保存至 'eeg_recommender_model.pth'")

    # 保存结果
    with open('evaluation_results.json', 'w') as f:
        # 将numpy数组和张量转换为Python原生类型
        serializable_results = {
            'test_loss': float(results['test_loss']),
            'hit_rate': float(results['hit_rate']),
            'ndcg': float(results['ndcg']),
            'user_recommendations': {
                str(user_id): [(int(item_id), float(score)) for item_id, score in recs]
                for user_id, recs in results['user_recommendations'].items()
            },
            'train_history': {
                'train_losses': [float(x) for x in results['train_history']['train_losses']],
                'val_losses': [float(x) for x in results['train_history']['val_losses']],
                'rating_losses': [float(x) for x in results['train_history']['rating_losses']],
                'contrastive_losses': [float(x) for x in results['train_history']['contrastive_losses']],
                'meta_losses': [float(x) for x in results['train_history']['meta_losses']]
            }
        }
        json.dump(serializable_results, f, indent=4)

    print("评估结果已保存至 'evaluation_results.json'")


if __name__ == "__main__":
    main()