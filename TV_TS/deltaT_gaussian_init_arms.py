import numpy as np
import pandas as pd
from tqdm import tqdm

train_data = pd.read_csv('../IJCAI_15_dataset/train_data_1000.csv')  # 137403
user_info = pd.read_csv('../IJCAI_15_dataset/user_1000.csv')    # 100

lamada = 0.0

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# 建臂 得到空的奖励分布：悲观初始值
def initial_values(seller_nums, brand_nums, cat_nums, user_id):
    theta_cat = np.full((len(user_id), cat_nums), 0, dtype=float)     # 二维数据用0填充
    var_cat = np.full((len(user_id), cat_nums), 1, dtype=float)
    deltaT_cat = np.full((len(user_id), cat_nums), 0, dtype=float) # 距离上一次拉动臂的时间间隔

    theta_brand = np.full((len(user_id), brand_nums), 0, dtype=float)
    var_brand = np.full((len(user_id), brand_nums), 1, dtype=float)
    deltaT_brand = np.full((len(user_id), brand_nums), 0, dtype=float) # 距离上一次拉动臂的时间间隔

    theta_seller = np.full((len(user_id), seller_nums), 0, dtype=float)
    var_seller = np.full((len(user_id), seller_nums), 1, dtype=float)
    deltaT_seller = np.full((len(user_id), seller_nums), 0, dtype=float) # 距离上一次拉动臂的时间间隔

    return theta_seller, var_seller, deltaT_seller, \
           theta_brand, var_brand, deltaT_brand, \
           theta_cat, var_cat, deltaT_cat


'''商家奖励'''
def init_seller_reward(user_log, u_index, seller_list, theta_seller, var_seller, deltaT_seller, user_reward_weight):
    reward_dict = {0: 1, 1: 2, 2: 3, 3: 2}  # 【action_type: weight】
    t_num = 0
    for log in user_log.reset_index(drop=True).itertuples():    # 遍历计算reward
        t_num += 1
        if(np.isnan(log.seller_id)):
            continue
        seller_index = seller_list.index(log.seller_id) # 当前拉动的臂
        map(lambda x: x + 1, deltaT_seller[u_index])
        deltaT_seller[u_index][seller_index] = 0.1 # 刚拉动的臂deltaT置1
        reward = reward_dict[log.action_type] * user_reward_weight[u_index][0]
        temp_weight = (abs(theta_seller[u_index][seller_index]-max(theta_seller[u_index]))+lamada)
        sum = user_reward_weight[u_index][0] * t_num + temp_weight
        # print(sum,' ', t_num+1)
        # user_reward_weight: 每个用户拥有一个奖励权重
        user_reward_weight[u_index][0] = sum/(t_num+1)
        # print(temp_weight, " ", user_reward_weight[u_index][0]);
        theta_seller[u_index][seller_index] = theta_seller[u_index][seller_index]*sigmoid(1.0/deltaT_seller[u_index][seller_index]) + reward
        var_seller[u_index][seller_index] += 1


'''品牌奖励'''
def init_brand_reward(user_log, u_index, brand_list, theta_brand, var_brand, deltaT_brand, user_reward_weight):
    reward_dict = {0: 1, 1: 2, 2: 3, 3: 2}  # 【action_type: weight】
    t_num = 0
    for log in user_log.reset_index(drop=True).itertuples():    # 遍历计算reward
        t_num += 1
        if(np.isnan(log.brand_id)):
            continue
        brand_index = brand_list.index(log.brand_id)
        map(lambda x: x + 1, deltaT_brand[u_index])
        deltaT_brand[u_index][brand_index] = 0.1 # 刚拉动的臂deltaT置1
        reward = reward_dict[log.action_type] * user_reward_weight[u_index][1]
        temp_weight = (abs(theta_brand[u_index][brand_index]-max(theta_brand[u_index]))+lamada)
        sum = user_reward_weight[u_index][1] * t_num + temp_weight
        user_reward_weight[u_index][1] = sum/(t_num+1)
        theta_brand[u_index][brand_index] = theta_brand[u_index][brand_index]*sigmoid(1.0/deltaT_brand[u_index][brand_index]) + reward
        var_brand[u_index][brand_index] += 1

'''类别奖励'''
def init_cat_reward(user_log, u_index, cat_list, theta_cat, var_cat, deltaT_cat, user_reward_weight):
    reward_dict = {0: 1, 1: 2, 2: 3, 3: 2}  # 【action_type: weight】
    t_num = 0

    for log in user_log.reset_index(drop=True).itertuples():    # 遍历计算reward
        t_num += 1
        if(np.isnan(log.cat_id)):
            continue
        cat_index = cat_list.index(log.cat_id)
        map(lambda x: x + 1, deltaT_cat[u_index])
        deltaT_cat[u_index][cat_index] = 0.1 # 刚拉动的臂deltaT置1
        reward = reward_dict[log.action_type] * user_reward_weight[u_index][2]
        temp_weight = (abs(theta_cat[u_index][cat_index] - max(theta_cat[u_index])) + lamada)
        sum = user_reward_weight[u_index][2] * t_num + temp_weight
        user_reward_weight[u_index][2] = sum / (t_num+1)
        theta_cat[u_index][cat_index] = theta_cat[u_index][cat_index]*sigmoid(1.0/deltaT_cat[u_index][cat_index]) + reward
        var_cat[u_index][cat_index] += 1


# 为每个用户 初始化臂的奖励分布
def init_reward(u, seller_list, brand_list, cat_list, theta_seller, var_seller, deltaT_seller,
                theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat, user_reward_weight):
    user_log = train_data[train_data['user_id'] == u]
    u_index = user_info[user_info['user_id'] == u].index.tolist()[0]  # 获取索引
    init_seller_reward(user_log, u_index, seller_list, theta_seller, var_seller, deltaT_seller, user_reward_weight)
    init_brand_reward(user_log, u_index, brand_list, theta_brand, var_brand, deltaT_brand, user_reward_weight)
    init_cat_reward(user_log, u_index, cat_list, theta_cat, var_cat, deltaT_cat, user_reward_weight)


# 预训练初始化臂的奖励分布
def init_train_reward(user_id, seller_list, brand_list, cat_list, theta_seller, var_seller, deltaT_seller,
                      theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat):
    user_reward_weight = np.full((len(user_id), 3), 1, dtype=float)

    for u in tqdm(user_id):
        init_reward(u, seller_list, brand_list, cat_list, theta_seller, var_seller, deltaT_seller,
                    theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat, user_reward_weight)
    user_reward_weight = pd.DataFrame(user_reward_weight)

    theta_seller = pd.DataFrame(theta_seller)
    var_seller = pd.DataFrame(var_seller)
    deltaT_seller = pd.DataFrame(deltaT_seller)

    theta_brand = pd.DataFrame(theta_brand)
    var_brand = pd.DataFrame(var_brand)
    deltaT_brand = pd.DataFrame(deltaT_brand)

    theta_cat = pd.DataFrame(theta_cat)
    var_cat = pd.DataFrame(var_cat)
    deltaT_cat = pd.DataFrame(deltaT_cat)

    user_reward_weight.to_csv('./TVTS_arm_reward_data/user_reward_weight.csv', index=False)

    theta_seller.to_csv('./TVTS_arm_reward_data/theta_seller.csv', index=False)
    var_seller.to_csv('./TVTS_arm_reward_data/var_seller.csv', index=False)
    deltaT_seller.to_csv('./TVTS_arm_reward_data/deltaT_seller.csv', index=False)

    theta_brand.to_csv('./TVTS_arm_reward_data/theta_brand.csv', index=False)
    var_brand.to_csv('./TVTS_arm_reward_data/var_brand.csv', index=False)
    deltaT_brand.to_csv('./TVTS_arm_reward_data/deltaT_brand.csv', index=False)

    theta_cat.to_csv('./TVTS_arm_reward_data/theta_cat.csv', index=False)
    var_cat.to_csv('./TVTS_arm_reward_data/var_cat.csv', index=False)
    deltaT_cat.to_csv('./TVTS_arm_reward_data/deltaT_cat.csv', index=False)



