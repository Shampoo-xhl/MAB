import numpy as np
from tqdm import tqdm
from itertools import chain
# from tkinter import _flatten
import random
import load_IJCAI
import deltaT_gaussian_init_arms
import item_reward
import get_regret

lamada = 3.0    # 超参

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def test_TS(u):  # 测试集TS算法推荐
    N = 3   # 商家、类别、品牌各摇臂一次

    # seller_arm
    sampling_seller = [0] * seller_nums  # t时刻每个动作的θ值
    # brand_arm
    sampling_brand = [0] * brand_nums
    # cat_arm
    sampling_cat = [0] * cat_nums

    j_item = [0] * N
    Hit_u = []  # 用户u的所有击中产品
    Hit = [0] * N
    Rec_u = []
    u_index = user_info[user_info['user_id'] == u].index.tolist()[0]  # 获取索引
    user_log = test_data[test_data['user_id'] == u]  # 用户u的记录

    regret = 0  # 累计遗憾
    t_num = 499    # 从1000算起

    Hit_nums = 0  # 击中次数
    Reco_nums = 0  # 推荐总次数

    for t in test_data[test_data['user_id'] == u].reset_index(drop=True).itertuples():
        t_num += 1
        if t_num % 10 == 0:  # 每10轮有一次不推荐
            continue
        Reco_nums += 1  # 推荐次数+1

        # 商家推荐
        for seller_index in range(seller_nums):  # 对每个臂进行gaussian采样
            sampling_seller[seller_index] = random.gauss(theta_seller[u_index][seller_index],
                                                        1.0/(var_seller[u_index][seller_index]))
        best_seller_index = np.argmax(sampling_seller)  # 最大theta的索引
        best_seller_id = seller_list[best_seller_index]  # 最大theta对应的商家
        flag_seller, Hit[0], S_best, W_best, j_item[0] = item_reward.get_seller_reward(t, best_seller_id, user_log, 1)  # 最佳动作产生推荐，获得反馈S_best,W_best, TVflag=1
        # temp_theta1_seller = theta1_seller
        # theta1_seller[best_seller_index] += S_best # 奖励
        if(~np.isnan(t.seller_id)):
            seller_index = seller_list.index(t.seller_id)
            map(lambda x: x + 1, deltaT_seller[u_index]) #比for循环效率高
            deltaT_seller[u_index][seller_index] = 0.1  # 刚拉动的臂deltaT置1
            temp_reward = S_best * user_reward_weight[u_index][0]
            temp_weight = (abs(theta_seller[u_index][seller_index] - max(theta_seller[u_index])) + lamada)
            sum = user_reward_weight[u_index][0] * t_num + temp_weight
            user_reward_weight[u_index][0] = sum / (t_num + 1)
            theta_seller[u_index][seller_index] = theta_seller[u_index][seller_index] * sigmoid(1.0/deltaT_seller[u_index][seller_index]) + temp_reward
            var_seller[u_index][best_seller_index] += 1

        regret_seller = get_regret.get_seller_regret(t, best_seller_id, user_log)  # 计算遗憾

        # 品牌推荐
        for brand_index in range(brand_nums):
            sampling_brand[brand_index] = random.gauss(theta_brand[u_index][brand_index],
                                                      1.0/(var_brand[u_index][brand_index]))
        best_brand_index = np.argmax(sampling_brand)
        best_brand_id = brand_list[best_brand_index]
        flag_brand, Hit[1], S_best, W_best, j_item[1] = item_reward.get_brand_reward(t, best_brand_id, user_log, 1)
        if(~np.isnan(t.brand_id)):
            brand_index = brand_list.index(t.brand_id)
            map(lambda x: x + 1, deltaT_brand[u_index])
            deltaT_brand[u_index][brand_index] = 0.1  # 刚拉动的臂deltaT置1
            temp_reward = S_best * user_reward_weight[u_index][1]
            temp_weight = (abs(theta_brand[u_index][brand_index] - max(theta_brand[u_index])) + lamada)
            sum = user_reward_weight[u_index][1] * t_num + temp_weight
            user_reward_weight[u_index][1] = sum / (t_num + 1)
            theta_brand[u_index][brand_index] = theta_brand[u_index][brand_index] * sigmoid(1.0/deltaT_brand[u_index][brand_index]) + temp_reward
            var_brand[u_index][best_brand_index] += 1

        regret_brand = get_regret.get_brand_regret(t, best_brand_id, user_log)  # 计算遗憾

        # 类别推荐
        for cat_index in range(cat_nums):
            sampling_cat[cat_index] = random.gauss(theta_cat[u_index][cat_index],
                                                  1.0/(var_cat[u_index][cat_index]))
        best_cat_index = np.argmax(sampling_cat)
        best_cat_id = cat_list[best_cat_index]
        flag_cat, Hit[2], S_best, W_best, j_item[2] = item_reward.get_cat_reward(t, best_cat_id, user_log, 1)
        if(~np.isnan(t.cat_id)):
            cat_index = cat_list.index(t.cat_id)
            map(lambda x: x + 1, deltaT_cat[u_index])
            deltaT_cat[u_index][cat_index] = 0.1  # 刚拉动的臂deltaT置1
            temp_reward = S_best * user_reward_weight[u_index][2]
            temp_weight = (abs(theta_cat[u_index][cat_index] - max(theta_cat[u_index])) + lamada)
            sum = user_reward_weight[u_index][2] * t_num + temp_weight
            user_reward_weight[u_index][2] = sum / (t_num + 1)
            theta_cat[u_index][cat_index] = theta_cat[u_index][cat_index] * sigmoid(1.0/deltaT_cat[u_index][cat_index]) + temp_reward
            var_cat[u_index][best_cat_index] += 1

        regret_cat = get_regret.get_cat_regret(t, best_cat_id, user_log)  # 计算遗憾

        '''计算累计遗憾'''
        regret += (regret_seller + regret_brand + regret_cat) / 3

        '''统计t时刻推荐完成后的推荐产品列表和击中列表'''
        Rec_t = list(chain.from_iterable(j_item))  # 当前时刻的推荐列表
        Hit_t = list(chain.from_iterable(Hit))  # 当前时刻的击中
        Rec_u.append(Rec_t)
        Hit_u.append(Hit_t)

        '''计算击中数'''
        if (flag_seller + flag_brand + flag_cat) > 0:
            Hit_nums += 1

    """计算评价指标"""

    buy_S = list(user_log[(user_log.action_type == 2)]['item_id'].values)  # 用户购买个数
    cart_S = list(user_log[(user_log.action_type == 1)]['item_id'].values)  # 用户加购个数
    fav_S = list(user_log[(user_log.action_type == 3)]['item_id'].values)  # 用户收藏个数
    click_S = list(user_log[(user_log.action_type == 0)]['item_id'].values)  # 用户点击个数


    event_item = list(set(click_S + buy_S + cart_S + fav_S))  # 用户交互的产品个数(减掉点击的物品)

    Hit_u = list(set(list(chain.from_iterable(Hit_u))))  # 用户u所有的击中列表（已去重）
    event_len = len(set(event_item))
    if event_len == 0:  # 防止分母为0
        event_len = 1
    recall = len(Hit_u) / event_len  # 去重后召回率
    if Reco_nums == 0:  # 防止分母为0
        Reco_nums = 1
    precision = Hit_nums / Reco_nums #

    # print('~~~~~~~~~~~~~~~~~~~~~对用户',u,'的推荐结果~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('precision = ', precision)
    # print('recall = ', recall)
    # print('累计遗憾', regret)
    return recall, precision, Rec_u, regret


if __name__ == '__main__':
    # 加载数据
    train_data, test_data, item_info, user_info, item_list, user_id = load_IJCAI.get_IJCAI()
    # 获取物品特征信息
    cat_list, brand_list, seller_list = load_IJCAI.get_item_feature(item_info)
    cat_nums = len(cat_list)  # 1053
    brand_nums = len(brand_list)  # 5695
    seller_nums = len(seller_list)  # 4949
    item_nums = len(item_list)

    # # 建臂 得到空的奖励分布 N(0,1)
    # theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat = deltaT_gaussian_init_arms.initial_values(seller_nums, brand_nums, cat_nums, user_id)
    # # 训练集初始化臂的奖励分布，存入csv文件中
    # deltaT_gaussian_init_arms.init_train_reward(user_id, seller_list, brand_list, cat_list, theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat)

    theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat = load_IJCAI.get_deltaT_reward_distribution()
    user_reward_weight = load_IJCAI.get_user_reward_weight()


    sum_p = 0
    sum_r = 0
    Rec_u = [0] * len(user_id)
    Regret = [0] * len(user_id)

    # i = 0
    for u in tqdm(user_id):
        # i += 1
        recall, precision, Rec_u[user_id.index(u)], Regret[user_id.index(u)] = test_TS(u)
        sum_p += precision
        sum_r += recall
        # if i == 10:
        #     break

    # recom_list = set(np.array(Rec_u).flatten().tolist())
    # recom_list = set(list(_flatten(Rec_u)))  # 二维数组拉为一维数组
    # avg_p = sum_p / len(user_id)
    # avg_r = sum_r / len(user_id)
    avg_p = sum_p / 1000
    avg_r = sum_r / 1000
    F1 = 2 * avg_p * avg_r / (avg_p + avg_r)
    # coverage = len(recom_list) / len(item_list)

    print("--------------------------最终结果-------------------------")
    print("平均精确率：", avg_p)
    print("平均召回率：", avg_r)
    print("F1值：", F1)
    print("总遗憾：", sum(Regret))
    print("lamada:",lamada)
