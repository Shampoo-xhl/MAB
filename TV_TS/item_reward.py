import random
import pandas as pd

item_info = pd.read_csv('../IJCAI_15_dataset/item_info_1000.csv')   # 449010
test_data = pd.read_csv('../IJCAI_15_dataset/test_data_1000.csv')   # 307104
user_info = pd.read_csv('../IJCAI_15_dataset/user_1000.csv')
item_list = item_info['item_id'].value_counts().index.tolist()  # 278426
item_nums = len(item_list)

def get_seller_item(t, seller_id, TVflag):
    u_index = user_info[user_info['user_id'] == t.user_id].index.tolist()[0]  # 获取索引
    filter_item = item_info[(item_info['seller_id'] == seller_id)]
    recommend_list = filter_item['item_id'].tolist()
    '''策略一：随机产生1个'''
    a_item = []
    if len(recommend_list) != 0:
        if (t.item_id in recommend_list) &((t.action_type == 1) | (t.action_type == 3)):  # 加购或收藏
            a_item.append(t.item_id)
        else:
            a_item = random.sample(list(recommend_list), 1)
    return recommend_list, a_item

def get_brand_item(t, brand_id, TVflag):
    u_index = user_info[user_info['user_id'] == t.user_id].index.tolist()[0]  # 获取索引
    filter_item = item_info[(item_info['brand_id'] == brand_id)]
    recommend_list = filter_item['item_id'].tolist()
    '''策略一：随机产生1个'''
    a_item = []
    if len(recommend_list) != 0:
        if (t.item_id in recommend_list) &((t.action_type == 1) | (t.action_type == 3)):  # 加购或收藏
            a_item.append(t.item_id)
        else:
            a_item = random.sample(list(recommend_list), 1)
    return recommend_list, a_item

def get_cat_item(t, cat_id, TVflag):
    u_index = user_info[user_info['user_id'] == t.user_id].index.tolist()[0]  # 获取索引
    filter_item = item_info[(item_info['cat_id'] == cat_id)]
    recommend_list = filter_item['item_id'].tolist()

    '''策略一：随机产生1个'''
    a_item = []
    if len(recommend_list) != 0:
        if (t.item_id in recommend_list) & ((t.action_type == 1) | (t.action_type == 3)):  # 加购或收藏
            a_item.append(t.item_id)
        else:
            a_item = random.sample(list(recommend_list), 1)
    return recommend_list, a_item


def get_seller_reward(t, seller_id, user_data, TVflag):  # 判断选择当前商家是否有奖励
    flag = 0
    S = 0   # 强偏好
    W = 0   # 弱偏好
    a_item = []
    Hit = [] 
    user_data.reset_index(drop=True, inplace=True)   # 未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id'] == t.item_id) & (user_data['action_type'] == t.action_type) & (user_data['time_stamp']==t.time_stamp)].index.values[0]
    
    verification = user_data[m+1:n]    # 验证数据   
    a_recommend, a_item = get_seller_item(t, seller_id, TVflag)   # 获取推荐候选集和推荐产品
    for item in a_item:
        if len(verification[verification['item_id'] == item]) == 0:   # 未交互行为的建模
            popular_count = test_data['item_id'].value_counts().reset_index()  # 统计各个产品被交互次数
            popular_items = list(popular_count['index'][popular_count['item_id'] > 20].values)# 热门产品的ID列表
            if item in popular_items:  # 热门产品却没有交互，负反馈
                W = W + 1
        else:   # 有交互
            flag = 1
            Hit.append(item)
            if TVflag == 0:
                S += 1
                continue
            interact_data = verification[verification['item_id'] == item]['action_type'].value_counts().to_frame().reset_index()
            interact_data.rename(columns={'index': 'action_type', 'action_type': 'number'}, inplace=True)
            if len(interact_data[interact_data['action_type'] == 0]) != 0:  # 点击了推荐产品但未购买
                a1 = interact_data[interact_data['action_type'] == 0]['number'].values[0]  # 点击次数
                if a1 > 1:    # 多次点击
                    S += 1
                else:  # 单次点击
                    W += 1
            if len(interact_data[interact_data['action_type'] == 3]) != 0:   # 收藏
                S += 2
            if len(interact_data[interact_data['action_type'] == 1]) != 0:   # 加入了购物车
                S += 2
            if len(interact_data[interact_data['action_type'] == 2]) != 0:   # 购买
                S += 3
    Hit = list(set(Hit))
    return flag, Hit, S, W, a_item     # 成功为r_ta,失败为0


def get_brand_reward(t, brand_id, user_data, TVflag):  # 判断选择当前品牌否有奖励
    flag = 0
    S = 0   # 强偏好
    W = 0   # 弱偏好
    a_item = []
    Hit = []
    user_data.reset_index(drop=True, inplace=True)  # 未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id'] == t.item_id)&(user_data['action_type'] == t.action_type)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    # 验证数据   
    a_recommend, a_item = get_brand_item(t, brand_id, TVflag)   # 获取推荐候选集和推荐产品
    for item in a_item:
        if len(verification[verification['item_id']==item]) == 0:   # 未交互行为的建模
            popular_count = test_data['item_id'].value_counts().reset_index()  # 统计各个产品被交互次数
            popular_items = list(popular_count['index'][popular_count['item_id'] > 20].values)# 热门产品的ID列表
            if item in popular_items:  # 热门产品却没有交互，负反馈
                W = W + 1
        else:   # 有交互
            flag = 1
            Hit.append(item)
            if TVflag == 0:
                S += 1
                continue
            interact_data = verification[verification['item_id'] == item]['action_type'].value_counts().to_frame().reset_index()
            interact_data.rename(columns={'index': 'action_type', 'action_type': 'number'}, inplace=True)
            if len(interact_data[interact_data['action_type'] == 0]) != 0:  # 点击了推荐产品但未购买
                a1 = interact_data[interact_data['action_type'] == 0]['number'].values[0]  # 点击次数
                if a1 > 1:  # 多次点击
                    S += 1
                else:  # 单次点击
                    W += 1
            if len(interact_data[interact_data['action_type'] == 3]) != 0:   # 收藏
                S += 2
            if len(interact_data[interact_data['action_type'] == 1]) != 0:   # 加入了购物车
                S += 2
            if len(interact_data[interact_data['action_type'] == 2]) != 0:   # 购买
                S += 3
    Hit = list(set(Hit))
    return flag, Hit, S, W, a_item     # 成功为r_ta,失败为0



def get_cat_reward(t,cat_id,user_data, TVflag):  # 判断选择当前类别否有奖励
    flag = 0
    S = 0   # 强偏好
    W = 0   # 弱偏好
    a_item = []
    Hit = []
    user_data.reset_index(drop=True,inplace=True)#未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id']==t.item_id)&(user_data['action_type']==t.action_type)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    # 验证数据   
    a_recommend,a_item = get_cat_item(t, cat_id, TVflag)   # 获取推荐候选集和推荐产品
    for item in a_item:
        if len(verification[verification['item_id']==item])==0:   # 未交互行为的建模
            popular_count = test_data['item_id'].value_counts().reset_index()  # 统计各个产品被交互次数
            popular_items = list(popular_count['index'][popular_count['item_id']>20].values)# 热门产品的ID列表
            if item in popular_items:  # 热门产品却没有交互，负反馈
                W = W + 1
        else: # 有交互
            flag = 1
            Hit.append(item)
            if TVflag == 0:
                S += 1
                continue
            interact_data = verification[verification['item_id']==item]['action_type'].value_counts().to_frame().reset_index()
            interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
            if len(interact_data[interact_data['action_type']==0]) != 0:  # 点击了推荐产品但未购买
                a1 = interact_data[interact_data['action_type']==0]['number'].values[0]  # 点击次数
                if a1 > 1:#多次点击
                    S += 1
                else:  # 单次点击
                    W += 1
            if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
                S += 2
            if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
                S += 2
            if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
                S += 3
    Hit = list(set(Hit))
    return flag, Hit, S, W, a_item     # 成功为r_ta,失败为0