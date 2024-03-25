import random
import pandas as pd

item_info = pd.read_csv('../IJCAI_15_dataset/item_info_1000.csv')  # 449010
test_data = pd.read_csv('../IJCAI_15_dataset/test_data_1000.csv') # 307104


'''动作遗憾：最佳动作预期奖励和t轮所选动作实际奖励'''

def get_seller_regret(t, seller_id, user_data):
    ''' '''
    R_opt = 0   # 预期奖励
    R_t = 0     # 实际奖励
    user_data.reset_index(drop=True, inplace=True) #某个用户的log 未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id']==t.item_id)&(user_data['action_type']==t.action_type)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    # 验证数据  
    
    if len(verification[verification['seller_id']==seller_id])!=0:   # 最佳动作预期奖励
        interact_data = verification[verification['seller_id']==seller_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={'index':'action_type', 'action_type': 'number'}, inplace=True)
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_opt += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_opt += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_opt += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_opt += 3
            
    if len(verification[verification['seller_id']==t.seller_id])!=0:   # 实际动作奖励
        interact_data = verification[verification['seller_id']==t.seller_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_t += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_t += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_t += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_t += 3
    return abs(R_opt-R_t)


def get_brand_regret(t,brand_id,user_data): 
    ''' '''
    R_opt = 0
    R_t = 0
    user_data.reset_index(drop=True,inplace=True)#未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id']==t.item_id)&(user_data['action_type']==t.action_type)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    # 验证数据  
    
    if len(verification[verification['brand_id']==brand_id])!=0:   # 最佳动作预期奖励
        interact_data = verification[verification['brand_id']==brand_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_opt += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_opt += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_opt += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_opt += 3
            
    if len(verification[verification['brand_id']==t.brand_id])!=0:   # 实际动作奖励
        interact_data = verification[verification['brand_id']==t.brand_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_t += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_t += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_t += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_t += 3
    return abs(R_opt-R_t)


def get_cat_regret(t,cat_id,user_data): 
    ''' '''
    R_opt = 0
    R_t = 0
    user_data.reset_index(drop=True,inplace=True)#未去重
    n = len(user_data)   # 该用户记录总长度
    m = user_data[(user_data['item_id']==t.item_id)&(user_data['action_type']==t.action_type)&(user_data['time_stamp']==t.time_stamp)].index.values[0]
    verification = user_data[m+1:n]    # 验证数据  
    
    if len(verification[verification['cat_id']==cat_id])!=0:   # 最佳动作预期奖励
        interact_data = verification[verification['cat_id']==cat_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_opt += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_opt += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_opt += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_opt += 3
            
    if len(verification[verification['cat_id']==t.cat_id])!=0:   # 实际动作奖励
        interact_data = verification[verification['cat_id']==t.cat_id]['action_type'].value_counts().to_frame().reset_index()
        interact_data.rename(columns={ 'index':'action_type', 'action_type': 'number'}, inplace=True)  
        if len(interact_data[interact_data['action_type']==0])!=0:  # 点击了推荐产品但未购买
            R_t += 1
        if len(interact_data[interact_data['action_type']==3])!=0:   # 收藏
            R_t += 2
        if len(interact_data[interact_data['action_type']==1])!=0:   # 加入了购物车
            R_t += 2
        if len(interact_data[interact_data['action_type']==2])!=0:   # 购买
            R_t += 3
    return abs(R_opt-R_t)
