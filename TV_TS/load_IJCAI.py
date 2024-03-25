import pandas as pd

def get_IJCAI():
    train_data = pd.read_csv('../IJCAI_15_dataset/train_data_1000.csv')  # 1306285
    test_data = pd.read_csv('../IJCAI_15_dataset/test_data_1000.csv')  # 147076
    item_info = pd.read_csv('../IJCAI_15_dataset/item_info_1000.csv')  # 278426
    user_info = pd.read_csv('../IJCAI_15_dataset/user_1000.csv')  # 1000
    user_list = user_info['user_id'].value_counts().index.tolist()  # 1000个
    item_list = item_info['item_id'].value_counts().index.tolist()  # 278426
    user_id = user_list
    return train_data, test_data, item_info, user_info, item_list, user_id

def get_item_feature(item_info):
    '''建臂'''
    cat = item_info[['cat_id']]
    cat = cat.drop_duplicates()
    cat_list = cat['cat_id'].tolist()

    brand = item_info[['brand_id']]
    brand = brand.drop_duplicates()
    brand_list = brand['brand_id'].tolist()

    seller = item_info[['seller_id']]
    seller = seller.drop_duplicates()
    seller_list = seller['seller_id'].tolist()

    return cat_list, brand_list, seller_list

def get_TSreward_distribution():
    theta_cat = pd.read_csv('./TS_arm_reward_data/theta_cat.csv').values
    var_cat = pd.read_csv('./TS_arm_reward_data/var_cat.csv').values
    theta_seller = pd.read_csv('./TS_arm_reward_data/theta_seller.csv').values
    var_seller = pd.read_csv('./TS_arm_reward_data/var_seller.csv').values
    theta_brand = pd.read_csv('./TS_arm_reward_data/theta_brand.csv').values
    var_brand = pd.read_csv('./TS_arm_reward_data/var_brand.csv').values
    return theta_cat, var_cat, theta_seller, var_seller, theta_brand, var_brand

def get_deltaT_reward_distribution():
    theta_cat = pd.read_csv('./TVTS_arm_reward_data/theta_cat.csv').values
    var_cat = pd.read_csv('./TVTS_arm_reward_data/var_cat.csv').values
    theta_seller = pd.read_csv('./TVTS_arm_reward_data/theta_seller.csv').values
    var_seller = pd.read_csv('./TVTS_arm_reward_data/var_seller.csv').values
    theta_brand = pd.read_csv('./TVTS_arm_reward_data/theta_brand.csv').values
    var_brand = pd.read_csv('./TVTS_arm_reward_data/var_brand.csv').values

    deltaT_seller = pd.read_csv('./TVTS_arm_reward_data/deltaT_seller.csv').values
    deltaT_brand = pd.read_csv('./TVTS_arm_reward_data/deltaT_brand.csv').values
    deltaT_cat = pd.read_csv('./TVTS_arm_reward_data/deltaT_cat.csv').values

    return theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat

def get_user_reward_weight():
    user_reward_weight = pd.read_csv('./TVTS_arm_reward_data/user_reward_weight.csv').values
    return user_reward_weight