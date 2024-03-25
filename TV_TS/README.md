### tensorflow 1.15
### 直接运行TV_TS.py文件即可
### 其中
11行：超参，为论文中的kesi值（一般取0-10）
> lamada = 3.0
> 
156-159行：训练部分，可将训练集初始化的奖励分布存入csv文件中

~~~
# 建臂 得到空的奖励分布 N(0,1)
theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat = deltaT_gaussian_init_arms.initial_values(seller_nums, brand_nums, cat_nums, user_id)
# 训练集初始化臂的奖励分布，存入csv文件中
deltaT_gaussian_init_arms.init_train_reward(user_id, seller_list, brand_list, cat_list, theta_seller, var_seller, deltaT_seller, theta_brand, var_brand, deltaT_brand, theta_cat, var_cat, deltaT_cat)
~~~

