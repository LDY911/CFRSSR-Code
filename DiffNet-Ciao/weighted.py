import pickle
import pandas as pd
import numpy as np
import sys
import time

dataset_name = 'Ciao'
thresh = '3, 7'
improvement_type = '2'

data_influenced_ui = pd.read_csv('./'+dataset_name+'_'+thresh+'_influenced_inter_data_j.csv')
data_inter = pd.read_csv('./dataset/'+dataset_name+'/'+dataset_name+'.inter', sep='\t')
data_inter = data_inter.rename(columns={'user_id:token': 'user_id', 'item_id:token': 'item_id'})
data_net = pd.read_csv('./dataset/'+dataset_name+'/'+dataset_name+'.net', sep='\t')
data_net = data_net.rename(columns={'source_id:token': 'friend_id', 'target_id:token': 'user_id'})

user_temp1 = np.array(pd.read_csv('./dataset/'+dataset_name+'/'+dataset_name+'.inter', sep='\t')[['user_id:token']].drop_duplicates()).flatten().tolist()
user_temp2 = np.array(pd.read_csv('./dataset/'+dataset_name+'/'+dataset_name+'.net', sep='\t')[['source_id:token']].drop_duplicates()).flatten().tolist()
user_temp3 = np.array(pd.read_csv('./dataset/'+dataset_name+'/'+dataset_name+'.net', sep='\t')[['target_id:token']].drop_duplicates()).flatten().tolist()
user_num = len(set(user_temp1 + user_temp2 + user_temp3))
user_base = pd.DataFrame({'user_id': range(1, user_num+1)})

user_inter_num = data_inter.groupby('user_id').count()['item_id'].reset_index()
user_inter_num = user_inter_num.rename(columns={'item_id': 'inter_num'})
user_friend_num = data_net.groupby('user_id').count()['friend_id'].reset_index()
user_friend_num = user_friend_num.rename(columns={'friend_id': 'friend_num'})
user_influenced_item_num = data_influenced_ui.groupby('user_id').count()['item_id'].reset_index()
user_influenced_item_num = user_influenced_item_num.rename(columns={'item_id': 'influenced_item_num'})
user_result = pd.merge(user_base, user_influenced_item_num, on=['user_id'], how='left')
user_result = pd.merge(user_result, user_friend_num, on=['user_id'], how='left')
user_result = pd.merge(user_result, user_inter_num, on=['user_id'], how='left')
user_result = user_result.fillna(0.1)

user_result['type_0'] = user_result['influenced_item_num']
user_result['type_0'] = (user_result['type_0'] - min(user_result['type_0'])) / (max(user_result['type_0']) - min(user_result['type_0']))
user_result['type_1'] = user_result['influenced_item_num'] / user_result['friend_num']
user_result['type_1'] = (user_result['type_1'] - min(user_result['type_1'])) / (max(user_result['type_1']) - min(user_result['type_1']))
user_result['type_2'] = user_result['influenced_item_num'] / user_result['inter_num']
user_result['type_2'] = (user_result['type_2'] - min(user_result['type_2'])) / (max(user_result['type_2']) - min(user_result['type_2']))
user_result['type_3'] = (user_result['type_1'] + user_result['type_2']) /2
del user_result['influenced_item_num']
del user_result['friend_num']
del user_result['inter_num']

user_result.loc[-1] = [0] * len(user_result.columns)
user_result.index = user_result.index + 1
user_result = user_result.sort_index()
user_result = np.array(user_result)

if improvement_type == '0':
    pickle.dump(user_result[:, 1], open('./dataset/'+dataset_name +'/'+dataset_name + '_weighted_j.pkl', 'wb'))
elif improvement_type == '1':
    pickle.dump(user_result[:, 2], open('./dataset/'+dataset_name +'/'+dataset_name + '_weighted_j.pkl', 'wb'))
elif improvement_type == '2':
    pickle.dump(user_result[:, 3], open('./dataset/'+dataset_name +'/'+dataset_name + '_weighted_j.pkl', 'wb'))
elif improvement_type == '3':
    pickle.dump(user_result[:, 4], open('./dataset/'+dataset_name +'/'+dataset_name + '_weighted_j.pkl', 'wb'))

print('Weighted, parameters:', thresh, improvement_type)
