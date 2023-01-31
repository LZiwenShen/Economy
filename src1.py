import random
import numpy as np
from numpy import matlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#输出下层对上层的匹配结果
def matrix(firm_num, purc_num, firm_price, purc_amount, sold_amount):
    # fp_matrix = np.random.randint(0, 3, size=(firm_num, purc_num))
    deal_matrix = np.matlib.zeros((firm_num, purc_num))
    deal_matrix = np.array(deal_matrix)

    firm_sort = np.argsort(firm_price)
    purc_sort = np.argsort(-purc_amount)

    for j in range(firm_num):
        i = firm_sort[j]
        if sold_amount[i] == 0:
            continue
        k = 0
        while k < purc_num:
            h = purc_sort[k]
            if purc_amount[h] != 0:
                if purc_amount[h] >= sold_amount[i]:
                    deal_matrix[i][h] = np.around(sold_amount[i], 2)
                    purc_amount[h] -= sold_amount[i]
                    sold_amount[i] = 0
                else:
                    deal_matrix[i][h] = np.around(purc_amount[h], 2)
                    sold_amount[i] -= purc_amount[h]
                    purc_amount[h] = 0
                purc_sort = np.argsort(-purc_amount)
                if sold_amount[i] == 0:
                    break
            else:
                k += 1

    return deal_matrix.T
#根据训练的net神经网络，推导最优的action行为,input_arr是预测依据的状态。
#返回添加了动作的完整序列
def optimize_action(net, input_arr):
    result = np.zeros([len(input_arr), 9])
    for t in range(0, len(input_arr)):
        res = list()
        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        for i in range(101):
            temp = np.append(input_arr[t], x[i])
            for j in range(101):
                res.append(np.append(temp, y[j]))
        print(res)
        grid = np.array(res)
        test = torch.from_numpy(grid)
        rew = net(test)
        ind = np.argmax(rew.detach().numpy())
        result[t, :] = test[ind].numpy()
    return result

#net_result对系统中每个层次进行训练，分别形成一个网络,input_array是网络的输入序列，industry_state是原始状态，n是系统层数
#返回下一个循环的state，action，temp和reward
def input_update(net_result, input_array, industry_state, n):
    update_state = []
    update_action = []
    update_temp = []
    update_reward = []
    tmp = 0
    raw_material_price = random.uniform(0.1, 0.2)
    for i in range(n):
        net_tmp = net_result[i]
        x = industry_state[i][0]
        state_action = optimize_action(net_tmp, input_array[tmp:tmp+x,0:7])
        update_state.append([industry_state[i][0],state_action[:, 0],state_action[:, 1],state_action[:, 2],state_action[:, 3],state_action[:, 4],state_action[:, 5],state_action[:, 6]])
        update_action.append([state_action[:,7],state_action[:,8]])
        tmp = tmp + x
    for i in range(n):
        # temp
        purc_amount = []
        sell_amount = []
        pro_upper = []
        # reward
        reward = []

        for j in range(update_state[i][0]):
            pro = update_state[i][4][j]*(update_state[i][2][j]**update_state[i][5][j])*(update_state[i][1][j]**(1-update_state[i][5][j]))
            if i == 0:
                update_action[i][1][j]=np.min([update_action[i][1][j], pro])
                purc_amount.append( update_action[i][1][j]/ update_state[i][3][j])
                sell_amount.append(update_action[i][1][j])
            else:
                update_action[i][1][j]=np.min([update_action[i][1][j], pro])
                purc_amount.append(update_action[i][1][j] / update_state[i][3][j])
                sell_amount.append(0)
            pro_upper.append(pro)
        purc_amount = np.array(purc_amount)
        sell_amount = np.array(sell_amount)
        pro_upper = np.array(pro_upper)
        reward = np.array(reward)
        update_reward.append([reward])
        update_temp.append([purc_amount, sell_amount, pro_upper])
    result = list()
    for i in range(n - 1):
        result_matrix = matrix(update_state[i][0], update_state[i + 1][0],
                               update_action[i][0], update_temp[i + 1][0], update_temp[i][1])
        sell = []
        for j in range(update_state[i + 1][0]):
            amount = 0
            for k in range(update_state[i][0]):
                amount += result_matrix[j][k]
            sell.append(np.min([update_action[i + 1][1][j], amount * update_state[i + 1][3][j]]))
        sell = np.array(sell)
        update_temp[i + 1][1] = sell
        result.append(result_matrix)
    settle = list()
    layer = n
    saving_rate = 0.2
    settle_matrix = np.zeros(np.shape(update_action[layer - 1][0])[0])
    current_firm_num = np.shape(update_action[layer - 1][0])[0]
    id_ascent = np.argsort(update_action[layer - 1][0])
    consumption = 0
    for i in range(0, layer):
        consumption = consumption + (1 - saving_rate) * ((update_state[i][1] * update_state[i][6]).sum())
    consumption_total = consumption
    i = 0
    while (consumption > 0) & (i < (current_firm_num - 1)):
        settle_matrix[id_ascent[i]] = update_temp[layer - 1][1][id_ascent[i]]
        i = i + 1
        consumption = consumption - update_action[layer - 1][0][id_ascent[i]] * update_temp[layer - 1][1][
                id_ascent[i]]
    if consumption < 0:
        settle_matrix[id_ascent[i - 1]] = (consumption + update_action[layer - 1][0][id_ascent[i]] *
                                               update_temp[layer - 1][1][id_ascent[i]]) / \
                                              update_action[layer - 1][0][id_ascent[i]]
    else:
        settle_matrix[id_ascent[i - 1]] = update_temp[layer - 1][1][id_ascent[i]]
    settle.append(settle_matrix)
    k = layer
    profit = list()
    post_profit = list()
    while k > 1:
        settle_matrix = settle[layer - k]
        result_matrix = result[k - 2]
        current_firm_num = update_state[k - 1][0]
        last_firm_num = update_state[k - 2][0]
        buy_price = update_action[k - 2][0]
        sell_price = update_action[k - 1][0]
        wage = update_state[k - 1][6]
        labor = update_state[k - 1][1]
        capital = update_state[k - 1][2]
        interest_rate = update_state[k - 1][7]
        sell_matrix = np.zeros([current_firm_num, last_firm_num])
        profit_matrix = np.zeros(current_firm_num)
        post_matrix = np.zeros(current_firm_num)
        intermediary_cost = np.zeros([current_firm_num, last_firm_num])
        for i in range(0, current_firm_num):
            intermediary_cost[i, :] = buy_price * result_matrix[i, :]
        if len(settle_matrix) != len(sell_price):
            print('成交矩阵和销售价格矩阵维度不符')
            break
        pre_settle_profit = settle_matrix * sell_price - wage * labor
        post_settle_profit = settle_matrix * sell_price - intermediary_cost.sum(axis=1) - wage * labor
        for i in range(0, current_firm_num):
            if post_settle_profit[i] > 0:
                sell_matrix[i, :] = result_matrix[i, :]
                profit_matrix[i] = post_settle_profit[i]
            else:

                if pre_settle_profit[i] > 0:
                    sell_matrix[i, :] = pre_settle_profit[i] * (
                                result_matrix[i, :] / np.sum(buy_price * result_matrix[i, :]))
                    post_matrix[i] = post_settle_profit[i]
                else:
                    profit_matrix[i] = pre_settle_profit[i]
                    post_matrix[i] = -intermediary_cost.sum(axis=1)[i]
                    pass
        settle.append(np.sum(sell_matrix, axis=0))
        profit.append(profit_matrix)
        post_profit.append(post_matrix)
        k = k - 1

    profit.append(settle[-1] * update_action[0][0] - update_temp[0][0] * raw_material_price - update_state[0][1] *
            update_state[0][6])
    post_profit.append(np.zeros(industry_state[0][0]))
    torch.set_default_tensor_type(torch.DoubleTensor)

    for i in range(n):
        j = n - 1 - i
        reward = []
        for k in range(update_state[j][0]):
            if (profit[i][k] > 0):
                reward.append(profit[i][k] + 10)
            else:
                if (update_action[j][0][k] == 0) | (update_action[j][1][k] == 0):
                    reward.append(profit[i][k] - 100+post_profit[i][k])
                else:
                    reward.append(profit[i][k]+post_profit[i][k])
        update_reward[j] = reward
    return [update_state,update_action,update_temp,update_reward]