import random
import numpy as np
from numpy import matlib
import torch
import torch.nn as nn
import torch.optim as optim

def matrix(firm_num, purc_num, firm_price, purc_amount, sold_amount, receptive_field):
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
            if purc_amount[h] != 0 and receptive_field[h][i] != 0:
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

#分层训练多个网络，每一个层训练一个网络
'''
def net_train(input, target, n, industry_state):
    net_list = []
    temp = 0
    for i in range(n):
        if i == 0:
            net = Net(10)
        else:
            net = Net(9 + industry_state[i - 1][0])
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        x = industry_state[i][0]
        for t in range(20):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(torch.from_numpy(np.array(input[i])))
            loss = criterion(output, target[temp:temp + x])
            loss.backward()
            optimizer.step()
        temp = temp + x
        net_list.append(net)
    return net_list
'''
'''
def optimize_action(net, input_arr, dim):
    result = np.zeros([len(input_arr), dim])
    for t in range(0, len(input_arr)):
        input_temp = input_arr[t]
        res = list()
        x = np.linspace(0, 100, 101)
        y = np.linspace(0, 100, 101)
        for i in range(101):
            input_temp[7] = x[i]
            for j in range(101):
                input_temp[8] = y[j]
                temp = np.array(input_temp)
                res.append(temp)
        grid = np.array(res)
        test = torch.from_numpy(grid)
        rew = net(test)
        ind = np.argmax(rew.detach().numpy())
        result[t, :] = test[ind].numpy()
    return result
'''
'''
def input_update(net_result, input_array, industry_state, n, receptive, raw_material_price):
    update_state = []
    update_action = []
    update_temp = []
    update_reward = []
    tmp = 0
    for i in range(n):
        net_tmp = net_result[i]
        x = industry_state[i][0]
        if i == 0:
            state_action = optimize_action(net_tmp, input_array[i], 10)
        else:
            state_action = optimize_action(net_tmp, input_array[i], 9 + industry_state[i - 1][0])
        update_state.append([industry_state[i][0], state_action[:, 0], state_action[:, 1], state_action[:, 2],
                             state_action[:, 3], state_action[:, 4], state_action[:, 5], state_action[:, 6]])
        update_action.append([state_action[:, 7], state_action[:, 8]])
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
                update_action[i][1][j] = np.min([update_action[i][1][j], pro])
                purc_amount.append(update_action[i][1][j] / update_state[i][3][j])
                sell_amount.append(update_action[i][1][j])
            else:
                update_action[i][1][j] = np.min([update_action[i][1][j], pro])
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
    update_receptive_price = []
    update_receptive_price.append(np.array([[raw_material_price]] * industry_state[0][0]))
    for i in range(n - 1):
        receptive_field = receptive[i]
        result_matrix = matrix(update_state[i][0], update_state[i + 1][0], update_action[i][0],
                               update_temp[i + 1][0], update_temp[i][1], receptive_field)
        update_receptive_price.append(receptive_field * update_action[i][0])
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
    consumption = consumption / 0.6
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
                reward.append(profit[i][k] + 5)
            else:
                if (update_action[j][0][k] == 0) | (update_action[j][1][k] == 0):
                    reward.append(profit[i][k] - 10+2*post_profit[i][k])
                else:
                    reward.append(profit[i][k]+2*post_profit[i][k])
        update_reward[j] = reward
    return [update_state, update_action, update_temp, update_reward, update_receptive_price]
'''

# ————————————————————————————初始化，但是包含一定的计算过程————————————————————————————————————————————
n = 3
industry_state = []
industry_temp = []
industry_action = []
industry_reward = []
raw_material_price = random.uniform(0.1, 0.2)
for i in range(n):
    # state
    if i == 0:
        num = 1
    else:
        num = random.randint(10, 10)
    labor = []
    capital = []
    inter_alpha = []
    TFP = []
    prod_alpha = []
    wage = []
    interest_rate = []
    # action
    price = []
    product_plan = []
    # temp
    purc_amount = []
    sell_amount = []
    pro_upper = []
    # reward
    reward = []

    for j in range(num):
        if i == 0:
            labor.append(random.randint(100, 150))
        else:
            labor.append(random.randint(10, 15))
        wage.append(random.uniform(1, 1.5))
        interest_rate.append(random.uniform(2, 2))
        product_alpha = random.uniform(0.6, 0.6)
        capital.append(product_alpha / (1 - product_alpha) * wage[-1] / interest_rate[-1] * labor[-1])
        # capital.append(random.uniform(1,100))
        inter_alpha.append(random.uniform(1, 1))
        TFP.append(random.uniform(1.1, 1.5))
        price.append(random.uniform(2, 4)) # 根据src或许可以设成(0,8)随机
        prod_alpha.append(product_alpha)
        pro = TFP[j] * (capital[j] ** product_alpha) * (labor[j] ** (1 - product_alpha))
        if i == 0:
            product_plan.append(np.min([random.uniform(100, 300), pro])) # 同理 范围应该和src对齐
            purc_amount.append(product_plan[-1] / inter_alpha[-1])
            sell_amount.append(product_plan[-1])
        else:
            product_plan.append(np.min([random.uniform(10, 30), pro]))
            purc_amount.append(product_plan[-1] / inter_alpha[-1])
            sell_amount.append(0)
        pro_upper.append(pro)
    labor = np.array(labor)
    capital = np.array(capital)
    inter_alpha = np.array(inter_alpha)
    prod_alpha = np.array(prod_alpha)
    TFP = np.array(TFP)
    wage = np.array(wage)
    interest_rate = np.array(interest_rate)
    price = np.array(price)
    product_plan = np.array(product_plan)
    purc_amount = np.array(purc_amount)
    sell_amount = np.array(sell_amount)
    pro_upper = np.array(pro_upper)
    reward = np.array(reward)

    industry_state.append([num, labor, capital, inter_alpha, TFP, prod_alpha, wage, interest_rate])
    industry_action.append([price, product_plan])
    industry_reward.append([reward])
    industry_temp.append([purc_amount, sell_amount, pro_upper])
i_state = industry_state
i_action = industry_action
result = list()
receptive_price = []
receptive_price.append(np.array([[raw_material_price]] * industry_state[0][0]))
receptive = []
for i in range(n - 1):
    receptive_field = np.random.randint(1, 2, size=(industry_state[i + 1][0], industry_state[i][0]))
    receptive.append(receptive_field)
    result_matrix = matrix(industry_state[i][0], industry_state[i + 1][0],
                           industry_action[i][0], industry_temp[i + 1][0],
                           industry_temp[i][1], receptive_field)
    receptive_price.append(receptive_field * industry_action[i][0])
    sell = []
    for j in range(industry_state[i + 1][0]):
        amount = 0
        for k in range(industry_state[i][0]):
            amount += result_matrix[j][k]
        sell.append(np.min([industry_action[i + 1][1][j], amount * industry_state[i + 1][3][j]]))
    sell = np.array(sell)
    industry_temp[i + 1][1] = sell
    result.append(result_matrix)
'''
# ————————————————————————————给出市场交易结果————————————————————————————————————————————
# settle 指代实际购买情况
# profit 指代利润情况
# reward 指代奖励函数情况
# result 两层之间的交易情况，长度是整体产业链长度-1
settle = list()
layer = n
saving_rate = 0.2
settle_matrix = np.zeros(np.shape(industry_action[layer - 1][0])[0])
current_firm_num = np.shape(industry_action[layer - 1][0])[0]
id_ascent = np.argsort(industry_action[layer - 1][0])
consumption = 0
for i in range(0, layer):
    consumption = consumption + (1 - saving_rate) * ((industry_state[i][1] * industry_state[i][6]).sum())
consumption_total = consumption
consumption = consumption/0.6
i = 0
while (consumption > 0) & (i < (current_firm_num - 1)):
    settle_matrix[id_ascent[i]] = industry_temp[layer - 1][1][id_ascent[i]]
    i = i + 1
    consumption = consumption - industry_action[layer - 1][0][id_ascent[i]] * industry_temp[layer - 1][1][id_ascent[i]]
if consumption < 0:
    settle_matrix[id_ascent[i - 1]] = (consumption + industry_action[layer - 1][0][id_ascent[i]] *
                                       industry_temp[layer - 1][1][id_ascent[i]]) / \
                                      industry_action[layer - 1][0][id_ascent[i]]
else:
    settle_matrix[id_ascent[i - 1]] = industry_temp[layer - 1][1][id_ascent[i]]
settle.append(settle_matrix)
k = layer
profit = list()
post_profit = list()
while k > 1:
    settle_matrix = settle[layer - k]
    result_matrix = result[k - 2]
    current_firm_num = industry_state[k - 1][0]
    last_firm_num = industry_state[k - 2][0]
    buy_price = industry_action[k - 2][0]
    sell_price = industry_action[k - 1][0]
    wage = industry_state[k - 1][6]
    labor = industry_state[k - 1][1]
    capital = industry_state[k - 1][2]
    interest_rate = industry_state[k - 1][7]
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
                post_matrix[i] = - intermediary_cost.sum(axis=1)[i]
                pass
    settle.append(np.sum(sell_matrix, axis=0))
    profit.append(profit_matrix)
    post_profit.append(post_matrix)
    k = k - 1

profit.append(settle[-1] * industry_action[0][0] - industry_temp[0][0] * raw_material_price - industry_state[0][1] *
              industry_state[0][6])
post_profit.append(np.zeros(industry_state[0][0]))
torch.set_default_tensor_type(torch.DoubleTensor)

for i in range(n):
    j = n - 1 - i
    reward = []
    for k in range(industry_state[j][0]):
        if (profit[i][k] > 0):
            reward.append(profit[i][k] + 5)
        else:
            if (industry_action[j][0][k] == 0) | (industry_action[j][1][k] == 0):
                reward.append(profit[i][k] - 10+2*post_profit[i][k])
            else:
                reward.append(profit[i][k]+2*post_profit[i][k])
    industry_reward[j] = reward

#————————————————————————————————————————————————定义神经网络——————————————————————————————————————————————
class Net(nn.Module):

    def __init__(self, dim):
        super(Net, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, self.dim)
        x = torch.tanh(self.fc1(x))
        layer_norm = nn.LayerNorm(128)
        x = layer_norm(x)
        x = torch.tanh(self.fc2(x))
        layer_norm = nn.LayerNorm(32)
        x = layer_norm(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#——————————————————————————————————————主体训练过程————————————————————————————————————————
input = []
target_list = []
for i in range(n):
    input_list = []
    for j in range(industry_state[i][0]):
        input_temp = [industry_state[i][1][j], industry_state[i][2][j], industry_state[i][3][j],
                      industry_state[i][4][j], industry_state[i][5][j], industry_state[i][6][j],
                      industry_state[i][7][j],
                      industry_action[i][0][j], industry_action[i][1][j]] + list(receptive_price[i][j])
        input_list.append(input_temp)
        target_list.append([industry_reward[i][j]])
    input.append(input_list)

target_numpy = np.array(target_list)
target = torch.from_numpy(target_numpy)
net_result = net_train(input, target, n, industry_state)

res = []
for tick in range(1):
    x = input_update(net_result, input, industry_state, n, receptive, raw_material_price)
    input = []
    for i in range(n):
        input_list = []
        for j in range(x[0][i][0]):
            input_temp = [x[0][i][1][j], x[0][i][2][j], x[0][i][3][j],
                          x[0][i][4][j], x[0][i][5][j], x[0][i][6][j],
                          x[0][i][7][j],
                          x[1][i][0][j], x[1][i][1][j]] + list(x[4][i][j])
            input_list.append(input_temp)
            target_list.append([industry_reward[i][j]])
        input.append(input_list)
    target_numpy = np.array(target_list)
    target = torch.from_numpy(target_numpy)
    net_result = net_train(input, target, n, industry_state)
    res.append(x)
    print(tick)
'''