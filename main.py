import random
import numpy as np
from numpy import matlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src1 import matrix
from src1 import input_update

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

#分层训练多个网络，每一个层训练一个网络
def net_train(input, target, n, industry_state):
    net_list = []
    temp = 0
    for i in range(n):
        net = Net()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        x = industry_state[i][0]
        for t in range(200):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(input[temp:temp + x])
            loss = criterion(output, target[temp:temp + x])
            loss.backward()
            optimizer.step()
        temp = temp + x
        net_list.append(net)
    return net_list


# ————————————————————————————初始化，但是包含一定的计算过程————————————————————————————————————————————
n = 3
industry_state = []
industry_temp = []
industry_action = []
industry_reward = []
raw_material_price = random.uniform(0.1, 0.2)
for i in range(n):
    # state
    num = random.randint(20, 30)
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
        labor.append(random.randint(10, 15))
        wage.append(random.uniform(0.5, 0.5))
        interest_rate.append(random.uniform(2, 2))
        product_alpha = random.uniform(0.6, 0.6)
        capital.append(product_alpha / (1 - product_alpha) * wage[-1] / interest_rate[-1] * labor[-1])
        # capital.append(random.uniform(1,100))
        inter_alpha.append(random.uniform(1.1, 1.1))
        TFP.append(random.uniform(1.1, 1.5))
        price.append(random.uniform(1, 100)) # 根据src或许可以设成(0,8)随机
        prod_alpha.append(product_alpha)
        pro = TFP[j] * (capital[j] ** product_alpha) * (labor[j] ** (1 - product_alpha))
        if i == 0:
            product_plan.append(np.min([random.uniform(50, 50), pro])) # 同理 范围应该和src对齐
            purc_amount.append(product_plan[-1] / inter_alpha[-1])
            sell_amount.append(product_plan[-1])
        else:
            product_plan.append(np.min([random.uniform(50, 50), pro]))
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
result = list()
for i in range(n - 1):
    result_matrix = matrix(industry_state[i][0], industry_state[i + 1][0],
                           industry_action[i][0], industry_temp[i + 1][0], industry_temp[i][1])
    sell = []
    for j in range(industry_state[i + 1][0]):
        amount = 0
        for k in range(industry_state[i][0]):
            amount += result_matrix[j][k]
        sell.append(np.min([industry_action[i + 1][1][j], amount * industry_state[i + 1][3][j]]))
    sell = np.array(sell)
    industry_temp[i + 1][1] = sell
    result.append(result_matrix)
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
i = 0
while (consumption > 0) & (i < (current_firm_num - 1)):
    settle_matrix[id_ascent[i]] = industry_temp[layer - 1][1][id_ascent[i]]
    # print(consumption)
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
            profit_matrix[i]=post_settle_profit[i]
        else:
            if pre_settle_profit[i] > 0:
                sell_matrix[i, :] = pre_settle_profit[i] * (
                        result_matrix[i, :] / np.sum(buy_price * result_matrix[i, :]))
                post_matrix[i]=post_settle_profit[i]
            else:
                profit_matrix[i]= pre_settle_profit[i]
                post_matrix[i]=-intermediary_cost.sum(axis=1)[i]
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
            reward.append(profit[i][k] + 10)
        else:
            if (industry_action[j][0][k] == 0) | (industry_action[j][1][k] == 0):
                reward.append(profit[i][k] - 100+post_profit[i][k])
            else:
                reward.append(profit[i][k]+post_profit[i][k])
    industry_reward[j] = reward

#————————————————————————————————————————————————定义神经网络——————————————————————————————————————————————
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = torch.relu(self.fc1(x))
        layer_norm = nn.LayerNorm(128)
        x = layer_norm(x)
        x = torch.relu(self.fc2(x))
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
net = Net()

input_list = []
target_list = []
for i in range(n):
    for j in range(industry_state[i][0]):
        input_temp = [industry_state[i][1][j], industry_state[i][2][j], industry_state[i][3][j],
                      industry_state[i][4][j], industry_state[i][5][j], industry_state[i][6][j],
                      industry_state[i][7][j],
                      industry_action[i][0][j], industry_action[i][1][j]]

        input_list.append(input_temp)
        target_list.append([industry_reward[i][j]])

input_numpy = np.array(input_list)
input = torch.from_numpy(input_numpy)
target_numpy = np.array(target_list)
target = torch.from_numpy(target_numpy)
net_result = net_train(input, target, n, industry_state)

for tick in range(1):
    x = input_update(net_result, input_numpy, industry_state, n)
    for i in range(n):
        for j in range(x[0][i][0]):
            input_temp = [x[0][i][1][j], x[0][i][2][j], x[0][i][3][j],
                          x[0][i][4][j], x[0][i][5][j], x[0][i][6][j],
                          x[0][i][7][j],
                          x[1][i][0][j], x[1][i][1][j]]
            input_list.append(input_temp)
            target_list.append([industry_reward[i][j]])
    input_numpy = np.array(input_list)
    input = torch.from_numpy(input_numpy)
    target_numpy = np.array(target_list)
    target = torch.from_numpy(target_numpy)
    net_result = net_train(input, target, n, industry_state)
    print(x[3])
    print(industry_reward)
    #print(tick)
    #print(x[1][2])

