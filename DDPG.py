import torch
import numpy as np
import torch.nn as nn
from initialization import i_state, raw_material_price, matrix, receptive, x

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound)

        # layer
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.sigmoid(self.output(a))
        # 对action进行放缩，实际上a in [0,1]
        scaled_a = a * self.action_bound
        return scaled_a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=1000, gamma=0.9, lr_a=0.001,
                 lr_c=0.002, batch_size=32):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()

        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1


def excute_action(industry_state, state, action, layer):
    update_state = industry_state
    update_action = []
    for i in range(layer):
        a = action[i]
        update_action.append([[a[k][j] for k in range(industry_state[i][0])] for j in range(2)])
    update_temp = []
    update_reward = []
    for i in range(layer):
        # temp
        purc_amount = []
        sell_amount = []
        pro_upper = []
        # reward
        reward = []

        for j in range(update_state[i][0]):
            pro = update_state[i][4][j] * (update_state[i][2][j] ** update_state[i][5][j]) * (
                        update_state[i][1][j] ** (1 - update_state[i][5][j]))
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
    for i in range(layer - 1):
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
    saving_rate = 0.2
    settle_matrix = np.zeros(np.shape(update_action[layer - 1][0])[0])
    current_firm_num = np.shape(update_action[layer - 1][0])[0]
    id_ascent = np.argsort(update_action[layer - 1][0])
    consumption = 0
    for i in range(0, layer):
        consumption = consumption + (1 - saving_rate) * ((update_state[i][1] * update_state[i][6]).sum())
    consumption_total = consumption
    consumption = consumption / 0.01
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
                    sell_matrix[i, :] = 0.5*pre_settle_profit[i] * (
                            result_matrix[i, :] / np.sum(buy_price * result_matrix[i, :]))
                    profit_matrix[i]=0.5*pre_settle_profit[i]-0.5*intermediary_cost.sum(axis=1)[i]
                    post_matrix[i] = post_settle_profit[i]
                else:
                    profit_matrix[i] = pre_settle_profit[i]-intermediary_cost.sum(axis=1)[i]
                    post_matrix[i] = -intermediary_cost.sum(axis=1)[i]
                    pass
        settle.append(np.sum(sell_matrix, axis=0))
        profit.append(profit_matrix)
        post_profit.append(post_matrix)
        k = k - 1

    profit.append(settle[-1] * update_action[0][0] - update_temp[0][0] * raw_material_price - update_state[0][1] *
                  update_state[0][6])
    post_profit.append(np.zeros(industry_state[0][0]))
    torch.set_default_tensor_type(torch.FloatTensor)

    for i in range(layer):
        j = layer - 1 - i
        reward = []
        for k in range(update_state[j][0]):
            if (profit[i][k] > 0):
                reward.append(profit[i][k] + 5)
            else:
                if (update_action[j][0][k] == 0) | (update_action[j][1][k] == 0):
                    reward.append(profit[i][k] - 10 + 2 * post_profit[i][k])
                else:
                    reward.append(profit[i][k] + 2 * post_profit[i][k])
        update_reward[j] = reward

    reward = update_reward
    next_state = [[], [], []]
    for i in range(layer):
        for j in range(industry_state[i][0]):
            competitor_price = []
            for k in range(industry_state[i][0]):
                if k != j:
                    competitor_price.append(action[i][k][0])
            next_state[i].append(state[i][j][0:7] + competitor_price)
    return reward, next_state, profit


if __name__ == '__main__':

    # hyper parameters
    MAX_EPISODES = 200
    MAX_EP_STEPS = 3
    MEMORY_CAPACITY = 1000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies
    layer = 3
    action_dim = 2
    for e in range(MAX_EPISODES):
        ddpg = []
        state = [[], [], []]
        start_action = [[], [], []]
        start_reward = [[], [], []]
        for i in range(layer):
            state_dim = 6 + i_state[i][0]  # 本身的state维度(7)+观测到的同层其他企业的定价
            ddpg.append(DDPG(state_dim, action_dim, action_bound=[10, 50],
                        replacement=REPLACEMENT, memory_capacity=MEMORY_CAPACITY))
            for j in range(i_state[i][0]):
                competitor_price = list(x[1][i][0])
                del competitor_price[j]
                state[i].append([i_state[i][1][j], i_state[i][2][j], i_state[i][3][j],
                                 i_state[i][4][j], i_state[i][5][j], i_state[i][6][j],
                                 i_state[i][7][j]] + competitor_price)
                start_action[i].append([x[1][i][0][j], x[1][i][1][j]])
                start_reward[i].append(x[3][i][j])
                ddpg[i].store_transition(state[i][j], start_action[i][j], start_reward[i][j], state[i][j])
        for t in range(1, MAX_EP_STEPS + 1):
            action = [[], [], []]
            for i in range(layer):
                for j in range(i_state[i][0]):
                    action[i].append(ddpg[i].choose_action(state[i][j]))
                    # 执行action 进行企业产量的更新
            reward, next_state = excute_action(i_state, state, action, layer)
            for i in range(layer - 1, -1, -1):
                for j in range(i_state[i][0]):
                    ddpg[i].store_transition(state[i][j], action[i][j], reward[i][j], next_state[i][j])
                    ddpg[i].learn()
            print('T:', t)
        print('episode:', e)