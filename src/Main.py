# -*- coding: utf-8 -*-

import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# 当每支队伍没有elo等级分时，赋予其基础elo等级分
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
# 存放数据的目录
folder = '../data'


# 根据每支队伍的Miscellaneous Opponent，Team统计数据csv文件进行初始化
def initialize_data(Mstat, Ostat, Tstat):
    # 移除不需要的列
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    # 将三个表，左对齐，合并起来
    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
    return team_stats1.set_index('Team', inplace=False, drop=True)


# 获取每支队伍的Elo Score等级分函数
def get_elo(team):
    try:
        return team_elos[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]


# 计算每个球队的elo值
def calc_elo(win_team, lose_team):
    winner_elo = get_elo(win_team)
    loser_elo = get_elo(lose_team)

    elo_diff = winner_elo - loser_elo
    exp = -elo_diff / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据rank级别修改K值
    if winner_elo < 2100:
        k = 4*8
    elif 2100 <= winner_elo < 2400:
        k = 3*8
    else:
        k = 2*8

    # 更新 elo 数值
    new_winner_elo = round(winner_elo + (k * (1 - odds)))
    new_loser_elo = round(loser_elo + (k * (0 - odds)))
    return new_winner_elo, new_loser_elo


# 构建数据集合
def build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    # 循环所有的比赛结果记录
    for index, row in all_data.iterrows():
        # 获取胜者和败者
        Wteam = row['WTeam']
        Lteam = row['LTeam']

        # 获取最初的elo或是每个队伍最初的elo值
        win_team_elo = get_elo(Wteam)
        lose_team_elo = get_elo(Lteam)

        # 给主场比赛胜利的队伍加上100的elo值，否则给客场队加100 elo值，主场优势？
        if row['WLoc'] == 'H':
            win_team_elo += 100
        else:
            lose_team_elo += 100

        # 胜负二队的特征值，把elo当为评价每个队伍的第一个特征值
        win_team_features = [win_team_elo]
        lose_team_features = [lose_team_elo]

        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key, value in team_stats.loc[Wteam].iteritems():
            win_team_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            lose_team_features.append(value)

        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(win_team_features + lose_team_features)
            y.append(0)
        else:
            X.append(lose_team_features + win_team_features)
            y.append(1)

        # if skip == 0:
        # print('X',X)
        #     skip = 1

        # 根据这场比赛的数据，   计算更新队伍的elo值
        new_winner_elo, new_loser_elo = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_elo
        team_elos[Lteam] = new_loser_elo

    return np.nan_to_num(X), y


# 预测胜利者
def predict_winner(visterTeam, hostTeam, model):
    features = []

    # team 1，客场队伍，拼接其特征值
    features.append(get_elo(visterTeam))
    for key, value in team_stats.loc[visterTeam].iteritems():
        features.append(value)

    # team 2，主场队伍，拼接其特征值
    features.append(get_elo(hostTeam) + 100)
    for key, value in team_stats.loc[hostTeam].iteritems():
        features.append(value)

    #处理空数据，NaN填0
    features = np.nan_to_num(features)
    #预测可能性
    return model.predict_proba([features])


if __name__ == '__main__':
    # 先读取三个表，获取到每个队的表现数据
    Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')
    # 然后合并在一起
    team_stats = initialize_data(Mstat, Ostat, Tstat)

    # 读取比赛结果表
    result_data = pd.read_csv(folder + '/15-16_result.csv')
    # 构建结果集合
    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))
    #初始化一个模型
    model = linear_model.LogisticRegression()
    # 填入数据
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    #依据数据统计，构建了一个模型
    print(cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1).mean())

    # 利用训练好的model在16-17年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
    result = []
    #循环未来所有的赛事安排
    for index, row in schedule1617.iterrows():
        visterTeam = row['Vteam']
        hostTeam = row['Hteam']
        #预测胜利者
        pred = predict_winner(visterTeam, hostTeam, model)
        prob = pred[0][0]
        if prob > 0.5:
            #如果可能性超过50%
            winner = visterTeam
            loser = hostTeam
            result.append([winner, loser, prob])
        else:
            # 如果可能性不超过50%
            winner = hostTeam
            loser = visterTeam
            result.append([winner, loser, (1 - prob)])

    with open(folder + '/16-17Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)
        print('done.')

    # pd.read_csv(folder+'16-17Result.csv',header=0)
