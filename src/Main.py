# -*- coding: utf-8 -*-

"""
先获取队伍状态表，队伍对手状态表，混合状态表，将三个表去掉不必要的列，按照队名对其，合并在一起

然后循环内当年的所有赛事结果，根据结果和主客场作战去修改此队伍的elo值，这个elo和其队伍的各项指标构成了其特征值，
每场赛事都会形成一个数组，循环完毕之后会形成一个二维数组,

将这个二维数组装入逻辑回归模型内，然后进行交叉验证，

然后获取将来的赛事安排，遍历所有赛事，每次获取主客队的elo分数和统计数据，形成特征值，
然后通过这个模型去判断胜负率，
"""


import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import glob
import os



# 当每支队伍没有elo等级分时，赋予其基础elo等级分
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
# 存放数据的目录
folder = '../data'


# 根据每支队伍的Miscellaneous Opponent，Team统计数据csv文件进行初始化
def initialize_data():
    # 先读取三个表，获取到每个队的表现数据
    # 移除不需要的列，将三个表，左对齐，合并起来

    first = 1
    for csvFile in glob.glob(folder + "/Miscellaneous_Stat*.csv"):
        new_Mstat=pd.read_csv(csvFile).drop(['Rk', 'Arena'], axis=1)
        if first == 1:
            old_team_stats = new_Mstat
        else:
            old_team_stats = pd.merge(old_team_stats, new_Mstat, how='left', on='Team')
        first=0
        #Mstat_array.append(new_Mstat)

    for csvFile in glob.glob(folder + "/Opponent_Per_Game_Stat*.csv"):
        new_Ostat=pd.read_csv(csvFile).drop(['Rk', 'G', 'MP'], axis=1)
        old_team_stats = pd.merge(old_team_stats, new_Ostat, how='left', on='Team')
        #Ostat_array.append()

    for csvFile in glob.glob(folder + "/Team_Per_Game_Stat*.csv"):
        new_Tstat=pd.read_csv(csvFile).drop(['Rk', 'G', 'MP'], axis=1)
        old_team_stats = pd.merge(old_team_stats, new_Tstat, how='left', on='Team')
        # Tstat_array.append()

    return old_team_stats.set_index('Team', inplace=False, drop=True)


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
    # global X,y
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
            # X=np.append(X, win_team_features + lose_team_features)
            y.append(0)
        else:
            X.append(lose_team_features + win_team_features)
            # X=np.append(X, lose_team_features + win_team_features)
            y.append(1)

        # 根据这场比赛的数据，   计算更新队伍的elo值
        new_winner_elo, new_loser_elo = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_elo
        team_elos[Lteam] = new_loser_elo

    # return , y


# 预测胜利者
def predict_winner(visterTeam, hostTeam, model):
    features = []

    # team 1，客场队伍，获取其处理之后的elo，拼接其特征值
    features.append(get_elo(visterTeam))
    for key, value in team_stats.loc[visterTeam].iteritems():
        features.append(value)

    # team 2，主场队伍，获取其处理之后的elo，拼接其特征值
    features.append(get_elo(hostTeam) + 100)
    for key, value in team_stats.loc[hostTeam].iteritems():
        features.append(value)

    #处理空数据，NaN填0
    features = np.nan_to_num(features)
    #预测可能性
    return model.predict_proba([features])


if __name__ == '__main__':
    # # 然后合并数据在一起
    team_stats = initialize_data()

    for csvFile in glob.glob(folder + "/Game_Result*.csv"):
        # 读取比赛结果表
        result_data = pd.read_csv(csvFile)
        # 构建结果集合
        build_dataSet(result_data)

    X=np.nan_to_num(X)
    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))
    #初始化一个逻辑回归模型
    model = linear_model.LogisticRegression()
    # 填入数据
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    #进行交叉验证
    print(cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1).mean())

    # 利用训练好的model在16-17年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(folder + '/Future_Schedule.csv')
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
            probability=prob
            # result.append([winner, loser, prob])
        else:
            # 如果可能性不超过50%
            winner = hostTeam
            loser = visterTeam
            probability = 1-prob
            # result.append([winner, loser, (1 - prob)])

        schedule1617.set_value(index,"win",index)
        schedule1617.set_value(index, "lose", index)
        schedule1617.set_value(index, "probability", index)

    # with open(folder + '/16-17Result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['win', 'lose', 'probability'])
    #     writer.writerows(result)
    #     print('done.')

    # pd.read_csv(folder+'16-17Result.csv',header=0)
