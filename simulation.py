#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/22 20:42
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : simulation.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import math


# set rules to change personal wealth amount
# No.1 personal income
# According to the Current Population Survey(CPS) conducted by US Census Bureau,
dataset = pd.read_table('personal_income_2019')
'''
# 大概还原real data
n = dataset.count()
income_points = []
for i in range(n[0]):
    income = [dataset.iloc[i,0]]*dataset.iloc[i,1]
    income_points += income
# try to plot and fit data to certain distribution
sns.distplot(income_points)
'''


# generate a list of personal income values based on historical distribution
def StableIncomeSample(size):
    ''' Combine 3 distributions in piecewise fashion to generate random sample
    :param size: number of people
    :return: sample: a list of personal income values
    '''
    sample_left = [0]*math.ceil(0.286 * size)
    sample_middle = list(np.random.triangular(0, 1250, 100000, math.ceil(0.643 * size)))
    sample_right = list(np.random.choice([100000, 10000000000], math.floor(0.071 * size)))
    sample = sample_left + sample_middle + sample_right
    return sample

def randomUnstableIncome(person_n,chance):
    '''
    :param size:
    :return:
    '''
    id_choice0 = list(np.random.choice(person_n, math.ceil(chance * len(person_n))))
    # unstable income like gains from lottery
    # assume the probability of winning $1 is 1/100
    unstable_df = pd.DataFrame({'income_unstable': [5] * len(id_choice0)}, index=id_choice0)
    return unstable_df



def simulation(person_n,fortune,round_no):
    ''' Simulate the gain and loss of personal wealth in
    :param fortune: the dataframe that contains initial personal wealth value of the whole people
    :param round_no: the number of simulation rounds
    :return: fortune: a column of new fortune value after one round of simulation
    '''
    # if there're people with zero wealth,
    # if len(fortune[fortune[round_no - 1]]==0) > 0:
    round_i = pd.DataFrame({'pre_round_wealth':fortune[round_no - 1],'income_stable':StableIncomeSample(len(person_n)),
                            'income_tax':0},index=person_n)
    round_i = pd.merge(round_i, randomUnstableIncome(person_n,1/100), how='left', left_index=True, right_index=True)
    round_i.fillna(0,inplace=True)
    # personal income tax rate
    round_i['income_tax'] = round_i['income_stable'].apply(lambda x: 0.1 * x if x <= 9700 else (
        0.1 * 9700 + 0.12 * (x - 9700) if x <= 39475 else (
            0.1 * 9700 + 0.12 * 30045 + 0.22 * (x - 39745) if x <= 84200 else (
                0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * (x - 82400) if x <= 160725 else (
                    0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * (x - 160725) if x <= 204100 else (
                        0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * 43375 + 0.35 * (
                                    x - 204100) if x <= 510300 else
                        0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * 43375 + 0.35 * 306200 + 0.37 * (
                                    x - 510300)))))))

    # social welfare: unemployment benefits, assume $400/week for at 6 weeks, isn't taxable
    # The standard time-length of unemployment compensation is six months
    round_i['social_welfare'] = round_i['income_stable'].apply(lambda x: 2400 if x == 0 else 0)

    # accidental_loss
    id_choice1 = list(np.random.choice(person_n, math.ceil((1 / 50) * len(person_n))))
    temp1 = pd.DataFrame({'accidental_loss': [500] * len(id_choice1)}, index=id_choice1)
    round_i = pd.merge(round_i, temp1, how='left', left_index=True, right_index=True)
    round_i.fillna(0,inplace=True)

    # a column of fortune value after one round
    fortune_i = round_i['pre_round_wealth'] + round_i['income_stable'] + round_i['income_unstable'] - \
                round_i['income_tax'] + round_i['social_welfare'] - round_i['accidental_loss']
    #fortune_i.astype('Int64')
    return fortune_i


# set initial values of personal wealth with uniform distribution
# assume there's almost no difference among people in one country at first
person_n = [x for x in range(1, 1001)]
fortune = pd.DataFrame([0 for i in range(1000)], index=person_n)
fortune.index.name = 'ID'

for round in range(1,100):
    fortune[round] = simulation(person_n,fortune,round)

result1 = fortune.T


