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

'''
# set rules to change personal wealth amount
# No.1 personal income
# According to the Current Population Survey(CPS) conducted by US Census Bureau,
dataset = pd.read_table('personal_income_2019')

# explore the distribution of  real data
n = dataset.count()
income_points = []
for i in range(n[0]):
    income = [dataset.iloc[i,0]]*dataset.iloc[i,1]
    income_points += income
# try to plot and fit data to certain distribution
sns.distplot(income_points)
'''


# generate a list of personal income values based on historical distribution
def stable_income_sample(size):
    ''' Combine 3 distributions in piecewise fashion to generate random sample with income values
    :param size: number of people
    :return: a ndarray of personal income values
    '''
    sample_left = np.ndarray([0]*math.ceil(0.286 * size))
    sample_middle = np.random.triangular(0, 1250, 100000, math.ceil(0.643 * size))
    sample_right = np.random.choice([100000, 10000000000], math.floor(0.071 * size))
    sample = np.append(sample_left,sample_middle,sample_right)
    return np.random.shuffle(sample)


def random_unstable_income(people,chance,amount):
    ''' Given a group of people, the chance of winning a certain amount of unstable income,
    get a column of unstable income for each person.
    :param people: a list of ID numbers which represents a group of people
    :param chance: the chance of earning unstable income
    :param amount: the amount of unstable income
    :return: unstable_df: a dataframe that contains unstable income values for each person
    '''
    id_choice0 = list(np.random.choice(people, math.ceil(chance * len(people))))
    # unstable income like gains from lottery
    # assume the probability of winning $5 is 1/100
    unstable_df = pd.DataFrame({'income_unstable': [amount] * len(id_choice0)}, index=id_choice0)
    return unstable_df


def simulation(people,fortune_df,round_no):
    ''' Simulate the gain and loss of personal wealth in
    :param people: a list of ID numbers which represents a group of people
    :param fortune_df: the dataframe that contains initial personal wealth value of the whole people
    :param round_no: the number of simulation rounds
    :return: fortune_i: a column of new fortune value after one round of simulation
    '''
    round_i = pd.DataFrame({'pre_round_wealth':fortune_df[round_no - 1],'income_stable':stable_income_sample(len(people)),
                            'income_tax':0})
    round_i = pd.merge(round_i, random_unstable_income(people,1/100), how='left', left_index=True, right_index=True)
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

    # social welfare: unemployment benefits, assume $400/week for at most 6 weeks, isn't taxable
    # The standard time-length of unemployment compensation is six months
    round_i['social_welfare'] = round_i['income_stable'].apply(lambda x: 2400 if x == 0 else 0)

    # accidental_loss
    id_choice1 = list(np.random.choice(people, math.ceil((1 / 50) * len(people))))
    temp1 = pd.DataFrame({'accidental_loss': [500] * len(id_choice1)}, index=id_choice1)
    round_i = pd.merge(round_i, temp1, how='left', left_index=True, right_index=True)
    round_i.fillna(0,inplace=True)

    # a column of fortune value after one round
    fortune_i = round_i['pre_round_wealth'] + round_i['income_stable'] + round_i['income_unstable'] - \
                round_i['income_tax'] + round_i['social_welfare'] - round_i['accidental_loss']
    fortune_i.astype('Int64')
    return fortune_i


# set initial values of personal wealth with uniform distribution
# assume there's almost no difference among people in one country at first
person_n = [x for x in range(1, 1001)]
fortune = pd.DataFrame([0 for i in range(1000)], index=person_n)
fortune.index.name = 'ID'

for round in range(1,100):
    fortune[round] = simulation(person_n,fortune,round)

result1 = fortune.T


"""
os.chdir('/Users/W/PycharmProjects/final_project/pic1')


def graph1(fortune_i,start,end,length):
    '''

    :param fortune:
    :param start:
    :param end:
    :param length:
    :return:
    '''
    for n in list(range(start,end,length)):
        value = fortune_i.iloc[n]
        plt.figure(figsize=(10,6))
        plt.bar(value.index,value.values,color='gray',alpha=0.8,width=0.9)
        plt.ylim([0,10000000000])
        plt.xlim([0,10001])
        plt.title('Round {}'.format(n))
        plt.xlabel('PlayerID')
        plt.ylabel('Personal wealth')
        plt.grid(color='gray',linestyle='--',linewidth=0.5)
        plt.savefig('graph1_round_{}'.format(n),dpi=200)
        print('success in plotting round {}'.format(n))


graph1(result1,0,100,10)
"""