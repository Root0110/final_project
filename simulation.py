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


def initial_wealth(people):
    '''
    :param people:
    :return:
    '''
    year_0_wealth = pd.DataFrame({'pre_year_wealth':[0]*(len(people)),'income_stable':stable_income_sample(len(people)),
                                  'income_tax':0,'income_unstable':0,'social_welfare':0,'accidental_loss':0},index=people)
    return year_0_wealth


# generate a list of personal income values based on historical distribution
def stable_income_sample(size):
    ''' Combine 3 distributions in piecewise fashion to generate random sample with income values
    :param size: number of people
    :return: a ndarray of personal income values
    '''
    sample_left = np.zeros(math.ceil(0.286 * size)).astype('Int32')
    sample_middle = np.random.triangular(0, 1250, 100000, math.ceil(0.643 * size)).astype('Int32')
    sample_right = np.random.choice([100000, 10000000000], math.floor(0.071 * size)).astype('Int32')
    sample = np.concatenate([sample_left,sample_middle,sample_right])
    np.random.shuffle(sample)
    return sample


def income_update(pre_income):
    ''' Update each person's income value based on that in last year
    :param pre_income:
    :return: pre_year_income: an updated column
    '''
    people_id_list = pre_income.index
    id_choice = list(np.random.choice(people_id_list,math.ceil(0.5 * len(people_id_list))))
    for i in pre_year_income.index:
        if i in id_choice:
            pre_income[i] = pre_income[i] * (1 + 0.1)
        else:
            pre_income[i] = pre_income[i] * (1 - 0.1)
    return pre_income


def random_values(sample,percent,amount):
    ''' Given a group of people, the chance of winning a certain amount of unstable income,
    get a column of unstable income for each person.
    :param people: a list of ID numbers which represents a group of people
    :param chance: the chance of earning unstable income
    :param amount: the amount of unstable income
    :return: unstable_df: a dataframe that contains unstable income values for each person
    '''
    id_choice = list(np.random.choice(sample,math.ceil(percent * len(sample))))
    part1 = pd.Series([amount] * len(id_choice),index=id_choice)
    rest_people = list(set(sample) - set(id_choice))
    part2 = pd.Series([0] * (len(rest_people)), index=rest_people)
    column_values = pd.concat([part1,part2])
    return column_values


def simulation(fortune_df,year_i,pre_year_income):
    ''' Simulate the gain and loss of personal wealth in
    :param fortune_df: the dataframe that contains initial personal wealth value of the whole people
    :param year_i:
    :return: fortune_i: a column of new fortune value after one round of simulation
    '''
    people = fortune_df.index
    year_i_wealth = pd.DataFrame({'pre_year_wealth':fortune_df[year_i - 1],'income_stable':income_update(pre_year_income)})
    # calculate the amount of tax to pay based on stable income
    year_i_wealth['income_tax'] = year_i_wealth['income_stable'].apply(lambda x: 0.1 * x if x <= 9700 else (
        0.1 * 9700 + 0.12 * (x - 9700) if x <= 39475 else (
            0.1 * 9700 + 0.12 * 30045 + 0.22 * (x - 39745) if x <= 84200 else (
                0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * (x - 82400) if x <= 160725 else (
                    0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * (x - 160725) if x <= 204100 else (
                        0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * 43375 + 0.35 * (
                                    x - 204100) if x <= 510300 else
                        0.1 * 9700 + 0.12 * 30045 + 0.22 * 42655 + 0.24 * 78325 + 0.32 * 43375 + 0.35 * 306200 + 0.37 * (
                                    x - 510300)))))))

    # unstable income like gains from lottery
    # assume the probability of winning $5 is 1/100
    year_i_wealth['income_unstable'] = random_values(people,1/100,5)

    # personal income tax rate
    # social welfare: unemployment benefits, assume $400/week for at most 6 weeks, isn't taxable
    # The standard time-length of unemployment compensation is six months
    year_i_wealth['social_welfare'] = year_i_wealth['income_stable'].apply(lambda x: 2400 if x == 0 else 0)

    # accidental_loss
    year_i_wealth['accidental_loss'] = random_values(people,1/50,500)
    '''id_choice2 = np.random.choice(people, math.ceil((1 / 50) * len(people)))
    temp1 = pd.DataFrame({'accidental_loss': [500] * len(id_choice2)}, index=id_choice2)
    year_i_wealth = pd.merge(year_i_wealth, temp1, how='left', left_index=True, right_index=True)
    year_i_wealth.fillna(0,inplace=True)'''
    year_i_wealth.sort_index()
    return year_i_wealth


def fortune_new(year_i_wealth):
    # a column of fortune value after one year passed
    fortune_i = year_i_wealth['pre_year_wealth'] + year_i_wealth['income_stable'] + year_i_wealth['income_unstable'] - \
                year_i_wealth['income_tax'] + year_i_wealth['social_welfare'] - year_i_wealth['accidental_loss']
    #fortune_i.astype('Int64')

    return fortune_i


if __name__ == '__main__':
    # set initial values of personal wealth with uniform distribution
    # assume there's almost no difference among people in one country at first
    person_n = [x for x in range(1, 1001)]
    fortune = pd.DataFrame([0 for i in range(1000)], index=person_n)
    fortune.index.name = 'ID'

    pre_year_income = initial_wealth(person_n)['income_stable']
    year_i_wealth = simulation(fortune,1,pre_year_income)
    fortune[1] = fortune_new(year_i_wealth)
    for year in range(2,60):
        pre_year_income = year_i_wealth['income_stable']
        year_i_wealth = simulation(fortune,year,pre_year_income)
        fortune[year] = fortune_new(year_i_wealth)


result1 = fortune.T


"""
os.chdir('/Users/W/PycharmProjects/final_project')

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