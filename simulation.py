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
import math
import warnings
warnings.filterwarnings('ignore')
# import modin.pandas as pd

'''
# dataset of personal income from the Current Population Survey(CPS) by US Census Bureau
dataset = pd.read_table('personal_income_2018')

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
    ''' Initialize personal fortune values that are consistent with facutal datasets
    :param people: a list of numbers representing persons in this simulation
    :return: year_0_wealth: the dataframe of initial wealth values for each person
    '''
    year_0_wealth = pd.DataFrame({'ID':people,'pre_year_wealth':[0]*(len(people)),'income_stable':stable_income_sample(len(people)),
                                  'income_tax':0,'income_unstable':0,'social_welfare':0,'accidental_loss':0})
    return year_0_wealth


# generate a list of personal income values based on historical distribution
def stable_income_sample(size):
    ''' Combine 3 distributions in piecewise fashion to generate random sample with income values
    :param size: the number of a group of people
    :return: sample: a ndarray of personal income values
    '''
    sample_left = np.zeros(math.ceil(0.287 * size)).astype('Int32')
    sample_middle = np.random.triangular(0, 1250, 100000, math.ceil(0.645 * size)).astype('Int32')
    sample_right = np.random.choice([100000, 1000000], math.floor(0.068 * size)).astype('Int32')
    sample = np.concatenate([sample_left,sample_middle,sample_right],axis=0)
    np.random.shuffle(sample)
    return sample


def income_update(pre_income,people_id_list):
    ''' Update each person's income value based on that in last year
    :param pre_income: a series of stable income values in lat year
    :param people_id_list: a list of id of people
    :return: pre_income: an updated column of stable income values
    '''
    id_choice = list(np.random.choice(people_id_list,math.ceil(0.5 * len(people_id_list))))
    for i in people_id_list:
        if i-1 in id_choice:
            pre_income[i-1] = pre_income[i-1] * (1 + 0.05)
        else:
            pre_income[i-1] = pre_income[i-1] * (1 - 0.05)
    return pre_income


def random_values(sample,percent,amount,name):
    ''' Given a group of people as sample, the percent of winning/losing a certain amount of money, get a column of values
     with a certain column name.
    :param sample: a list of ID numbers which represents a group of people
    :param percent: the chance of earning or losing money
    :param amount: the amount of gains or losses
    :return: df: the dataframe that contains money values for each person
    '''
    id_choice = list(np.random.choice(sample,math.ceil(percent * len(sample))))
    df = pd.DataFrame({'ID':id_choice,'{}'.format(name):[amount] * len(id_choice)})
    return df


def simulation(fortune_df,year_i,pre_year_income,flat):
    ''' Simulate the gain and loss of personal wealth
    :param fortune_df: the dataframe that contains personal net wealth values during previous years
    :param year_i: an integer that indicates the number of simulation rounds
    :param pre_year_income: present year to be simulated and calculated
    :param flat: a bool value, true means the income tax is flat (same for all income values)
    :return: fortune_i: a column of new fortune value after one round of simulation
    '''
    people = fortune_df['ID']
    year_i_wealth = pd.DataFrame({'ID':people,'pre_year_wealth':fortune_df['year_{}'.format(year_i-1)],
                                  'income_stable':income_update(pre_year_income,people)})
    # stable income tax
    if not flat:
        year_i_wealth['income_tax'] = year_i_wealth['income_stable'].apply(lambda x: 0.1 * x if x <= 9700 else (
            0.1 * 9700 + 0.15 * (x - 9700) if x <= 39475 else (
                0.1 * 9700 + 0.15 * 30045 + 0.25 * (x - 39745) if x <= 84200 else (
                    0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * (x - 82400) if x <= 160725 else (
                        0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * (x - 160725) if x <= 204100 else (
                            0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * 43375 + 0.45 * (x - 204100) if x <= 510300 else
                                0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * 43375 + 0.45 * 306200 + 0.55 * (x - 510300)))))))
    else:
        year_i_wealth['income_tax'] = year_i_wealth['income_stable'].apply(lambda x: x * 0.35)

    # unstable income like gains from lottery
    # assume the probability of winning $1000 is 1/1000
    temp_df1 = random_values(people,1/1000,1000,'income_unstable')
    year_i_wealth = year_i_wealth.merge(temp_df1,how='outer',on='ID')
    year_i_wealth.fillna(0,inplace=True)
    year_i_wealth['income_unstable'].astype('Int32')
    # personal income tax rate
    # social welfare: unemployment benefits, assume $400/week for at most 6 weeks, isn't taxable
    # The standard time-length of unemployment compensation is six months
    year_i_wealth['social_welfare'] = year_i_wealth['income_stable'].apply(lambda x: 400 if x == 0 else 0)  # ???? <0

    # accidental_loss
    temp_df2 = random_values(people,1/50,5000,'accidental_loss')
    year_i_wealth = year_i_wealth.merge(temp_df2,how='outer',on='ID')
    year_i_wealth.fillna(0, inplace=True)
    year_i_wealth['accidental_loss'].astype('Int32')

    return year_i_wealth


def fortune_new(year_i_wealth):
    ''' Given the wealth dataframe of one year, calculate the net wealth value for each person during this year
    :param year_i_wealth: a dataframe contains several columns that are different aspects of personal wealth
    :return: fortune_i: a series of personal net fortune during this year
    '''
    # a column of fortune value after one year passed
    fortune_i = year_i_wealth['pre_year_wealth'] + year_i_wealth['income_stable'] + year_i_wealth['income_unstable'] \
                - year_i_wealth['income_tax'] + year_i_wealth['social_welfare'] - year_i_wealth['accidental_loss']
    # fortune_i.astype('Int64')
    return fortune_i


def graph1(fortune_t,start,end,length):
    ''' Plot fortune values for each person each year
    :param fortune_t: the dataframe contains all year's net fortune values
    :param start: start year of plotting
    :param end: end year of plotting
    :param length: interval between the year every two graphs present
    :return:
    '''
    for n in list(range(start,end,length)):
        # filter by rows
        # new_index = range(1,len(fortune_t.iloc[n+1])+1)
        year_fortune = fortune_t.iloc[n+1].sort_values() # fortune values of all the people during nth year
        year_fortune.reset_index(drop=True,inplace=True)
        plt.figure(figsize=(10,6))
        plt.bar(year_fortune.index,year_fortune.values,color='gray',width=0.5)
        plt.axis([0,1001,-3000,35000000])
        plt.title('year {}'.format(n))
        plt.xlabel('PlayerID')
        plt.ylabel('Personal wealth in {}th year'.format(n))
        plt.grid(color='gray',linestyle='--',linewidth=0.6)
        plt.savefig('graph1_year_{}'.format(n),dpi=200)
        print('Success in plotting round {}'.format(n))



if __name__ == '__main__':
    # set initial values of personal wealth with uniform distribution
    # assume there's almost no difference among people in one country at first
    person_n = [x for x in range(1, 1001)]
    fortune = pd.DataFrame({'ID':person_n,'year_0':[0 for i in range(1000)]})

    start_time = time.time()

    pre_year_income = initial_wealth(person_n)['income_stable']
    year_i_wealth = simulation(fortune,1,pre_year_income,False)
    fortune['year_1'] = fortune_new(year_i_wealth)
    for year in range(2,46):
        pre_year_income = year_i_wealth['income_stable']
        year_i_wealth = simulation(fortune,year,pre_year_income,False)
        fortune['year_{}'.format(year)] = fortune_new(year_i_wealth)
    print(max(fortune['year_10']),min(fortune['year_10']))
    end_time = time.time()

    result1 = fortune.T
    os.chdir('/Users/W/PycharmProjects/final_project/Graph/Graph_loss')
    #graph1(result1,0,46,1)

    ranking1 = pd.DataFrame({'ID':result1.iloc[0],'Fortune':result1.iloc[46]}).sort_values(
        by='Fortune',ascending=False).reset_index(drop=True)
    #ranking1.index_name = 'Rank'
    ranking1['Percent'] = ranking1['Fortune']/ranking1['Fortune'].sum()
    ranking1['Cumulative_sum'] = ranking1['Percent'].cumsum()
    temp_df = pd.DataFrame({'ID':result1.iloc[0],'Initial_fortune':result1.iloc[2]})
    ranking1 = ranking1.merge(temp_df,how='outer',on='ID')
    ranking1['Increased_by'] = ranking1['Fortune']/ranking1['Initial_fortune'] - 1

    # why the round function doesn't work after adding new columns to the dataframe???
    '''
    ranking1 = ranking1.round({'ID':0,'Fortune':0,'Percent':4,'Cumulative_sum':4,'Initial_fortune':0})
    ranking1['ID'] = ranking1['ID'].apply(lambda x: format(x,'.0f'))
    ranking1['Fortune'] = ranking1['Fortune'].apply(lambda x: format(x, '.0f'))
    ranking1['Percent'] = ranking1['Percent'].apply(lambda x: format(x,'.4f'))
    ranking1['Cumulative_sum'] = ranking1['Cumulative_sum'].apply(lambda x: format(x, '.4f'))
    ranking1['Initial_fortune'] = ranking1['Initial_fortune'].apply(lambda x: format(x, '.0f'))'''
    ranking1['Increased_by'] = ranking1['Increased_by'].apply(lambda x: format(x, '.2%'))
    print(ranking1[ranking1['Increased_by']<0].count)
    print(ranking1)

    # bankrupted at 35 years old
    turnpoint_35 = pd.DataFrame({'ID':result1.iloc[0],'Fortune':result1.iloc[36]})

