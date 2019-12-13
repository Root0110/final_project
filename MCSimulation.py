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
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
from tabulate import tabulate


def plot_dist(filename):
    """
    Given historical data file, basically plot the distribution of personal income values.
    :param filename: this file must be in the same directory with simulation.py file
    :return:
    """
    os.chdir('/Users/W/PycharmProjects/final_project/Datasets')
    # dataset of personal income from the Current Population Survey(CPS) by US Census Bureau
    dataset = pd.read_table(filename)
    # explore the distribution of real data
    n = dataset.count()
    income_points = []
    for i in range(n[0]):
        income = [dataset.iloc[i,0]]*dataset.iloc[i,1]
        income_points += income
    # try to plot and fit data to certain distribution
    sns_plot = sns.distplot(income_points)
    sns_plot.figure.savefig('Income_distribution.png')


def initial_wealth(people):
    """
    Initialize personal fortune values that are consistent with historical datasets
    :param people: a list of numbers representing persons in this simulation
    :return: year_0_wealth: the dataframe of initial wealth values for each person
    """
    year_0_wealth = pd.DataFrame({'ID':people,'pre_year_wealth':[0]*(len(people)),'income_stable':stable_income_sample(len(people)),
                                  'income_tax':0,'income_unstable':0,'social_welfare':0,'accidental_loss':0})
    return year_0_wealth


def stable_income_sample(size: int):
    """
    Combine 3 distributions in piecewise fashion to generate random sample with income values
    :param size: the number of a group of people
    :return: sample: a ndarray of personal income values
    >>> a = stable_income_sample(10)
    >>> b = stable_income_sample(10)
    >>> list(a) == list(b)  # Elements in two arrays are not totally the same
    False
    >>> len(stable_income_sample(10)) # Expected output has the same length as the input
    10
    """
    # the following three percentage numbers are averaged from three real datasets(2016-2018 CPS)
    sample_left = np.zeros(math.ceil(0.287 * size)).astype('Int32')
    sample_middle = np.random.triangular(0, 1250, 100000, math.ceil(0.645 * size)).astype('Int32')
    sample_right = np.random.choice([100000, 1000000], math.floor(0.068 * size)).astype('Int32')
    sample = np.concatenate([sample_left,sample_middle,sample_right],axis=0)
    np.random.shuffle(sample)
    return sample


def income_update(pre_income, people_id_list, work_harder_game=False, work_harder_id=[]):
    """
    Update each person's income value based on that in last year
    :param pre_income: a series of stable income values in lat year
    :param people_id_list: a list of id of people
    :param work_harder_id: a bool value, if true, there're some people who work harder than the rest
    :param work_harder_game: a bool value, if true, those who work harder would have advantages over others
    :return: pre_income: an updated list of stable income values
    >>> pre_income = [0,1,5,1,10,2]
    >>> people_id_list = [1,2,3,4,5,6]
    >>> new_income = income_update(pre_income, people_id_list, False)
    >>> for i in new_income:
    """
    if not work_harder_game:
        work_harder_id = []
    rest_id = list(set(people_id_list) - set(work_harder_id))
    # a list of people whose income gets increased
    id_choice = list(np.random.choice(rest_id, math.ceil(0.5 * len(rest_id))))
    # a list of people who work harder and get income increased much more
    for i in people_id_list:
        if i in work_harder_id:
            pre_income[i-1] = pre_income[i-1] * (1 + 0.15)
        elif i in id_choice:
            pre_income[i-1] = pre_income[i-1] * (1 + 0.05)
        else:
            pre_income[i-1] = pre_income[i-1] * (1 - 0.05)
    return pre_income


def random_values(sample, percent, amount, name: str):
    """
    Given a group of people as sample, the percent of winning/losing a certain amount of money, get a column of values
    with a certain column name.
    :param sample: a list of ID numbers which represents a group of people
    :param percent: the chance of earning or losing money
    :param amount: the amount of gains or losses
    :param name: the name for new column in this dataframe
    :return: df: the dataframe that contains money values for each person
    >>> sample = [1,2,3,4]
    >>> df = random_values(sample,0.5,10,'test')
    >>> df['test'].count()  # len(sample) * percent
    2
    """
    # randomly choose a certain number of people id numbers
    id_choice = list(np.random.choice(sample,math.ceil(percent * len(sample))))
    # assign certain values for those who are chosen
    df = pd.DataFrame({'ID':id_choice,'{}'.format(name):[amount] * len(id_choice)})
    return df


def simulation(fortune_df, year_i: int, pre_year_income, flat=False, work_harder_game=False, work_harder_id=None):
    """
    Simulate the gain and loss of personal wealth
    :param fortune_df: the dataframe that contains personal net wealth values during previous years
    :param year_i: an integer that indicates the number of simulation rounds
    :param pre_year_income: present year to be simulated and calculated
    :param flat: a bool value, true means the income tax is flat (same for all income values)
    :param work_harder_game: a bool value, if true, those who work harder would have advantages over others
    :param work_harder_id: a list of people who work harder to get more income increased
    :return: fortune_i: a column of new fortune value after one round of simulation
    """
    people = fortune_df['ID']  # a list of number from 1 to 1000
    year_i_wealth = pd.DataFrame({'ID':people,'pre_year_wealth':fortune_df['year_{}'.format(year_i-1)],
                                 'income_stable':income_update(pre_year_income,people,work_harder_game,work_harder_id)})
    # the amount of stable income tax
    if not flat:
        # the below method for evaluating income tax is piecewise
        year_i_wealth['income_tax'] = year_i_wealth['income_stable'].apply(lambda x: 0.1 * x if x <= 9700 else (
           0.1 * 9700 + 0.15 * (x - 9700) if x <= 39475 else (
               0.1 * 9700 + 0.15 * 30045 + 0.25 * (x - 39745) if x <= 84200 else (
                   0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * (x - 82400) if x <= 160725 else (
                       0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * (x - 160725) if x <= 204100 else (
                           0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * 43375 + 0.45 * (x - 204100) if x <= 510300 else
                               0.1 * 9700 + 0.15 * 30045 + 0.25 * 42655 + 0.3 * 78325 + 0.35 * 43375 + 0.45 * 306200 + 0.55 * (x - 510300)))))))
    else:
        # with flat income tax rate of 35%, evaluate the amount of income tax
        year_i_wealth['income_tax'] = year_i_wealth['income_stable'].apply(lambda x: x * 0.35)

    # unstable income like gains from lottery
    # assume the probability of winning $1000 is 1/1000 one year
    temp_df1 = random_values(people,1/1000,1000,'income_unstable')
    year_i_wealth = year_i_wealth.merge(temp_df1,how='outer',on='ID')
    year_i_wealth.fillna(0,inplace=True)
    year_i_wealth['income_unstable'].astype('Int32')

    # personal income tax rate
    # social welfare: unemployment benefits, assume $400/week for at most 6 weeks, isn't taxable
    # The standard time-length of unemployment compensation is six months
    year_i_wealth['social_welfare'] = year_i_wealth['income_stable'].apply(lambda x: 2400 if x == 0 else 0)

    # accidental_loss like sudden illness, car accidents, law suit,etc.
    # assume the probability of losing $5000 is 1/50 one year
    temp_df2 = random_values(people,1/50,5000,'accidental_loss')
    year_i_wealth = year_i_wealth.merge(temp_df2,how='outer',on='ID')
    year_i_wealth.fillna(0, inplace=True)
    year_i_wealth['accidental_loss'].astype('Int32')
    return year_i_wealth


def fortune_new(year_i_wealth):
    """
    Given the wealth dataframe of one year, calculate the net wealth value for each person during this year
    :param year_i_wealth: a dataframe contains several columns that are different aspects of personal wealth
    :return: fortune_i: a series of personal net fortune during this year
    """
    # a column of fortune value after one year passed
    fortune_i = year_i_wealth['pre_year_wealth'] + year_i_wealth['income_stable'] + year_i_wealth['income_unstable'] \
                - year_i_wealth['income_tax'] + year_i_wealth['social_welfare'] - year_i_wealth['accidental_loss']
    return fortune_i


def graph(fortune_t, start: int, end: int, length, work_harder_game=False, work_harder_list=None):
    """
    Plot fortune values for each person each year.
    :param fortune_t: the dataframe contains all year's net fortune values
    :param start: start year of plotting
    :param end: end year of plotting
    :param length: interval between the year every two graphs present
    :param work_harder_game: a bool value, if true, those who work harder would have advantages over others
    :param work_harder_list: a list of people who work harder
    :return:
    """
    # change directory to save graph images under different situations
    os.chdir('/Users/W/PycharmProjects/final_project/Graph')
    for n in list(range(start,end,length)):
        # fortune values of all the people during nth year
        year_fortune = pd.DataFrame({'ID':fortune_t.iloc[0],'Fortune':fortune_t.iloc[n+1],'color':'gray'}).sort_values(by='ID')
        if work_harder_game:
            # change the color of bars to make the graph more intuitive
            year_fortune['color'].iloc[list(work_harder_list)] = 'red'
        year_fortune.sort_values(by='Fortune',inplace=True,ascending=True)  # sort by people's total fortune values
        year_fortune.reset_index(drop=True,inplace=True)
        plt.figure(figsize=(10,6))
        plt.bar(year_fortune.index,year_fortune['Fortune'],color=year_fortune['color'],width=0.65)
        plt.axis([0,1001,-5000,30000000])
        plt.title('year {}'.format(n))
        plt.xlabel('PlayerID')
        plt.ylabel('Personal wealth in {}th year'.format(n))
        plt.grid(color='gray',linestyle='--',linewidth=0.6)
        plt.savefig('graph1_year_{}'.format(n),dpi=200)
        print('Success in plotting round {}'.format(n))


def analyze(round_result, year):
    """
    Given the final round of simulation results, analyze the data to find interesting rules.
    :param round_result: the dataframe contains the total wealth of each person after  years
    :param year: the number of rounds of simulation
    :return: ranking: the dataframe containing analysis data
    """
    ranking = pd.DataFrame({'ID':round_result.iloc[0],'Fortune':round_result.iloc[year+1]}).sort_values(
        by='Fortune',ascending=False).reset_index(drop=True)
    ranking['Percent'] = ranking['Fortune'] / ranking['Fortune'].sum()  # the ratio of one's fortune to the whole fortune
    ranking['Cumulative_sum'] = ranking['Percent'].cumsum()  # the cumulative sum value
    temp_df = pd.DataFrame({'ID': round_result.iloc[0], 'Initial_fortune': round_result.iloc[2]})
    ranking = ranking.merge(temp_df, how='outer', on='ID')  # add initial fortune of each person to make comparisons
    ranking['Increased_by'] = ranking['Fortune'] / ranking['Initial_fortune'] - 1
    # ranking['Increased_by'] = ranking['Increased_by'].apply(lambda x: format(x, '.2%'))
    # the number of people who have less wealth than 45 years ago
    m = ranking[ranking['Increased_by'] < 0]['ID'].count()
    n = ranking.loc[99]  # the amounts of wealth top 10% people would obtain
    q = ranking.loc[199][3]  # the accumulative sum of wealth percentage that the 200th person owns
    return ranking


def wealth_difference(acc_list):
    """
    Given the list of accumulative sum of wealth shares, save the data in each year, and plot the tendency.
    :param acc_list: a list of wealth shares that top 20% people would obtain during years
    :return:
    """
    os.chdir('/Users/W/PycharmProjects/final_project')
    with open('Wealth_difference_years.txt','w') as f2:
        for item in acc_list:
            f2.write('{}\n'.format(item))
    x = list(range(1,46))
    y = acc_list
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('Year')
    plt.ylabel('Accumulative sum of wealth shares of the 200th person')
    plt.savefig('wealth_difference_tendency.png')


if __name__ == '__main__':
    plot_dist('personal_income_2018')

    person_n = [x for x in range(1, 1001)]  # each id number represents a person
    work_harder_id = range(1, 1001, 100)  # choose 10 persons who work harder with fixed id numbers
    work_harder_list = list(work_harder_id)  # make range() type to list() so it can be changed
    for i in range(len(work_harder_list)):
        work_harder_list[i] -= 1  # because the difference between real ID numbers and column index is 1

    fortune = pd.DataFrame({'ID':person_n,'year_0':[0 for i in range(1000)]})
    # during the 1st year, the stable income values are assigned randomly according to factual distribution
    pre_year_income = initial_wealth(person_n)['income_stable']
    year_i_wealth = simulation(fortune,1,pre_year_income,False,False,work_harder_id)
    fortune['year_1'] = fortune_new(year_i_wealth)
    ranking_1 = analyze(fortune.T,1)
    acc_list = [ranking_1.loc[199][3]]  # accumulative sum of wealth percentage that the 200th person owns

    # from 2nd year on, stable income values would increase/decrease a little based on initial income levels
    for year in range(2,46):
        pre_year_income = year_i_wealth['income_stable']
        year_i_wealth = simulation(fortune,year,pre_year_income,False,False,work_harder_id)
        fortune['year_{}'.format(year)] = fortune_new(year_i_wealth)
        ranking = analyze(fortune.T,year)
        acc_list.append(ranking.loc[199][3])  # append wealth shares that top 20% people would obtain during years

    # plot the tendency
    wealth_difference(acc_list)

    # plot the final round of simulation results
    graph(fortune.T,0,46,1,False,work_harder_list)

    # analyze and store the final round of simulation results
    os.chdir('/Users/W/PycharmProjects/final_project')
    with open('Final_round_ranking.txt', 'w') as f3:
        print(tabulate(ranking, headers='keys', tablefmt='psql'), file=f3)


    # Bool values in simulation() can be turned on for different situations (basic, flat income tax, work harder game).
    # In the third situation, we need to manually change the probability and money amounts in simulation().

