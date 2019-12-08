# IS590_PR_FinalProject_MCS
**Discover the distribution and evolution of social wealth (under different situations) with the use of Monte Carlo Simulations Method**   

By Xiaohan Wang   
December 2019

## Project Background
Wealth inequality is a severe problem worldwide, and in the United States it has increased sharply in recent decades. 
It's of great necessity for researchers to design a set of rules to maintain relative justice in the society, since it matters a lot to all of us. 
Before that, we need to understand the distribution of wealth within the whole society, to see how it would be assigned to each individual.
There's a popular rule called Pareto Principle(also known as 80/20 rule), which can be applied on many fields and events. 
In economics, it means the richest __20%__ of world's population control more than __80%__ of the world's income.

Based on that, I think it would be interesting to conduct more accurate and detailed analysis on the distribution of wealth problem, like adding more influential factors, etc.
Since the amount of money one person gains and loses are random variables, so I consider Monte Carlo Simulation a great method. 
The __Monte Carlo simulation__ involves running many scenarios with different random inputs and summarizing the distribution of the results from potential result.
This distribution can inform the likelihood that the result will be within a certain window.

## Hypothesis
    
* A small portion of people would obtain the most of wealth within the whole society, just as Pareto Principle states. 
* As time goes by, the difference between top 20% wealthy people's wealth and the rest 80% would be greater and greater.    
* If the income tax rate changes to flat tax, the difference between top 20% wealthy people's wealth and the rest 80% would be greater.
* People who work harder, even a little, they would have more wealth amounts as long as they persist year by year.

## Project Design 
Firstly, the distribution of personal income values is plotted, according to datasets from the Current Population Survey(CPS) by US Census Bureau. 
* __"Status Quo" (Initial amount of wealth)__  
    Based on 
It's realistic that some of the social members are born with some assets, like the any kinds of heritage form their ancestors. S
ince not everyone has uniform amount of assets, so I initialize personal wealth value with random numbers. The __distribution/proportion__ is consistent with xxxxx research/report.

Secondly, generate random values for each variable based on certain predefined rules.
Every round of simulation represents the change of personal wealth during one year.
And suppose people start work at 20 years old, and would retire after 45 years. So, the final result would be achieved after 45 rounds of simulations.
Personal wealth(net worth), is defined as gains minus losses, as the followings.
Then, conduct fundamental analysis of accumulative personal wealth amounts.
* __Gains__
    1. Stable Income:  
       Individual's total earnings from wages, investment interest, and other sources. It's the main part in a person's assets
    2. Unstable Income:  
       Like gains from stock or lottery, this kind of events occurs at a certain chance.
    3. Social Welfare:  
       In modern society, the government feels responsible for providing some welfare for __every__ social member. 
       It can be medical insurance, pension fee, and other kinds of life guarantee. 
       This is a kind of stable addition to personal wealth, and it's not distinguishing between the poor and the rich.       
* __Losses__
    1. Income Tax:
       It's an efficient way for the government to get citizens paying all kinds of tax to be involved in the construction of country. 
       Here I mainly focus on personal income tax rate. 
       And to be more representative of the real situation, I adopt piecewise income tax rate as follows.
    2. Accidental Loss:
       Like illness, natural disasters, suffering from wars, and so on. This value would be randomly assigned to a certain percentage of people.

Then, I made some changes on above factors, to run models representing countries under different situations.
1. __Situation A__: With flat personal income rate, people with different income need to pay the same amount of tax.
2. __Situation B__: People are suffering more accidental losses, because of natural disasters, escalating wars, etc.   
3. __Situation C__: There're some people who work harder than others, which means, their income would be increased 10% more than others.


#### Reference & Data Source
https://en.wikipedia.org/wiki/Pareto_principle#In_economics
https://pbpython.com/monte-carlo.html
https://www.mikulskibartosz.name/monte-carlo-simulation-in-python/