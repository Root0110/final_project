# IS590_PR_FinalProject_MCS
**Discover the distribution and evolution of social wealth (under different situations) with the use of Monte Carlo Simulations Method**   

By Xiaohan Wang   
December 2019

## Project Background
Wealth inequality is a severe problem worldwide, and in the United States it has increased sharply in recent decades. 
There's a popular rule called `Pareto Principle`(also known as 80/20 rule), which can be applied on many fields and events. 
In economics, it means the richest __20%__ of world's population control more than __80%__ of the world's income.

Based on that, I think it would be interesting to conduct more accurate and detailed analysis on the distribution of wealth problem.
Since the amount of money one person gains and loses are random variables, so I consider Monte Carlo Simulation a great method. 
`Monte Carlo simulation` involves running many scenarios with different random inputs and summarizing the distribution of the results from potential result.

## Hypothesis
* A small portion of people would have the most of wealth within the whole society, just as Pareto Principle states. 
* As time goes by, the difference between top 20% wealthy people's wealth and the rest 80% would be greater and greater.    
* If the piecewise income tax rate changes to flat tax, the difference would be greater.
* Living in situations that would cause much more accidental loss, but with more chance for unstable income, the difference would be greater. 
* People who work harder, even achieved a little more than others, they would have more wealth amounts as long as they persist year by year.

## Project Design 
1. __"Status Quo"__  
    Initial amount of each person's wealth is 0, but the level everyone's income amounts is randomly generated, consistent with below distributions.  
    Based on datasets from `Current Population Survey` by `US Census Bureau`, get the basic distribution of personal income amounts.  
    ![](https://github.com/Root0110/final_project/master/income_distribution.png)   
2. __Run simulations__
Generate random values for each variable based on certain predefined rules.
And suppose people start work at __20__ years old, and would retire at __65__ years old. So, the final result would be achieved after __45__ rounds of simulations(45 years).
Personal wealth(net worth), is defined as gains minus losses, as the followings.
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
3. __Experiment__  
I made some changes on some of above factors, to run models representing different situations.  

| Situation | Stable Income | Unstable Income | Social Welfare | Income Tax | Accidental Loss |
| --- | --- | --- | --- | --- | --- |
| Basic | ±5% | 1/1000,$1000 | $2400 | piecewise | 1/50,-$5000 |
| A | ±5% | 1/1000,$1000 | $2400 | flat(35%) | 1/50,-$5000 |
| B | ±5% | 1/50,$1000 | $2400 | piecewise | 1/5, -$5000 |
| C | +15% -5% | 1/1000,$1000 | $2400 | piecewise | 1/50, -$5000 |

## Future work


### Reference & Data Source
>https://www.census.gov/cps/data/cpstablecreator.html  
>https://en.wikipedia.org/wiki/Pareto_principle#In_economics  
>https://en.wikipedia.org/wiki/Personal_income_in_the_United_States#Income_distribution    
>https://pbpython.com/monte-carlo.html  
>https://www.mikulskibartosz.name/monte-carlo-simulation-in-python/  
>https://equitablegrowth.org/the-distribution-of-wealth-in-the-united-states-and-implications-for-a-net-worth-tax/  
