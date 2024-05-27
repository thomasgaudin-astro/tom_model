#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:26:44 2024

@author: thomasgaudin
"""

#!/usr/bin/env python3

import Tom_Model as TM
import pandas as pd
import numpy as np
import random

from collections import defaultdict
from copy import deepcopy

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import matplotlib.pyplot as plt
import matplotlib.colors as cl
from pandas.plotting import table

from tabulate import tabulate

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings

warnings.filterwarnings("ignore")

theta = 1.7

iterations = 10000

print("Running the Tom Model for Copa America 2024. \n")

init_table_file = 'Copa_Am_info.csv'
current_table_file = 'Copa_Am_info_current.csv'

init_sched_file = 'Copa_Am_schedule.csv'
current_sched_file = 'Copa_Am_schedule_current.csv'

current = input('Do you want to use current Elo, Points, and Schedule? \n Y - Start from Current Spot. \n N - Start from the Beginning. \n')

if current == 'Y':
    ca_group_table = pd.read_csv(current_table_file, header=0)
    matches = pd.read_csv(current_sched_file, header=0)
    
elif current == 'N':
    ca_group_table = pd.read_csv(init_table_file, header=0)
    matches = pd.read_csv(init_sched_file, header=0)
    
else:
    print('Please Pick a Valid Response.')
    
ca_group_table = ca_group_table.set_index('Code')
ca_group_table['Points'] = 0

init_group_table = deepcopy(ca_group_table)

init_matches = deepcopy(matches)

team_tuples = [(x, y) for x, y in zip(ca_group_table['Group'], ca_group_table.index)]

cols = pd.MultiIndex.from_tuples(team_tuples)

#Create dataframe with MultiIndex columns
placement_table = pd.DataFrame(0,
                            index = np.arange(1, 5),
                            columns = cols)

#finish table
finish_table = pd.DataFrame(0,
                            index = ca_group_table.index,
                            columns = ['QF', 'SF', '3rd', '2nd', '1st'])

#initialize lists, dictionaries, arrays

groups = ["A", "B", "C", "D"]

total_table = ca_group_table.loc[:, ['Points', 'Group']].copy()

total_table['QF'] = 0
total_table['SF'] = 0
total_table['3rd'] = 0
total_table['2nd'] = 0
total_table['1st'] = 0

group_stage = matches[matches['Match Num'] < 25]

num_group_matches = 3

host = ca_group_table[ca_group_table['Host'] == 'Y'].index[0]

groups_dict = {'A': [25, 26], 'B': [26, 25], 'C': [27, 28], 'D': [28, 27]}

kos_dict = {'25': [29, 'Home'], '26': [29, 'Away'], '27': [30, 'Home'], '28': [30, 'Away'], 
            '29': [32, 'Home'], '30': [32, 'Away']}

for iteration in range(iterations):
    for match in (group_stage['Match Num']-1):
        
        #calculate win expectancies
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(group_stage, 
                                                                       match, ca_group_table, host)
        
        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

    
        ca_group_table = TM.play_gs_match_neutral(ca_group_table, home_team, home_we, home_elo, home_wp, 
                          away_team, away_we, away_elo, away_wp, draw_wp)
            
    
    for team in ca_group_table.index:
        team_points = ca_group_table.loc[team, 'Points']
        total_table.loc[team, 'Points'] += team_points
        
    for group in groups:
        
        group_tab = ca_group_table[ca_group_table['Group'] == group]
        ranked_group_tab = group_tab.sort_values(by='Points', ascending=False)
        ranked_group_tab['Rank'] = group_tab['Points'].rank(method='max', ascending = False)
        
        for team in ranked_group_tab.index:
            rank = ranked_group_tab.loc[team, 'Rank']
            group = ranked_group_tab.loc[team, 'Group']
            
            placement_table.loc[rank, (group, team)] += 1
        
        #assign teams to knockout games:
        
        #winner goes to a match first
        winner_match = groups_dict[group][0]-1
        matches.loc[winner_match, 'Home'] = ranked_group_tab.index[0]
        
        #2nd place goes to a match next
        second_match = groups_dict[group][1]-1
        matches.loc[second_match, 'Away'] = ranked_group_tab.index[1]
        
    QF = matches[matches['Group'] == 'QF']
    SF = matches[matches['Group'] == 'SF']
    Third = matches[matches['Group'] == '3P']
    Final = matches[matches['Group'] == 'Final']
    
    for match in (QF['Match Num']-1):
        
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(QF, 
                                                                                           match, 
                                                                                           ca_group_table,
                                                                                             host)

        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

        ca_group_table, QF = TM.play_ko_match_neutral(ca_group_table, QF, match,
                                  home_team, home_we, home_elo, home_wp, 
                                  away_team, away_we, away_elo, away_wp, draw_wp)
        
        next_match = int(kos_dict[str(match+1)][0]-1)
        h_a = kos_dict[str(match+1)][1]
        
        SF.loc[next_match, h_a] = QF.loc[match, 'Winner']
        
    for team in total_table.index:
        if team in list(QF['Winner']):
            total_table.loc[team, 'SF'] += 1
        else:
            continue
        
    for match in (SF['Match Num']-1):
        
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(SF, 
                                                                                           match, 
                                                                                           ca_group_table, host)

        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

        ca_group_table, SF = TM.play_ko_match_neutral(ca_group_table, SF, match,
                                  home_team, home_we, home_elo, home_wp, 
                                  away_team, away_we, away_elo, away_wp, draw_wp)
        
        next_match = int(kos_dict[str(match+1)][0]-1)
        h_a = kos_dict[str(match+1)][1]
        
        teams = [SF.loc[match, 'Home'], SF.loc[match, 'Away']]
        
        Final.loc[next_match, h_a] = SF.loc[match, 'Winner']
        
        teams.remove(SF.loc[match, 'Winner'])
        Third.loc[30, h_a] = teams[0]
        
    
    home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(Third, 
                                                                                        30, 
                                                                                        ca_group_table, host)

    #Determine win probability for each team
    home_wp = TM.davidson_home_wp(home_we, away_we, theta)
    away_wp = TM.davidson_away_wp(home_we, away_we, theta)
    draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

    ca_group_table, Third = TM.play_ko_match_neutral(ca_group_table, Third, 31,
                                  home_team, home_we, home_elo, home_wp, 
                                  away_team, away_we, away_elo, away_wp, draw_wp)
    
    home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(Final, 
                                                                                        31, 
                                                                                        ca_group_table,
                                                                                         host)
    
    tp = Third.loc[31, 'Winner']
    
    total_table.loc[tp, '3rd'] += 1

    #Determine win probability for each team
    home_wp = TM.davidson_home_wp(home_we, away_we, theta)
    away_wp = TM.davidson_away_wp(home_we, away_we, theta)
    draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

    ca_group_table, Final = TM.play_ko_match_neutral(ca_group_table, Final, 31,
                                  home_team, home_we, home_elo, home_wp, 
                                  away_team, away_we, away_elo, away_wp, draw_wp)
    
    champ = Final.loc[31, 'Winner']
    total_table.loc[champ, '1st'] += 1
    
    final_teams = [Final.loc[31, 'Home'], Final.loc[31, 'Away']]

    final_teams.remove(Final.loc[31, 'Winner'])
    runner_up = final_teams[0]
    
    total_table.loc[runner_up, '2nd'] += 1
    
            
    ca_group_table = deepcopy(init_group_table)
    matches = deepcopy(init_matches)
    
    print(f'Running Iteration {iteration+1} / {iterations}.')
    print(f'Winner of Iteration {iteration+1}: {champ}')
    
total_table['Avg Points'] = total_table['Points'] / iterations
total_table['Avg PPG'] = total_table['Points'] / (iterations*num_group_matches)

percent_finish = placement_table / iterations

for team in total_table.index:
    group = total_table.loc[team, 'Group']
    finish_table.loc[team, 'QF'] = round((percent_finish.loc[1, (group, team)] + percent_finish.loc[2, (group, team)]),4)
    
    finish_table.loc[team, 'SF'] = round(total_table.loc[team, 'SF'] / iterations,4)
    finish_table.loc[team, '3rd'] = round(total_table.loc[team, '3rd'] / iterations,4)
    finish_table.loc[team, '2nd'] = round(total_table.loc[team, '2nd'] / iterations,4)
    finish_table.loc[team, '1st'] = round(total_table.loc[team, '1st'] / iterations,4)
    
sorted_finish = finish_table.sort_values('1st', ascending=False)

fig1, ax = plt.subplots(1, 1, facecolor='white')
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=sorted_finish.values, cellColours=plt.cm.YlOrRd(sorted_finish.values),
                 rowLabels=sorted_finish.index, colLabels=sorted_finish.columns, 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(3, 4)

table_props = table.properties()
table_cells = table_props['children']
for cell in table_cells:
    if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.4')):
        cell.get_text().set_color('white')


# ab = AnnotationBbox(getImage("./country-flags-main/png1000px/ar.png"), (200, 1837), 
#                     frameon=False, xycoords='figure points')
# ax.add_artist(ab)


for val in range(0,5):
    head = table[0,val]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

for val in range(1,17):
    head = table[val,-1]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

    
fig1.savefig('./Tom Model Outputs/Copa_America_Odds.png', bbox_inches="tight")

fig2, ax = plt.subplots(4, 1, figsize=(12,10), facecolor='white')

for val in range(0,4):
    
    group = groups[val]
    
    pct_fin_group = percent_finish[group].sort_values(1, axis=1, ascending=False)
    
    ax[val % 4].axis('off')
    ax[val % 4].axis('tight')
    
    ax[val % 4].set_title(f"Group {group}", fontsize=18, fontweight='bold')
    
    table = ax[val % 4].table(cellText=pct_fin_group.values, 
                                     cellColours=plt.cm.YlOrRd(pct_fin_group.values),
                                     rowLabels=pct_fin_group.index, colLabels=pct_fin_group.columns, 
                                     loc='center')
    table.scale(1,1.7)
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    
    table_props = table.properties()
    table_cells = table_props['children']
    for cell in table_cells:
        if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.4')):
            cell.get_text().set_color('white')

    
    for val in range(1,5):
        row = table[val,-1]
        row.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')
        row.PAD = 0.4

        head = table[0,val-1]
        head.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')

fig2.savefig('./Tom Model Outputs/Copa_America_Placements.png', bbox_inches="tight")

print('Simulation is Complete. \n')
print('Outputting 2 Tables showing your results.')
