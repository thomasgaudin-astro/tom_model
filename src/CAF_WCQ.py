#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:09:43 2024

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
from pandas.plotting import table
import matplotlib.colors as cl

from tabulate import tabulate

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import warnings

warnings.filterwarnings("ignore")

theta = 1.7

iterations = 10000

print("\n Running the Tom Model for CONMEBOL 2026 World Cup Qualifying. \n")


init_table_file = 'CAF_initial.csv'
current_table_file = 'CAF_current.csv'

init_sched_file = 'CAF_schedule_init.csv'
current_sched_file = 'CAF_schedule_current.csv'

current = input('Do you want to use current Elo, Points, and Schedule? \n Y - Start from Current Spot. \n N - Start from the Beginning. \n')

if current == 'Y':
    quals_group_table = pd.read_csv(current_table_file, header=0)
    matches = pd.read_csv(current_sched_file, header=0)
    
elif current == 'N':
    quals_group_table = pd.read_csv(init_table_file, header=0)
    matches = pd.read_csv(init_sched_file, header=0)
    
else:
    print('Please Pick a Valid Response.')
    
quals_group_table = quals_group_table.set_index('Code')

init_group_table = deepcopy(quals_group_table)

team_tuples = [(x, y) for x, y in zip(quals_group_table['Group'][:54], quals_group_table.index[:54])]

cols = pd.MultiIndex.from_tuples(team_tuples)

#Create dataframe with MultiIndex columns
placement_table = pd.DataFrame(0,
                            index = np.arange(1, 7),
                            columns = cols)

#finish table
finish_table = pd.DataFrame(0,
                            index = quals_group_table.index[:54],
                            columns = ['Interconfed Playoff', 'Qualify'])

#second place table
second_table = pd.DataFrame(0,
                            index = np.arange(1, 10),
                            columns = ['Team', 'Points'])

init_second_table = deepcopy(second_table)

groups = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

total_table = quals_group_table.loc[:, ['Points', 'Group']].copy()
total_table['R2 SF'] = 0
total_table['R2 F'] = 0

num_matches = len(matches['Match Num'])

for iteration in range(iterations):
    
    r1_table = quals_group_table.iloc[:len(quals_group_table)-4, :]
    r2_table = quals_group_table[quals_group_table['Group'] == "R2"]
    
    r1_matches = matches[:num_matches-3]
    r2_SF = matches[matches['Group'] == "R2 SF"]
    r2_F = matches[matches['Group'] == "R2 F"]
    
    for match in range(len(r1_matches.index)-1):
        
        #calculate win expectancies
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_match_We(matches, 
                                                                       match, quals_group_table)
        
        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

        quals_group_table = TM.play_gs_match_neutral(quals_group_table, home_team, home_we, home_elo, home_wp, 
                          away_team, away_we, away_elo, away_wp, draw_wp)
        
    for team in r1_table.index:
        team_points = r1_table.loc[team, 'Points']
        total_table.loc[team, 'Points'] += team_points
        
    for ind, group in enumerate(groups):
        
        group_tab = r1_table[r1_table['Group'] == group]
        ranked_group_tab = group_tab.sort_values(by='Points', ascending=False)
        ranked_group_tab['Rank'] = group_tab['Points'].rank(method='max', ascending = False)

        for team in ranked_group_tab.index:
            rank = ranked_group_tab.loc[team, 'Rank']
            group = ranked_group_tab.loc[team, 'Group']

            placement_table.loc[rank, (group, team)] += 1
            
        second_place = ranked_group_tab.index[1]
            
        second_table.loc[ind, 'Team'] = second_place
        second_table.loc[ind, 'Points'] = ranked_group_tab.loc[second_place, 'Points']
        
    second_table_ranked = second_table.sort_values(by='Points', ascending=False)
    
    r2_table = r2_table.set_index(second_table_ranked['Team'][0:4])
    
    for team in r2_table.index:
        r2_table.loc[team, ['Points', 'Elo']] = r1_table.loc[team, ['Points', 'Elo']]
        total_table.loc[team, 'R2 SF'] += 1
    
    r2_SF.loc[num_matches-3, 'Home'] = r2_table.index[0]
    r2_SF.loc[num_matches-3, 'Away'] = r2_table.index[3]
    
    r2_SF.loc[num_matches-2, 'Home'] = r2_table.index[1]
    r2_SF.loc[num_matches-2, 'Away'] = r2_table.index[2]
    
    
    for match in range(num_matches-3, num_matches-1):
            
        #calculate win expectancies
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_match_We(r2_SF, match, r2_table)
        
        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

        r2_table, r2_SF = TM.play_ko_match(r2_table, r2_SF, match, home_team, home_we, 
                                                       home_elo, home_wp, away_team, 
                                                       away_we, away_elo, away_wp, 
                                                       draw_wp)
        
    R2_winners = list(r2_SF['Winner'])
    
    winners_points = []
    
    #higher ranked country hosts final
    for winner in R2_winners:
        winners_points.append(r2_table.loc[winner, 'Points'])
    
    if winners_points[0] == winners_points[1]:
        
        h_a = random.sample([0,1], 2)
        
        home = h_a[0]
        away = h_a[1]
    
    else:
        
        home = winners_points.index(max(winners_points))
        away = winners_points.index(min(winners_points))
    
    r2_F.loc[num_matches-1, 'Home'] = R2_winners[home]
    r2_F.loc[num_matches-1, 'Away'] = R2_winners[away]
    
    home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_match_We(r2_F, num_matches-1, r2_table)

    #Determine win probability for each team
    home_wp = TM.davidson_home_wp(home_we, away_we, theta)
    away_wp = TM.davidson_away_wp(home_we, away_we, theta)
    draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

    r2_table, r2_F = TM.play_ko_match(r2_table, r2_F, num_matches-1, home_team, home_we, 
                                      home_elo, home_wp, away_team, 
                                      away_we, away_elo, away_wp, draw_wp)
    
    for team in finish_table.index:
        if team in list(r2_SF['Winner']):
            total_table.loc[team, 'R2 F'] += 1
        else:
            continue
            
    r2_winner = r2_F.loc[num_matches-1,'Winner']
    
    finish_table.loc[r2_winner, 'Interconfed Playoff'] += 1
            
    quals_group_table = deepcopy(init_group_table)
    
    print(f'Running Iteration {iteration+1} / {iterations}.')
    
total_table['Avg Points'] = total_table['Points'] / iterations
total_table['Avg PPG'] = total_table['Points'] / (iterations*10)

percent_finish = placement_table / iterations
finish_table['Interconfed Playoff'] = finish_table['Interconfed Playoff'] / iterations

for team_tup in placement_table.columns: 
    finish_table.loc[team_tup[1], 'Qualify'] = round(percent_finish.loc[1, team_tup],4)
    
sorted_finish = finish_table.sort_values('Qualify', ascending=False)[:int(54/2)]
sorted_pct_fin = percent_finish.sort_values(1, axis=1, ascending=False)

fig1, ax = plt.subplots(1, 1, facecolor='white')
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=sorted_finish.values, cellColours=plt.cm.YlOrRd(sorted_finish.values),
                 rowLabels=sorted_finish.index, colLabels=sorted_finish.columns, 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(3, 5)

table_props = table.properties()
table_cells = table_props['children']
for cell in table_cells:
    if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.4')):
        cell.get_text().set_color('white')

# ab = AnnotationBbox(getImage("./country-flags-main/png1000px/ar.png"), (200, 1837), 
#                     frameon=False, xycoords='figure points')
# ax.add_artist(ab)


for val in range(0,2):
    head = table[0,val]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

for val in range(1,28):
    head = table[val,-1]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

    
fig1.savefig('./Tom Model Outputs/2026_CAF_Odds.png', bbox_inches="tight")

fig2, ax = plt.subplots(5, 2, figsize=(27,13), facecolor='white')

for val in range(0,8):
    
    group = groups[val]
    
    pct_fin_group = percent_finish[group].sort_values(1, axis=1, ascending=False)
    
    if group == "B":
        pct_fin_group = pct_fin_group.rename(columns={'Democratic_Republic_of_Congo': 'Democratic\nRepublic of Congo'})

    if group == "H":
        pct_fin_group = pct_fin_group.rename(columns={'Sao_Tome_and_Principe': 'Sao Tome\nand Principe'}) 
            
    ax[val % 4][val//4].axis('off')
    ax[val % 4][val//4].axis('tight')
    
    ax[val % 4][val//4].set_title(f"Group {group}", fontsize=18, fontweight='bold')
    
    table = ax[val % 4][val//4].table(cellText=pct_fin_group.values, 
                                     cellColours=plt.cm.YlOrRd(pct_fin_group.values),
                                     rowLabels=pct_fin_group.index, colLabels=pct_fin_group.columns, 
                                     loc='center')
    table.scale(1.1,1.3)
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    
    table_props = table.properties()
    table_cells = table_props['children']
    for cell in table_cells:
        if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.4')):
            cell.get_text().set_color('white')
    
    for val in range(1,7):
        row = table[val,-1]
        row.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')
        row.PAD = 0.4

        head = table[0,val-1]
        head.set_text_props(fontsize=9, fontweight='bold', verticalalignment='center')
        head.set_height(0.2)
        
pct_fin_group = percent_finish["I"].sort_values(1, axis=1, ascending=False)
pct_fin_group = pct_fin_group.rename(columns={'Central_African_Republic': 'Central African\nRepublic'})

ax[4][0].axis('off')
ax[4][0].axis('tight')

ax[4][0].set_title(f"Group I", fontsize=18, fontweight='bold')

table = ax[4][0].table(cellText=pct_fin_group.values, 
                                 cellColours=plt.cm.YlOrRd(pct_fin_group.values),
                                 rowLabels=pct_fin_group.index, colLabels=pct_fin_group.columns, 
                                 loc='center')
table.scale(1.1,1.3)

table.auto_set_font_size(False)
table.set_fontsize(14)

table_props = table.properties()
table_cells = table_props['children']
for cell in table_cells:
    if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.4')):
        cell.get_text().set_color('white')

for val in range(1,7):
    row = table[val,-1]
    row.set_text_props(fontsize=14, fontweight='bold', verticalalignment='center')
    row.PAD = 0.4

    head = table[0,val-1]
    head.set_text_props(fontsize=9, fontweight='bold', verticalalignment='center')
    head.set_height(0.2)
    
ax[4][1].axis('off')

fig2.savefig('./Tom Model Outputs/2026_CAF_Placement.png', bbox_inches="tight")

print('Simulation is Complete. \n')
print('Outputting 2 Tables showing your results.')