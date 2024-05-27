#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:07:07 2024

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

print("Running the Tom Model for CONMEBOL 2026 World Cup Qualifying. \n")

#table files
init_table_file = 'CONMEBOL_initial.csv'
current_table_file = 'CONMEBOL_current.csv'

#schedule files
init_sched_file = 'CONMEBOL_schedule_init.csv'
current_sched_file = 'CONMEBOL_schedule_current.csv'

#check for run current or initial odds
current = input('Do you want to use current Elo, Points, and Schedule? \n Y - Start from Current Spot. \n N - Start from the Beginning. \n')

if current == 'Y':
    quals_group_table = pd.read_csv(current_table_file, header=0)
    matches = pd.read_csv(current_sched_file, header=0)
    
elif current == 'N':
    quals_group_table = pd.read_csv(init_table_file, header=0)
    matches = pd.read_csv(init_sched_file, header=0)
    
else:
    print('Please Pick a Valid Response.')

#add a new index and create the initial table that will be copied to reset the tournament
quals_group_table = quals_group_table.set_index('Code')
init_group_table = deepcopy(quals_group_table)

#generate some table for all teams that are needed for analysis
#total cumulative points table
total_table = quals_group_table.loc[:, ['Points']].copy()
total_table['Points'] = 0

#table tracking the placement of each team
placement_table = pd.DataFrame(0,
                            index = np.arange(1, len(quals_group_table.index)+1),
                            columns = quals_group_table.index)

#table tracking the number of times finishing in the qualifying spots
finish_table = pd.DataFrame(0,
                            index = quals_group_table.index,
                            columns = ['Interconfed Playoff', 'Qualify'])

num_matches = len(matches['Match Num'])

#run the simulation
for iteration in range(iterations):
    for match in range(num_matches-1):
        
        #calculate win expectancies
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_match_We(matches, 
                                                                       match, quals_group_table)
        
        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

        quals_group_table = TM.play_gs_match_neutral(quals_group_table, home_team, home_we, home_elo, home_wp, 
                          away_team, away_we, away_elo, away_wp, draw_wp)
    
        #append results to the table
    for team in quals_group_table.index:
        team_points = quals_group_table.loc[team, 'Points']
        total_table.loc[team, 'Points'] += team_points
    
    #sort by points and rank teams
    ranked_group_tab = quals_group_table.sort_values(by='Points', ascending=False)
    ranked_group_tab['Rank'] = quals_group_table['Points'].rank(method='max', ascending = False)
    
    #add finishing tally to placement table
    for team in ranked_group_tab.index:
        rank = int(ranked_group_tab.loc[team, 'Rank'])
        
        placement_table.loc[rank, team] += 1
    
    #reset table for next iteration
    quals_group_table = deepcopy(init_group_table)
    
    print(f'Running Iteration {iteration+1} / {iterations}.')
    #print(f'Winner of Iteration {iteration+1}: {tourney_winner}')
    
total_table['Avg Points'] = total_table['Points'] / iterations
total_table['Avg PPG'] = total_table['Points'] / (iterations*18)

percent_finish = placement_table / iterations

#determine who qualifies and who makes the IC Playoff
for team in total_table.index:    
    finish_table.loc[team, 'Qualify'] = round(sum(percent_finish.loc[1:6, team]),4)
    finish_table.loc[team, 'Interconfed Playoff'] = round(percent_finish.loc[7, team],4)

#sort by qualification likelihood
sorted_finish = finish_table.sort_values('Qualify', ascending=False)
sorted_pct_fin = percent_finish.sort_values(1, axis=1, ascending=False)

#plot odds to advance
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

for val in range(1,11):
    head = table[val,-1]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

    
fig1.savefig('./Tom Model Outputs/2026_CONMEBOL_Odds.png', bbox_inches="tight")

#plot odds to finish in each place
fig2, ax = plt.subplots(1, 1, facecolor='white')
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

for val in range(1,11):
    head = table[val,-1]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

    
fig2.savefig('./Tom Model Outputs/2026_CONMEBOL_placements.png', bbox_inches="tight")

print('Simulation is Complete. \n')
print('Outputting 2 Tables showing your results.')
