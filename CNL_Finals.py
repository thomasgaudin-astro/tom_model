#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:39:00 2024

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

from collections import Counter

from tabulate import tabulate

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import itertools

theta = 1.7

iterations = 10000

init_table_file = 'CNL_initial.csv'
init_sched_file = 'CNL_schedule.csv'

host = "United_States"
quals_group_table = pd.read_csv(init_table_file, header=0)

matches = pd.read_csv(init_sched_file, header=0)

quals_group_table = quals_group_table.set_index('Code')
init_group_table = deepcopy(quals_group_table)

finish_table = pd.DataFrame(0,
                            index = Counter(quals_group_table.index).keys(),
                            columns = ['Final', 'Winner'])

r1_SF = matches[matches['Group'] == 'SF']
r1_F = matches[matches['Group'] == 'F']

for iteration in range(iterations):
    
    for match in range(2):
        
        #calculate win expectancies
        home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(r1_SF, 
                                                                       match, quals_group_table, host)
        
        #Determine win probability for each team
        home_wp = TM.davidson_home_wp(home_we, away_we, theta)
        away_wp = TM.davidson_away_wp(home_we, away_we, theta)
        draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)

    
        quals_group_table, r1_SF = TM.play_ko_match_neutral(quals_group_table, r1_SF, match,
                                  home_team, home_we, home_elo, home_wp, 
                                  away_team, away_we, away_elo, away_wp, draw_wp)
        
        
    SF1_winner = r1_SF.loc[0, 'Winner']
    SF2_winner = r1_SF.loc[1, 'Winner']
    
    r1_F.loc[2, 'Home'] = SF1_winner
    r1_F.loc[2, 'Away'] = SF2_winner
    
    finish_table.loc[SF1_winner, 'Final'] += 1
    finish_table.loc[SF2_winner, 'Final'] += 1
    
    #calculate win expectancies
    home_team, home_we, home_elo, away_team, away_we, away_elo = TM.calc_neutral_match_We(r1_F, 
                                                                       2, quals_group_table, host)
    
    #Determine win probability for each team
    home_wp = TM.davidson_home_wp(home_we, away_we, theta)
    away_wp = TM.davidson_away_wp(home_we, away_we, theta)
    draw_wp = TM.davidson_tie_prob(home_we, away_we, theta)
    

    quals_group_table, r1_F = TM.play_ko_match_neutral(quals_group_table, r1_F, 2,
                              home_team, home_we, home_elo, home_wp, 
                              away_team, away_we, away_elo, away_wp, draw_wp)
    
    tourney_winner = r1_F.loc[2, 'Winner']
    finish_table.loc[tourney_winner, 'Winner'] += 1
    
    quals_group_table = deepcopy(init_group_table)
    
    print(f'Running Iteration {iteration+1} / {iterations}.')
    print(f'Winner of Iteration {iteration+1}: {tourney_winner}')
    
sorted_finish = finish_table.sort_values('Winner', ascending=False) / iterations

fig, ax = plt.subplots(facecolor='white')
ax.axis('off')
#ax.axis('tight')
table = ax.table(cellText=sorted_finish.values, cellColours=plt.cm.YlOrRd(sorted_finish.values),
                 rowLabels=sorted_finish.index, colLabels=sorted_finish.columns, 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(2, 6)

table_props = table.properties()
table_cells = table_props['children']
for cell in table_cells:
    if sum(cell.properties()['facecolor']) < sum(cl.to_rgba('0.43')):
        cell.get_text().set_color('white')

# ab = AnnotationBbox(getImage("./country-flags-main/png1000px/ar.png"), (200, 1837), 
#                     frameon=False, xycoords='figure points')
# ax.add_artist(ab)


for val in range(0,2):
    head = table[0,val]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

for val in range(1,5):
    head = table[val,-1]
    head.set_text_props(fontsize=18, fontweight='bold', verticalalignment='center')

#plt.savefig('./Tom Model Outputs/2024_CNL_Finals.pdf')
plt.savefig('./Tom Model Outputs/2024_CNL_Finals.png', bbox_inches="tight")