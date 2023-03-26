#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:46:57 2023

@author: thomasgaudin
"""

import csv, random
import numpy as np
import pandas as pd
import re
import requests

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

from bs4 import BeautifulSoup
from time import sleep
from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#############################
# Functions for Running the Tom Model for any tournament
#############################

#webscraping function

def get_Elo_ranks():
    """Scrapes Elo Ratings to pull most current ratings values.
       Returns a Pandas dataframe of Elo ranks for all nations."""
    
    #define webdriver. For me on Mac, this is Safari. Could be different for others.
    driver = webdriver.Safari()
    
    #Using this website for ratings
    url = "https://www.eloratings.net"
    
    driver.get(url)
    
    #Rewrite the html as lxml
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    #Find all cells of the table containing a country
    countries = soup.find_all("div",{"class":"slick-cell l1 r1 team-cell narrow-layout"})
    
    nations = []
    
    #Pull all country names and append to nations list
    for nation in countries:
        country = re.findall("handleLink\(\'([a-zA-Z\_\-]+)\'\)", str(nation))
        nations.append(country[0])
    
    #Find all cells of the table containing an Elo ranking value
    rankings = soup.find_all("div",{"class":"slick-cell l2 r2 rating-cell narrow-layout"})
    
    ranks = []
    
    #Pull all rating values and append to rankings list
    for rank in rankings:
        ranking = re.findall('">(\d*)</div>', str(rank))
        ranks.append(ranking[0])
    
    #create a dictionary out of the zipped lists and turn it into a pandas dataframe
    elo = dict()
    
    for nation, rank in zip(nations, ranks):
        elo[nation] = rank
        
    elos = pd.DataFrame(elo.items(), 
                   index = np.arange(1, len(nations)+1),
                   columns = ['Team', 'Points'])
    
    return elos

#Elo Calculation functions

def calculate_We(Ro, opponent_Ro, location):
    """ Calculate the We from the formula given by ELO.
        Inputs:
            Ro - Elo rating of team (int)
            opponent_Ro - Elo rating of opponent (int)
            location - either the string 'home' or 'away' 
        Returns:
            Win Expectancy, type: float
    """
    
    #if the team is at home, calcuate difference in Elo with home boost
    if location == 'home':
        dr = (Ro + 100) - opponent_Ro
        
    #if the team is on the road, calculate difference in Elo, boost for oppoent
    elif location == 'away':
        dr = Ro - (opponent_Ro + 100)
        
    #formula from eloratings.net
    We = 1 / ( (10 ** (-dr / 400)) + 1)

    return We

def calculate_neutral_We(Ro, opponent_Ro):
    """ Calculate the We from the formula given by ELO assuming neutral site.
        Inputs:
            Ro - Elo rating of team (int)
            opponent_Ro - Elo rating of opponent (int)
        Returns:
            Win Expectancy, type: float
    """
    
    #calculate difference in Elo
    dr = Ro - opponent_Ro
    
    #formula from eloratings.net
    We = 1 / ( (10 ** (-dr / 400)) + 1)

    return We

def davidson_home_wp(home_We, away_We, theta=1.7):
    """ Calculates the probability of a win for any team given the win expectancy
        calculated from the difference in Elo for each team. Formula given by 
        Davidson (1970).
        Inputs:
            home_We - win expectancy for home team (float)
            away_We - win expectancy for away team (float)
            theta - float fudge factor to make equations work, I have found 
                    that 1.7 gives realistic results
        Returns:
            Win Probability, type: float
    """
    
    hwp = home_We / (home_We + (theta * away_We) )
    
    return hwp

def davidson_away_wp(home_We, away_We, theta):
    """ Calculates the probability of a loss for any team given the win expectancy
        calculated from the difference in Elo for each team. Formula given by 
        Davidson (1970).
        Inputs:
            home_We - win expectancy for home team (float)
            away_We - win expectancy for away team (float)
            theta - float fudge factor to make equations work, I have found 
                    that 1.7 gives realistic results
        Returns:
            Loss Probability, type: float
    """
    
    awp = away_We / ( (theta * home_We) + away_We)
    
    return awp

def davidson_tie_prob(home_We, away_We, theta):
    """ Calculates the probability of a draw for any match given the win expectancy
        calculated from the difference in Elo for each team. Formula given by 
        Davidson (1970).
        Inputs:
            home_We - win expectancy for home team (float)
            away_We - win expectancy for away team (float)
            theta - float fudge factor to make equations work, I have found 
                    that 1.7 gives realistic results
        Returns:
            Draw Probability, type: float
    """
    
    tie = ( (theta**2 - 1) * home_We * away_We ) / ((home_We + (theta * away_We) ) * ( (theta * home_We) + away_We))
    
    return tie

def calculate_home_win_probability(home_Ro, away_Ro):
    """ Win probability formula from Elo ratings website. 
        Can be used, but doesn't work well."""

    wp = min((1 / (1 + 10**((away_Ro - home_Ro)/400)))**1.75 + 0.1, 1)

    return wp

def calculate_away_win_probability(home_Ro, away_Ro):
    """ Loss probability formula from Elo ratings website. 
        Can be used, but doesn't work well."""

    wp = max((1 / (1 + 10**((home_Ro - away_Ro)/400)))**1.75 - 0.1, 0)

    return wp

def calculate_elo(Ro, We, WLD, Km, GD=1):
    """ ELO formula used for calculation of new Elo after a match.
        Can calculate real Elo if your code simulates goals scored.
        Inputs:
            Ro - pre-match Elo (int)
            We - team win expectancy for match (float)
            WDL - Determined by the outcome generater function
                  1.0 for win, 0.5 for draw, 0.0 for loss (float)
            GD - Goal Difference of match, defaults to 1 (int)
            Km - tournament weight constant from eloratings.net (int)
        Returns:
            Post-match Elo, type: float
    """
    
    #Adjust weight constant based on match GD
    if GD < 2:
        GDM = Km

    elif GD == 2:
        GDM = (1.5 * Km)

    elif GD == 3:
        GDM = (1.75 * Km)

    elif GD >= 4:
        GDM = (Km * (1.75 + (GD - 3) / 8 ))
        
    #calculate new Elo
    Rn = Ro + (GDM * (WLD - We))

    return Rn

def print_probabilities(matches, elo_rank):
    """ Prints the pre-tournament win probability for every match in a tournament 
        given pre-tournament Elo ratings.
        Inputs:
            matches - List of lists for every match in the tournament
                      home team is listed first in each sub-list
            elo_rank - Dataframe of all team Elo ratings
            host - Name of the host nation for neutral tournament
        Returns:
            Nothing
    """
    
    for match in matches:

        #initialize home team and ELO
        home_team = match[0]
        home_elo = elo_rank[home_team]

        #initialize away team and ELO
        away_team = match[1]
        away_elo = elo_rank[away_team]

        #calculate We for new ELO calc
        home_we = calculate_We(home_elo, away_elo, 'home')
        away_we = calculate_We(away_elo, home_elo, 'away')

        #Determine win probability for each team
        home_wp = davidson_home_wp(home_we, away_we)
        away_wp = davidson_away_wp(home_we, away_we)
        draw_wp = davidson_tie_prob(home_we, away_we)

        print(f'{home_team} / draw / {away_team}')
        print(f'{round(home_wp,2)} / {round(draw_wp,2)} / {round(away_wp,2)}')

def print_neutral_probabilities(matches, elo_rank, host):
    """ Prints the pre-tournament win probability for every match in a neutral 
        site tournament given pre-tournament Elo ratings.
        Inputs:
            matches - List of lists for every match in the tournament
            elo_rank - Dataframe of all team Elo ratings
            host - Name of the host nation for neutral tournament
        Returns:
            Nothing
    """
    
    for match in matches:
        
        if host in match:
            
            #initialize home team and ELO
            home_team = host
            home_elo = int(elo_rank.loc[home_team]['Points'])
            
            #initialize away team and ELO
            if match[0] == host:
                away_team = match[1]
                
            elif match[1] == host:
                away_team = match[0]
                
            else:
                continue
                
            away_elo = int(elo_rank.loc[away_team]['Points'])
            
            #calculate We for new ELO calc using neutral site
            home_we = calculate_We(home_elo, away_elo, 'home')
            away_we = calculate_We(away_elo, home_elo, 'away')
            
        else:
            
            #initialize home team and ELO
            home_team = match[0]
            home_elo = int(elo_rank.loc[home_team]['Points'])

            #initialize away team and ELO
            away_team = match[1]
            away_elo = int(elo_rank.loc[away_team]['Points'])

            #calculate We for new ELO calc using neutral site
            home_we = calculate_neutral_We(home_elo, away_elo)
            away_we = calculate_neutral_We(away_elo, home_elo)

        #Determine win probability for each team
        home_wp = davidson_home_wp(home_we, away_we)
        away_wp = davidson_away_wp(home_we, away_we)
        draw_wp = davidson_tie_prob(home_we, away_we)

        print(f'{home_team} / draw / {away_team}')
        print(f'{round(home_we,2)} /    / {round(away_we,2)}')
        print(f'{round(home_wp,2)} / {round(draw_wp,2)} / {round(away_wp,2)}')
        
def outcome_generator(home_wp, away_wp, draw_wp=0):
    """ Code that simulates each game. Chooses outcome of win/draw/loss based
        on weighted random nnumber generator. Weights come from win probability
        calculations.
        Inputs:
            home_wp - Win Probability for home team (float)
            away_wp - Win Probability for away team (float)
            draw_wp - Probaility of a draw (float)
        Returns:
            outcome - either 1.0 for home win, 0.5 for draw, 0.0 for home loss
                      Type: float
    """
    
    #sort weights, outcomes dict: win = 1, draw = 0.5, loss = 0.0
    weights = {1.0: home_wp, 0.5: draw_wp, 0.0: away_wp}
    sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}

    #print(sorted_weights)

    weights_list = []

    outcomes = []
    probabilities = []

    for weight in sorted_weights.keys():
        weights_list.append((weight, sorted_weights[weight]))

    for outcome in weights_list:
        outcomes.append(outcome[0])

    for probability in weights_list:
        probabilities.append(probability[1])

    #choose a random outcome 
    outcome = random.choices(outcomes, weights=probabilities, k=1)

    #print(outcomes)
    #print(probabilities)
    #print(outcome)
    
    return outcome[0]

def simulate_group(matches, elo_rank, init_elo_rank, points_table, init_points_table, 
                   total_table, placement_table, iterations=1, count=1):
    
    #There are two ways to write country names. This is converter code between the two
    list_teams = [points_table['Team'][ind] for ind in points_table.index]
    #teams_list = [points_table.index[val][1] for val in range(len(points_table.index))]
            
    #teams_dict = {teams_list[i]: list_teams[i] for i in range(len(list_teams))}

    for num in range(iterations):

        for match in matches:

            #initialize home team and ELO
            home_team = match[0]
            home_elo = elo_rank[home_team]

            #initialize away team and ELO
            away_team = match[1]
            away_elo = elo_rank[away_team]

            #calculate We for new ELO calc
            home_we = calculate_We(home_elo, away_elo, 'home')
            away_we = calculate_We(away_elo, home_elo, 'away')

            #Determine win probability for each team
            home_wp = davidson_home_wp(home_we, away_we)
            away_wp = davidson_away_wp(home_we, away_we)
            draw_wp = davidson_tie_prob(home_we, away_we)

            #randomly determine match outcome using weighted probabilities
            outcome = outcome_generator(home_wp, away_wp, draw_wp)

            #home win
            if outcome == 1:

                #update table
                points_table.loc[points_table['Team'] == home_team, ['Points']] += 3
                points_table.loc[points_table['Team'] == away_team, ['Points']] += 0

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome, 1, 50)

                elo_rank[home_team] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, 0, 1, 50)

                elo_rank[away_team] = new_away_elo

            elif outcome == 0.5:

                #update table
                points_table.loc[points_table['Team'] == home_team, ['Points']] += 1
                points_table.loc[points_table['Team'] == away_team, ['Points']] += 1

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome, 0, 50)

                elo_rank[home_team] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome, 0, 50)

                elo_rank[away_team] = new_away_elo

            #away win
            else:

                #update table
                points_table.loc[points_table['Team'] == home_team, ['Points']] += 0
                points_table.loc[points_table['Team'] == away_team, ['Points']] += 3

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, 0, 1, 50)

                elo_rank[home_team] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome, 1, 50)

                elo_rank[away_team] = new_away_elo


        #create final table, append to total table, and reset to initial table
        final_table = deepcopy(points_table)

        for team in list_teams:
            final_points = final_table.loc[final_table['Team'] == team, 'Points'].values[0]
            total_table[team].append(final_points)

        points_table = deepcopy(init_points_table)

        #reset ELO rankings
        for team in elo_rank.keys():
            elo_rank[team] = deepcopy(init_elo_rank[team])

        #rank final table and append to placement table
        final_table['Rank'] = final_table['Points'].rank(ascending = False)
        for team in list_teams:
            rank = int(final_table.loc[final_table['Team'] == team, ['Rank']].values[0])
            placement_table.loc[rank, team] += 1

        print(count)
        count += 1
    
    average_final_table = deepcopy(final_table)
    
    #create average points gained table
    for team in list_teams:
        total_points = sum(total_table[team])

        average_final_table[team] = round(total_points / iterations, 1)

    avg_final_table = pd.DataFrame(average_final_table.items(),
                                   index = range(1, len(average_final_table.index)+1),
                                   columns = ['Team', 'Points'])
    
    final_tab = avg_final_table.sort_values('Points', ascending = False)

    #create table showing the percent chance to finish in each spot
    percent_finish = (1 / iterations) * placement_table
    
    #add average ppg column to final table
    final_tab['Average PPG'] = 0
    
    #calculate average ppg
    for team in list_teams:
        final_tab.loc[final_tab['Team'] == team, ['Average PPG']] = final_tab.loc[final_tab['Team'] == team, 
                                                                              ['Points']].values[0] / 14
        
    #add chance to qualify automatically tab to final table
    final_tab['Chance to Qualify'] = 0

    #calculate percent qualify and append to average points table
    for column in percent_finish:
        percent_qualify = sum(percent_finish.loc[0:3, column].values)

        final_tab.loc[final_tab['Team'] == column, ['Chance to Qualify']] = percent_qualify
    
    return percent_finish, final_tab

def simulate_neutral_groups(matches, groups, elo_rank, init_elo_rank, points_table, init_points_table, 
                   total_table, placement_table, host, iterations=1, count=1):
    
    list_teams = [points_table['Team'][ind] for ind in points_table.index]
    teams_list = [points_table.index[val][1] for val in range(len(points_table.index))]
            
    teams_dict = {teams_list[i]: list_teams[i] for i in range(len(list_teams))}
            
    #print(list_teams)
    #print(teams_list)
        
    num_games = (len(list_teams) / len(groups)) - 1
        
    for num in range(iterations):

        for match in matches:
            
            if host in match:
            
                #initialize home team and ELO
                home_team = host
                home_elo = int(elo_rank.loc[home_team]['Points'])
            
                #initialize away team and ELO
                if match[0] == host:
                    away_team = match[1]
                
                elif match[1] == host:
                    away_team = match[0]
                
                away_elo = int(elo_rank.loc[away_team]['Points'])
                
                #calculate We for new ELO calc
                home_we = calculate_We(home_elo, away_elo, 'home')
                away_we = calculate_We(away_elo, home_elo, 'away')
                
            else:

                #initialize home team and ELO
                home_team = match[0]
                home_elo = int(elo_rank.loc[home_team]['Points'])

                #initialize away team and ELO
                away_team = match[1]
                away_elo = int(elo_rank.loc[away_team]['Points'])

                #calculate We for new ELO calc
                home_we = calculate_neutral_We(home_elo, away_elo)
                away_we = calculate_neutral_We(away_elo, home_elo)

            #Determine win probability for each team
            home_wp = davidson_home_wp(home_we, away_we)
            away_wp = davidson_away_wp(home_we, away_we)
            draw_wp = davidson_tie_prob(home_we, away_we)

            #randomly determine match outcome using weighted probabilities
            outcome = outcome_generator(home_wp, away_wp, draw_wp)

            #home win
            if outcome == 1:

                #update table
                points_table.loc[points_table.index.get_level_values("Code") == home_team, ['Points']] += 3
                points_table.loc[points_table.index.get_level_values("Code") == away_team, ['Points']] += 0

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome, 1, 50)

                elo_rank.loc[home_team]['Points'] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, 0, 1, 50)

                elo_rank.loc[away_team]['Points'] = new_away_elo

            elif outcome == 0.5:

                #update table
                points_table.loc[points_table.index.get_level_values("Code") == home_team, ['Points']] += 1
                points_table.loc[points_table.index.get_level_values("Code") == away_team, ['Points']] += 1

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome, 0, 50)

                elo_rank.loc[home_team]['Points'] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome, 0, 50)

                elo_rank.loc[away_team]['Points'] = new_away_elo

            #away win
            else:

                #update table
                points_table.loc[points_table.index.get_level_values("Code") == home_team, ['Points']] += 0
                points_table.loc[points_table.index.get_level_values("Code") == away_team, ['Points']] += 3

                #new home elo
                new_home_elo = calculate_elo(home_elo, away_elo, home_we, 0, 1, 50)

                elo_rank.loc[home_team]['Points'] = new_home_elo

                #new away elo
                new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome, 1, 50)

                elo_rank.loc[away_team]['Points'] = new_away_elo


        #create final table, append to total table, and reset to initial table
        final_table = deepcopy(points_table)
        
        #print(final_table)

        for team in list_teams:
            final_points = final_table.loc[final_table['Team'] == team, 'Points'].values[0]
            group = final_table.loc[final_table['Team'] == team].index[0][0]
            code = final_table.loc[final_table['Team'] == team].index[0][1]
            total_table[code].append(final_points)

        points_table = deepcopy(init_points_table)
        
        #reset ELO rankings
        for team in elo_rank.index:
            elo_rank.loc[team]['Points'] = deepcopy(init_elo_rank.loc[team]['Points'])

        #rank final table and append to placement table
        final_table['Rank'] = final_table.groupby(level=0)['Points'].rank(ascending = False)
        for team in list_teams:
            rank = int(final_table.loc[final_table['Team'] == team, ['Rank']].values[0])
            group = final_table.loc[final_table['Team'] == team].index[0][0]
            code = final_table.loc[final_table['Team'] == team].index[0][1]
            placement_table.loc[rank, (group, code)] += 1

        print(count)
        count += 1
        
    average_final_table = deepcopy(final_table)
    
    #print(average_final_table)

    #create average points gained table, sort by largest points per group
    for team in teams_list:
        total_points = sum(total_table[team])
        
        average_final_table.loc[average_final_table["Team"] == teams_dict[team], ['Points']] = total_points / iterations
        
    print(average_final_table)
    
    final_tab = average_final_table.sort_values(['Group','Points'], ascending=[True,False])
    
    final_tab['Rank'] = final_tab.groupby(level=0)['Points'].rank(ascending = False)
    
    #print(average_final_table)
    #print(final_tab)

    #create table showing the percent chance to finish in each spot
    percent_finish = (1 / iterations) * placement_table
    
    #add average ppg column to final table
    final_tab['Average PPG'] = 0
    
    #calculate average ppg
    for team in list_teams:
        print(team)
        final_tab.loc[final_tab['Team'] == team, ['Average PPG']] = final_tab.loc[final_tab['Team'] == team, 
                                                                              ['Points']].values[0] / num_games
        
    #add chance to qualify automatically tab to final table
    final_tab['Chance to Qualify'] = 0

    #calculate percent qualify and append to average points table
    for column in percent_finish:
        percent_qualify = sum(percent_finish.loc[0:2, column].values)

        final_tab.loc[final_tab.index.get_level_values("Code") == column[1], ['Chance to Qualify']] = percent_qualify
    
    return percent_finish, final_tab