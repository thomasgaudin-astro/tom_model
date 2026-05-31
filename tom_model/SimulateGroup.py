import numpy as np
import pandas as pd

import SimulateSingleGame

class SimulateSingleGroup:
    """
        Code to randomly generate an outcome for all games in a group
        Inputs:
            matches (DataFrame) -> pandas DataFrame containing all matches played in the tournament 
                                   including the matches for the group to be simulated
            points_table (DataFrame) -> pandas DataFrame containing all groups, points earned, 
                                        and Elo ratings for each team 
            location (str) -> either 'hosted', or 'neutral'
            Km (int) -> tournament weight constant from eloratings.net 
            group (string) -> name of the group to simulate
        Outputs:
            group_stage (DataFrame) -> pandas DataFrame containing points earned and
                                       updated Elo ratings for each team in the group
            placement_table (DataFrame) -> pandas DataFrame containing counts of how often each team ends 
                                           up in each place of the group
    """

    def __init__(self, matches, points_table, location, Km, group):

        self.group_table, self.placement_table = SimulateGroup(matches, 
                                                                points_table, 
                                                                location, Km, group
                                                                )           
    
    def SimulateGroup(self, matches, points_table,  
                      location, Km, group
                      ):

        """
        Randomly generate an outcome for all games in a group
        Inputs:
            matches (DataFrame) -> pandas DataFrame containing all matches played in the tournament 
                                   including the matches for the group to be simulated
            points_table (DataFrame) -> pandas DataFrame containing all groups, points earned, 
                                        and Elo ratings for each team 
            location (str) -> either 'hosted', or 'neutral'
            Km (int) -> tournament weight constant from eloratings.net 
            group (string) -> name of the group to simulate
        Outputs:
            group_stage (DataFrame) -> pandas DataFrame containing points earned and
                                       updated Elo ratings for each team in the group
            placement_table (DataFrame) -> pandas DataFrame containing counts of how often each team ends 
                                           up in each place of the group
        """

        #create a placement table
        num_teams = len(total_group_table.index)
        placement_table = pd.DataFrame(np.zeros(num_teams, num_teams),
                                       index=range(1, num_teams+1), 
                                       columns=list(total_group_table.loc[:, 'Team'])
                                       )

        group_table = points_table[points_table['Group'] == group]
        group_schedule = matches[matches['Group'] == group]

        for match in group_schedule.index:

            #initialize home team and ELO
            home_team = group_schedule.loc[match, 'Home']
            home_elo = group_table.loc[home_team, 'Elo Rating']

            #initialize away team and ELO
            away_team = group_schedule.loc[match, 'Away']
            away_elo = group_table.loc[away_team, 'Elo Rating']

            if group_schedule.loc[match, 'Host?'] == 'Y':
                location = 'hosted'
            else:
                location = 'neutral'

            #use SimulateSingleGame class to generate all math associated with a single game
            game = SimulateSingleGame(home_elo, away_elo, location, Km)

            #update Elo ratings
            group_table.loc[home_team, 'Elo Rating'] = game.new_home_elo
            group_table.loc[away_team, 'Elo Rating'] = game.new_away_elo

            #home win
            if game.home_outcome == 1:

                #update table
                group_table.loc[home_team, 'Points'] += 3
                group_table.loc[away_team, 'Points'] += 0    

            #draw
            elif game.home_outcome == 0.5:

                #update table
                group_table.loc[home_team, 'Points'] += 1
                group_table.loc[away_team, 'Points'] += 1

            #away win
            else:

                #update table
                group_table.loc[home_team, 'Points'] += 0
                group_table.loc[away_team, 'Points'] += 3

            #rank each team in group
            group_table['Rank'] = group_table['Points'].sample(frac=1).rank(ascending=False, method='first')

            #update placement_table
            for team in group_table.index:
                team_name = group_table.loc[team, 'Team']
                team_rank = group_table.loc[team, 'Rank']

                placement_table.loc[team_rank, team_name] += 1

        return group_table, placement_table
