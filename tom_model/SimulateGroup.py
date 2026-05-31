import pandas as pd

import SimulateSingleGame

class SimulateSingleGroup:

    def __init__(self, group, location, Km, elo_rank, init_elo_rank, points_table, init_points_table, 
                       total_table, placement_table, iterations=1, count=1):

        self.group_table = SimulateGroup(matches, 
                                        points_table, 
                                        location, Km, group,
                                        iterations=1, count=1
                                        )
    
    def SimulateGroup(matches, points_table,  
                      location, Km, group,
                      iterations=1, count=1
                      ):

        total_group_table = points_table[points_table['Group'] == group]

        for num in range(iterations):

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

        total_group_table.add(group_table, fill_value=0)

        return total_group_table
