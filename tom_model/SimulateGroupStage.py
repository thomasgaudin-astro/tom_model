import pandas as pd

from copy import deepcopy

from collections import Counter

from .SimulateGroup import SimulateSingleGroup

class SimulateGroupStage:

    def __init__(self, matches, points_table, 
                 total_table, total_placements, 
                 Km, third='N'):

        #initiate
        self.groups = Counter(points_table.loc[:, 'Group']).keys()
        self.group_tables = {}
        self.placement_tables = {}
        if third == 'Y':
            self.third_place = pd.DataFrame()

        #run SimulateGroup for all groups
        self.group_tables, self.placement_tables = self.SimulateAllGroups(self.groups, 
                                                                          self.group_tables,
                                                                          self.placement_tables,
                                                                          matches, 
                                                                          points_table, 
                                                                          Km
                                                                          )

        #add results to total tables
        for group in self.groups:
            group_table = self.group_tables[group]
            placement_table = self.placement_tables[group]

            for team in group_table.index:
                total_table.loc[team, 'Points'] += group_table.loc[team, 'Points']
                rank = int(group_table.loc[team, 'Rank'])
                total_placements[group].loc[rank, team] += 1

                if (third=='Y') & (rank == 3):
                    third_place_info = deepcopy(group_table.loc[team, ['Team', 'Points', 'Group']])
                    self.third_place = pd.concat([self.third_place, third_place_info])

        self.total_table = total_table
        self.total_placements = total_placements
    
    def SimulateAllGroups(self, groups, group_tables, placement_tables, matches, points_table, Km):

        #for each group, simulate all games and append to game dictionary
        for group in groups:

            group_results = SimulateSingleGroup(matches, points_table, Km, group)

            group_tables[group] = group_results.group_table
            placement_tables[group] = group_results.placement_table

        return group_tables, placement_tables
            


    

        


            

