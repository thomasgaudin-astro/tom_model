import pandas as pd

from collections import Counter

from .SimulateGroup import SimulateSingleGroup

class SimulateGroupStage:

    def __init__(self, matches, points_table, 
                 total_table, total_placements, 
                 Km):

        #initiate
        self.groups = Counter(points_table.loc[:, 'Group']).keys()
        self.group_tables = {}
        self.placement_tables = {}

        #run SimulateGroup for all groups
        self.SimulateAllGroups(self.groups, matches, points_table, Km)

        #add results to total tables
        for group in self.groups:
            group_table = self.group_tables[group]
            placement_table = self.placement_tables[group]

            for team in group_table.index:
                total_table.loc[team, 'Points'] += group_table.loc[team, 'Points']
                rank = int(group_table.loc[team, 'Rank'])
                total_placements[group].loc[rank, team] += 1

        self.total_table = total_table
        self.total_placements = total_placements
    
    def SimulateAllGroups(self, groups, matches, points_table, Km):

        #for each group, simulate all games and append to game dictionary
        for group in groups:

            group_results = SimulateSingleGroup(matches, points_table, Km, group)

            self.group_tables[group] = group_results.group_table
            self.placement_tables[group] = group_results.placement_table

    

        


            

