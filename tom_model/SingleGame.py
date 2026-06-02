import random
import .EloMath as EM

class SimulateSingleGame:
    """
        Code to randomly generate an outcome for a single game
        Inputs:
            home_Ro (int) -> Elo rating of home team 
            away_Ro (int) -> Elo rating of away team 
            location (str) -> either 'hosted', or 'neutral'
            Km (int) -> tournament weight constant from eloratings.net 
        Outputs:
            home_wp (float) -> home Win Probability
            away_wp (float) -> away Win Probability
            draw_prob (float) -> Draw Probability
            home_outcome (float) -> home team match outcome. 
                                    1.0 for win, 0.5 for draw, 0.0 for loss
            away_outcome (float) -> inverse value of home team outcome.
            new_home_elo (float) -> home post-match Elo
            new_away_elo (float) -> away post-match Elo
    """

    def __init__(self, home_Ro, away_Ro, location, Km):

        #run EloMath to get the initial win, draw, and loss probabilities
        PreMatchResults = EM(home_Ro, away_Ro, location, Km)
        self.home_wp = PreMatchResults.home_wp
        self.away_wp = PreMatchResults.away_wp
        self.draw_prob = PreMatchResults.draw_prob

        #Simulate match. Take the inverse of the outcome to assign to away team
        self.home_outcome, self.away_outcome = outcome_generator(self.home_wp, 
                                                                 self.away_wp, 
                                                                 self.draw_prob
                                                                 )

        #recalculate EloMatch to account for changes in Elo
        PostMatchResults = EM(home_Ro, away_Ro, location, Km, 
                              self.home_outcome, self.away_outcome)
        
        self.new_home_elo = PostMatchResults.new_home_elo
        self.new_away_elo = PostMatchResults.new_away_elo

        
    def outcome_generator(self, home_wp, away_wp, draw_wp=0):
    """ Code that simulates each game. Chooses outcome of win/draw/loss based
        on weighted random nnumber generator. Weights come from win probability
        calculations.
        Inputs:
            home_wp (float) -> Win Probability for home team
            away_wp (float) -> Win Probability for away team
            draw_wp (float) -> Probaility of a draw
        Returns:
            outcome (float) -> either 1.0 for home win, 0.5 for draw, 0.0 for home loss
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
        
        return outcome[0], 1 - outcome[0]