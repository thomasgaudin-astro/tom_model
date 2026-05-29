import random
import EloMath as EM

class SimulateSingleGame:

    def __init__(self, home_Ro, away_Ro, location, Km):

        #run EloMath to get the initial win, draw, and loss probabilities
        PreMatchResults = EM(home_Ro, away_Ro, location, Km)
        self.home_wp = PreMatchResults.home_wp
        self.away_wp = PreMatchResults.away_wp
        self.draw_prob = PreMatchResults.draw_prob

        #Simulate match. Take the inverse of the outcome to assign to away team
        self.home_outcome = outcome_generator(self.home_wp, self.away_wp, self.draw_prob)
        self.away_outcome = 1 - self.home_outcome #inverse result 

        #recalculate EloMatch to account for changes in Elo
        PostMatchResults = EM(home_Ro, away_Ro, location, Km, 
                              self.home_outcome, self.away_outcome)
        
        self.new_home_elo = PostMatchResults.new_home_elo
        self.new_away_elo = PostMatchResults.new_away_elo

        
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