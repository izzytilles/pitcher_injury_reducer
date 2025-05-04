import heapq

class PitcherState:
    def __init__(self, avg_pitches, pitch_breakdown, velo_dict, spin_dict, games_played, cumulative_risk):
        self.avg_pitches = avg_pitches  # average number of pitches thrown per game
        self.pitch_breakdown = pitch_breakdown  # dict of what % of the time a pitcher throws each pitch
        self.velo_dict = velo_dict # dict of the avg velocity for each pitch
        self.spin_dict = spin_dict # dict of the avg spin rate for each pitch
        self.games_played = games_played  # number of games played
        self.cumulative_risk = cumulative_risk  # injury risk accumulated
    
    def __lt__(self, other):
        return self.cumulative_risk < other.cumulative_risk # less-than definition

    def __str__(self):
        return (
            f"Optimal State for the rest of the season:\n"
            f"  Avg Pitches/Game: {self.avg_pitches:.2f}\n"
            f"  Games Played: {self.games_played}\n"
            f"  Cumulative Risk: {self.cumulative_risk:.3f}\n"
            # f"  Pitch Usage (%): {self.pitch_breakdown}\n"
            # f"  Velocity (MPH): {self.velo_dict}\n"
            # f"  Spin Rate (RPM): {self.spin_dict}\n"
        )

    def __repr__(self):
        return self.__str__()

def workload_risk(avg_pitches, games_played):
    max_pitches = 100
    return (avg_pitches / max_pitches) * games_played

def stress_factor(pitch_pct_dict, velo_dict, spin_dict):
    max_velo_dict = {"4-seam": 96.9, "slider": 90.6, "changeup": 89.9, "curve": 83.2, "sinker": 97.0, "cutter": 95.3, "splitter": 89.6}
    max_spin_dict = {"4-seam": 2557, "slider": 2982, "changeup": 2477, "curve": 3285, "sinker": 2968, "cutter": 2742, "splitter": 2007}
    return sum(pitch_pct_dict[key] * ((velo_dict[key] / max_velo_dict[key]) + (spin_dict[key] / max_spin_dict[key])) for key in pitch_pct_dict if key in max_velo_dict)

def age_penalty(age):
    ideal_age = 25
    return 1 + (age - ideal_age)/10  # older pitchers have slightly higher risk

def cost_function(state, age):
    return (C1 * workload_risk(state.avg_pitches, state.games_played) +
            C2 * stress_factor(state.pitch_breakdown, state.velo_dict, state.spin_dict) +
            C3 * age_penalty(age))

def heuristic_function(remaining_games, avg_pitches, max_games, age):
    max_workload_risk = workload_risk(avg_pitches, max_games)  # Assumed max workload risk (tunable)
    max_stress_factor = stress_factor({key: 1/7 for key in ["4-seam", "slider", "changeup", "curve", "sinker", "cutter", "splitter"]},
                                      {key: 100 for key in ["4-seam", "slider", "changeup", "curve", "sinker", "cutter", "splitter"]},
                                      {key: 3000 for key in ["4-seam", "slider", "changeup", "curve", "sinker", "cutter", "splitter"]})
    return (remaining_games / max_games) * (2 * max_workload_risk + max_stress_factor) * age_penalty(age)

def a_star_search(start_state, age, max_games):
    open_set = []
    heapq.heappush(open_set, (0, start_state))
    
    while open_set:
        _, current_state = heapq.heappop(open_set)
        
        if current_state.games_played >= max_games:
            return current_state  # reached end of season

        for pitch_count in [50, 75, 100]:  # different workload choices
            new_avg_pitches = (current_state.avg_pitches * current_state.games_played + pitch_count) / (current_state.games_played + 1)
            new_risk = cost_function(current_state, age)
            new_state = PitcherState(new_avg_pitches, current_state.pitch_breakdown, current_state.velo_dict, current_state.spin_dict, current_state.games_played + 1, new_risk)

            f_score = new_risk + heuristic_function(max_games - new_state.games_played, start_state.avg_pitches, max_games, age)
            heapq.heappush(open_set, (f_score, new_state))

    return None  # no optimal path found

# constants 
# TODO: train
C1, C2, C3 = 1.0, 2.0, 0.5

# example usage
logan_webb = PitcherState(97.03, 
    {"4-seam": 5.4, "slider": 0, "changeup": 30.9, "curve": 0, "sinker": 39.8, "cutter": 2.6, "splitter": 0},
    {"4-seam": 92.6, "slider": 0, "changeup": 87.4, "curve": 0, "sinker": 92.6, "cutter": 91, "splitter": 0},
    {"4-seam": 2061, "slider": 0, "changeup": 1460, "curve": 0, "sinker": 1939, "cutter": 2236, "splitter": 0},
    20, 0)

optimal_state = a_star_search(logan_webb, age=27, max_games=30)
print(optimal_state)
