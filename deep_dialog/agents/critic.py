import numpy as np

class Critic:
    def __init__(self, slot_dict, max_turn=10):
        self.slots = {v: k for k, v in slot_dict.items()}
        self.max_turn = max_turn
        
    def get_reward(self, state, next_state, goal, t):
        if goal.sum() == 0:
            return -1, False
        proposed = set(state['current_slots']['request_slots'].keys()) - \
                set(next_state['current_slots']['request_slots'].keys())
        goal = self.slots[np.argmax(goal)]
        
        if goal in proposed:
            return self.max_turn * 2, True
        if t >= self.max_turn:
            return -self.max_turn, True
        return -1, False
        
