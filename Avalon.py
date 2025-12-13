class Avalon:
    def __init__(self):
        self.agents = [f"player_{i}" for i in range(6)]
        self.state = {} # Your board data
        self.vote_buffer = {} # For the simultaneous phase

    def reset(self):
        self.state = "FRESH_BOARD"
        self.turn_order = iter(self.agents) # Simple Python iterator
        return self.get_observation(self.agents[0])

    def step(self, action, current_agent):
        # 1. Logic for Turn-Based Phase
        if self.phase == "TURN":
            self.update_board(action)
        
        # 2. Logic for Voting Phase (Simultaneous)
        elif self.phase == "VOTE":
            self.vote_buffer[current_agent] = action
            if len(self.vote_buffer) == len(self.agents):
                self.resolve_votes()
        
        # 3. Get next player and return info
        try:
            next_agent = next(self.turn_order)
        except StopIteration:
            self.turn_order = iter(self.agents) # Loop back or start new round
            next_agent = next(self.turn_order)

        return self.get_observation(next_agent), self.get_reward(current_agent), False

    def get_observation(self, agent):
        # Return the specific view for this agent (1s and 0s)
        return [0, 1, 0...]