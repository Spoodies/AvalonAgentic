import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, Box
import numpy as np

class CustomGameEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "my_custom_game_v1"}

    def __init__(self):
        super().__init__()
        # 1. Define Agents
        self.agents = [f"player_{i}" for i in range(6)]
        self.possible_agents = self.agents[:]
        
        # 2. Define Spaces (The shape of inputs/outputs)
        # Example: Action is a number 0-9
        self.action_spaces = {agent: Discrete(10) for agent in self.agents} 
        # Example: Observation is a list of 100 numbers
        self.observation_spaces = {agent: Box(low=0, high=1, shape=(100,), dtype=np.float32) for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Custom Game State Init
        self.board_state = [0] * 100 
        self.vote_buffer = {}
        self.phase = "TURN_BASED"

        # PettingZoo specific: Agent Selector handles the turn order
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        # RETURN: The specific view of the board for 'agent'
        # Must match self.observation_spaces shape
        return np.array(self.board_state, dtype=np.float32)

    def step(self, action):
        # 1. Identify who is playing
        agent = self.agent_selection

        # 2. Check if they are already out of the game
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # --- YOUR CUSTOM GAME LOGIC HERE ---
        
        if self.phase == "TURN_BASED":
            # Apply move
            self.board_state[0] = action # Example logic
            self.rewards[agent] += 1     # Example reward
            
            # Pass turn to next person
            self.agent_selection = self._agent_selector.next()

        elif self.phase == "VOTE":
            self.vote_buffer[agent] = action
            
            # Logic: If this was the last person to vote
            if len(self.vote_buffer) == len(self.agents):
                # Resolve votes
                winner = max(self.vote_buffer, key=self.vote_buffer.get)
                self.rewards[winner] += 10
                self.vote_buffer = {} # Clear buffer
                self.phase = "TURN_BASED" # Switch phases
            
            # Pass turn (Still need to iterate through everyone so they can vote)
            self.agent_selection = self._agent_selector.next()

        # 3. Accumulate rewards (Required by PettingZoo)
        self._accumulate_rewards()