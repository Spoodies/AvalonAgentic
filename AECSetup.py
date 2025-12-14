import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, Box, Dict, MultiBinary, MultiDiscrete, Tuple
import random
import numpy as np

# =============================================================================
# AECEnv = Agent-Environment-Cycle Environment
# This is PettingZoo's turn-based multi-agent environment class
# Each agent takes turns acting, even during "simultaneous" actions like voting
# =============================================================================

class CustomGameEnv(AECEnv):
    """
    This class defines HOW THE GAME WORKS.
    It's the "rules engine" that agents will interact with.
    
    Key Concept: This is NOT the agent. This is the WORLD the agents play in.
    Think of it like a chess board - it enforces rules, tracks state, but doesn't play.
    """
    
    metadata = {"render_modes": ["human"], "name": "my_custom_game_v1"}

    def __init__(self):
        """
        Called ONCE when you create the environment: env = CustomGameEnv()
        
        PURPOSE: Define the structure of the game (not the state, just the rules)
        - How many agents?
        - What can they do? (action space)
        - What can they see? (observation space)
        
        This is like defining the board size and piece types in chess,
        not actually setting up the pieces (that's done in reset()).
        """
        super().__init__()
        
        # 1. Define Agents
        # For Avalon: typically 5-10 players, but we'll start with 6
        # NOTE: These are agent NAMES, not their roles (roles assigned in reset())
        self.agents = ["Merlin", "Morgana", "Percival", "Assassin", "Generic 1", "Generic 2"]
        self.possible_agents = self.agents[:]  # PettingZoo requirement: list of ALL possible agents
        self.max_num_agents = len(self.agents)
        
        self.action_spaces = { agent: Tuple([
                Discrete(4),      # 0=propose, 1=vote, 2=quest, 3=assassinate
                MultiBinary(6),    # Team selection for mission
                Discrete(2),      # Voting or failing quest. 0 or 1.
                Discrete(6)       # Target player for assassination.
            ]) for agent in self.agents
        }

        # 3. Define OBSERVATION SPACE (What can agents SEE?)
        # OUTPUT TYPE: observe(agent) must return data matching this space
        self.observation_spaces = {agent: Dict({
            "known_spies": MultiBinary(6),                                     # Only for Merlin or Evil.
            "percival_view": MultiBinary(6),                                   # Only for Percival.
            "all_prior_votes": MultiBinary(150),                               # History of all votes x 5 missions x 5 attempts each
            "who_choose_prior_mission": MultiDiscrete([6]*25),                 # 6 people for 5 missions x 5 attempts
            "number_of_fails_per_prior_missions": MultiDiscrete([3]*5),        # 0-2 fails per mission
            "whos_winning": MultiDiscrete([3]*5),                              # 0=evil won, 1=not played, 2=good won
            "current_proposed_team": MultiBinary(6),                           # [1,0,1,0,0,0] = players 0&2 on team
            "who_choose_current_mission": MultiBinary(6),                      # Who is on the current mission team
            "did_i_choose_current_team": Discrete(2),                          # 1=yes, 0=no
            "mission_number": Discrete(6),                                     # Current mission (1-5)
            "reject_count": Discrete(5)                                        # How many consecutive rejections (0-4)
        }) for agent in self.agents}

    def reset(self, seed=None, options=None):
        """Start a new game. Assigns roles, resets counters, determines turn order."""
        
        # PettingZoo required initialization
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Avalon game state
        self.roles = {}  # TODO: Assign roles randomly
        self.good_team = []
        self.evil_team = []
        self.mission_number = 1
        self.mission_results = []  # List of True/False for each mission (order matters)
        self.mission_fail_counts = []  # List of integers: how many fails per mission
        self.leader_index = 0
        self.proposed_team = []
        self.current_mission_team = []
        self.reject_count = 0
        
        # Vote tracking
        self.all_votes_history = []  # Stores all votes across all missions
        self.who_chose_history = []  # Track who was leader for each proposal
        self.vote_buffer = {}
        self.quest_buffer = {}
        
        # Phase management
        self.phase = "TEAM_PROPOSAL"
        
        # Turn order
        random.shuffle(self.agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        """Return what this agent can see. Different agents see different info (partial observability)."""
        
        # Known spies (only if I'm Merlin or Evil)
        known_spies = np.zeros(6, dtype=np.int8)
        if agent in self.evil_team or agent == "Merlin":
            for i, other_agent in enumerate(self.agents):
                if other_agent in self.evil_team:
                    known_spies[i] = 1
        
        # Percival sees Merlin + Morgana (can't distinguish)
        percival_view = np.zeros(6, dtype=np.int8)
        if agent == "Percival":
            for i, other_agent in enumerate(self.agents):
                if other_agent in ["Merlin", "Morgana"]:
                    percival_view[i] = 1
        
        # All prior votes (flatten vote history)
        all_prior_votes = np.zeros(150, dtype=np.int8)
        for i, vote in enumerate(self.all_votes_history[:150]):
            all_prior_votes[i] = vote
        
        # Who chose prior missions
        who_choose_prior = np.zeros(25, dtype=np.int8)
        for i, leader_idx in enumerate(self.who_chose_history[:25]):
            who_choose_prior[i] = leader_idx
        
        # Number of fails per mission (0-2 fails possible)
        fails_per_mission = np.zeros(5, dtype=np.int8)
        for i, fail_count in enumerate(self.mission_fail_counts):
            fails_per_mission[i] = fail_count
        
        # Who's winning per mission (0=evil won, 1=not played, 2=good won)
        whos_winning = np.ones(5, dtype=np.int8)  # Default to 1 (not played)
        for i, result in enumerate(self.mission_results):
            whos_winning[i] = 2 if result else 0  # True=good won (2), False=evil won (0)
        
        # Current proposed team
        current_team = np.zeros(6, dtype=np.int8)
        for i, team_agent in enumerate(self.agents):
            if team_agent in self.proposed_team:
                current_team[i] = 1
        
        # Who is on current mission
        who_on_mission = np.zeros(6, dtype=np.int8)
        for i, team_agent in enumerate(self.agents):
            if team_agent in self.current_mission_team:
                who_on_mission[i] = 1
        
        return {
            "known_spies": known_spies,
            "percival_view": percival_view,
            "all_prior_votes": all_prior_votes,
            "who_choose_prior_mission": who_choose_prior,
            "number_of_fails_per_prior_missions": fails_per_mission,
            "whos_winning": whos_winning,
            "current_proposed_team": current_team,
            "who_choose_current_mission": who_on_mission,
            "did_i_choose_current_team": 1 if self.agents[self.leader_index] == agent else 0,
            "mission_number": self.mission_number,
            "reject_count": self.reject_count
        }
    
    def step(self, action):
        """Process agent action. Core Avalon game logic."""
        
        agent = self.agent_selection
        
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        
        # Unpack action tuple
        action_type, team_selection, vote_or_quest, target_player = action
        
        # TODO: Implement Avalon phases
        # Phase: TEAM_PROPOSAL
        #   - Only leader acts (check if agent == self.agents[self.leader_index])
        #   - Extract team from team_selection binary array
        #   - Set self.proposed_team
        #   - Transition to TEAM_VOTE phase
        
        # Phase: TEAM_VOTE
        #   - Collect vote_or_quest (0=reject, 1=approve) in self.vote_buffer
        #   - When all voted, count approvals
        #   - If approved: set self.current_mission_team, transition to QUEST
        #   - If rejected: increment self.reject_count, rotate leader, back to TEAM_PROPOSAL
        #   - If self.reject_count == 5: evil wins (set terminations)
        #   - Record votes to self.all_votes_history
        
        # Phase: QUEST
        #   - Only mission team members act
        #   - Collect vote_or_quest (0=fail, 1=success) in self.quest_buffer
        #   - When all played, count fails
        #   - Determine mission success (0 fails = success, except mission 4 with 7+ players needs <=1 fail)
        #   - Append to self.mission_results and self.mission_fail_counts
        #   - Check if 3 missions passed or failed
        #   - If good wins 3: transition to ASSASSINATION
        #   - If evil wins 3: game over (set terminations, assign rewards)
        #   - Otherwise: next mission, rotate leader, back to TEAM_PROPOSAL
        
        # Phase: ASSASSINATION
        #   - Only assassin acts
        #   - Check if target_player is Merlin
        #   - If Merlin killed: evil wins
        #   - If wrong target: good wins
        #   - Set terminations, assign rewards
        
        # Win condition rewards:
        #   - Good wins: self.rewards[agent] = 1 for good_team, -1 for evil_team
        #   - Evil wins: self.rewards[agent] = -1 for good_team, 1 for evil_team
        
        self._accumulate_rewards()