import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, Box, Dict, MultiBinary, MultiDiscrete, Tuple
import random
import numpy as np

# TODO: Improvments: Create a dicussion phase where agents can "discuss" probabilites and
# Such mimic human players discussing during the game.

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
        # Agent names ARE their roles - each agent specializes in one role
        self.agents = ["Merlin", "Morgana", "Percival", "Assassin", "Generic_Good_1", "Generic_Good_2"]
        self.possible_agents = self.agents[:]  # PettingZoo requirement: list of ALL possible agents
        
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
            "who_choose_prior_missions": MultiDiscrete([6]*25),                 # 6 people for 5 missions x 5 attempts
            "number_of_fails_per_prior_missions": MultiDiscrete([3]*5),        # 0-2 fails per mission
            "whos_winning": MultiDiscrete([3]*5),                              # 0=evil won, 1=not played, 2=good won
            "current_proposed_team": MultiBinary(6),                           # [1,0,1,0,0,0] = players 0&2 on team
            "who_choose_current_mission": Discrete(6),                         # Index of leader who proposed current team
            "did_i_choose_current_team": Discrete(2),                          # 1=yes, 0=no
            "mission_number": Discrete(6),                                     # Current mission (1-5)
            "reject_count": Discrete(5),                                       # How many consecutive rejections (0-4)
            "evil_merlin_guesses": Box(0.0, 1.0, shape=(6, 6), dtype=np.float32)  # Each evil player's probability guesses [evil_player][target]
        }) for agent in self.agents}

    def reset(self, seed=None, options=None):
        """Start a new game. Assigns roles, resets counters, determines turn order."""
        
        # PettingZoo required initialization
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Avalon game state
        # Each agent always plays their named role (agent name = role)
        # Randomize ORDER of agents to vary game dynamics
        random.shuffle(self.agents)
        self.roles = {agent: agent for agent in self.agents}  # Agent name IS the role
        
        self.good_team = ["Merlin", "Percival", "Generic_Good_1", "Generic_Good_2"]
        self.evil_team = ["Morgana", "Assassin"]
        
        self.mission_number = 1
        self.mission_sizes = [2, 3, 4, 3, 4]  # Required team sizes for each mission (6 players)
        self.mission_results = []  # List of True/False for each mission (order matters)
        self.mission_fail_counts = []  # List of integers: how many fails per mission
        self.leader_index = 0  # Will be set after shuffle
        self.current_mission_leader = 0  # Index of leader who proposed current team
        self.proposed_team = []
        self.current_mission_team = []
        self.reject_count = 0
        self.evil_guesses_buffer = {}  # Evil team's probability guesses for Merlin identity
        self.evil_merlin_votes = np.zeros((6, 6), dtype=np.float32)  # Each evil player's probabilities [evil_idx][target_idx]
        self.last_mission_outcome = None  # Track last mission result for logging
        
        # Vote tracking
        self.all_votes_history = []  # Stores all votes across all missions
        self.who_chose_history = []  # Track who was leader for completed missions
        self.vote_buffer = {}
        self.quest_buffer = {}
        
        # Phase management
        self.phase = "TEAM_PROPOSAL"
        
        # Turn order - agents list determines order, leader_index points into possible_agents
        # Set first agent in shuffled order as the leader
        self.leader_index = self.possible_agents.index(self.agents[0])
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent):
        """Return what this agent can see. Different agents see different info (partial observability)."""
        
        # Create agent-relative ordering: current agent is always at index 0
        # This prevents the Assassin from learning "Merlin = index 0"
        agent_idx = self.agents.index(agent)
        relative_order = self.agents[agent_idx:] + self.agents[:agent_idx]
        
        # Known spies (only if I'm Merlin or Evil)
        known_spies = np.zeros(6, dtype=np.int8)
        agent_role = self.roles.get(agent, "")
        if agent in self.evil_team or agent_role == "Merlin":
            for i, other_agent in enumerate(relative_order):
                if other_agent in self.evil_team:
                    known_spies[i] = 1
        
        # Percival sees Merlin + Morgana (can't distinguish)
        percival_view = np.zeros(6, dtype=np.int8)
        if agent_role == "Percival":
            for i, other_agent in enumerate(relative_order):
                other_role = self.roles.get(other_agent, "")
                if other_role in ["Merlin", "Morgana"]:
                    percival_view[i] = 1
        
        # All prior votes (flatten vote history)
        all_prior_votes = np.zeros(150, dtype=np.int8)
        for i, vote in enumerate(self.all_votes_history[:150]):
            all_prior_votes[i] = vote
        
        # Who chose prior missions (convert to relative positions)
        who_choose_prior = np.zeros(25, dtype=np.int8)
        for i, leader_idx in enumerate(self.who_chose_history[:25]):
            # Convert absolute leader_index to agent, then to relative position
            past_leader = self.agents[leader_idx % len(self.agents)]
            relative_past_leader_idx = relative_order.index(past_leader)
            who_choose_prior[i] = relative_past_leader_idx
        
        # Number of fails per mission (0-2 fails possible)
        fails_per_mission = np.zeros(5, dtype=np.int8)
        for i, fail_count in enumerate(self.mission_fail_counts):
            fails_per_mission[i] = fail_count
        
        # Who's winning per mission (0=evil won, 1=not played, 2=good won)
        whos_winning = np.ones(5, dtype=np.int8)  # Default to 1 (not played)
        for i, result in enumerate(self.mission_results):
            whos_winning[i] = 2 if result else 0  # True=good won (2), False=evil won (0)
        
        # Current proposed team (use relative ordering)
        current_team = np.zeros(6, dtype=np.int8)
        for i, team_agent in enumerate(relative_order):
            if team_agent in self.proposed_team:
                current_team[i] = 1
        
        # Evil team's probability guesses for Merlin (only visible to evil team)
        # Remap both dimensions to relative ordering
        evil_guesses = np.zeros((6, 6), dtype=np.float32)
        if agent in self.evil_team:
            for i, evil_agent in enumerate(relative_order):
                for j, target_agent in enumerate(relative_order):
                    # Get absolute indices
                    abs_evil_idx = self.agents.index(evil_agent)
                    abs_target_idx = self.agents.index(target_agent)
                    # Copy probability from absolute to relative position
                    evil_guesses[i][j] = self.evil_merlin_votes[abs_evil_idx][abs_target_idx]
        
        # Convert leader index to relative position
        leader_agent = self.agents[self.leader_index % len(self.agents)]
        relative_leader_idx = relative_order.index(leader_agent)
        
        return {
            "known_spies": known_spies,
            "percival_view": percival_view,
            "all_prior_votes": all_prior_votes,
            "who_choose_prior_mission": who_choose_prior,
            "number_of_fails_per_prior_missions": fails_per_mission,
            "whos_winning": whos_winning,
            "current_proposed_team": current_team,
            "who_choose_current_mission": relative_leader_idx,
            "did_i_choose_current_team": 1 if leader_agent == agent else 0,
            "mission_number": self.mission_number,
            "reject_count": self.reject_count,
            "evil_merlin_guesses": evil_guesses
        }
    
    def step(self, action):
        """Process agent action. Core Avalon game logic."""
        
        agent = self.agent_selection
        
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        
        # Unpack action tuple
        action_type, team_selection, vote_or_quest, target_player = action
        
        # Track if we should skip _accumulate_rewards (only for early returns)
        should_accumulate = True
        
        if self.phase == "TEAM_PROPOSAL":
            # Only the current leader can propose a team
            if agent != self.possible_agents[self.leader_index]:
                # Not leader's turn, skip to next agent
                self.agent_selection = self._agent_selector.next()
                return
            
            # Leader proposes team (action_type should be 0)
            if action_type == 0:
                # Extract selected players from binary array
                selected_indices = [i for i, selected in enumerate(team_selection) if selected == 1]
                
                # Validate team size matches mission requirements
                required_size = self.mission_sizes[self.mission_number - 1]
                if len(selected_indices) != required_size:
                    # Invalid team size, skip this action
                    self.agent_selection = self._agent_selector.next()
                    return
                
                self.proposed_team = [self.possible_agents[i] for i in selected_indices]
                
                # Record who chose this team (current, not historical yet)
                self.current_mission_leader = self.leader_index
                
                # Transition to voting phase
                self.phase = "TEAM_VOTE"
                self.vote_buffer = {}
            
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        if self.phase == "TEAM_VOTE":
            # All agents vote on the proposed team (action_type should be 1)
            if action_type == 1:
                # Record this agent's vote (0=reject, 1=approve)
                self.vote_buffer[agent] = vote_or_quest
            
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
            
            # Check if all agents have voted
            if len(self.vote_buffer) == len(self.agents):
                # Record votes to history (flatten vote_buffer to list in agent order)
                for agent_name in self.agents:
                    self.all_votes_history.append(self.vote_buffer[agent_name])
                
                # Count approvals
                approvals = sum(self.vote_buffer.values())
                majority = len(self.agents) // 2 + 1
                
                if approvals >= majority:
                    # Team approved: move to quest phase
                    self.current_mission_team = self.proposed_team[:]
                    self.reject_count = 0  # Reset reject counter
                    self.phase = "QUEST"
                    self.quest_buffer = {}
                else:
                    # Team rejected
                    self.reject_count += 1
                    
                    if self.reject_count == 5:
                        # Evil wins if 5 consecutive rejections
                        for agent_name in self.agents:
                            self.terminations[agent_name] = True
                            if agent_name in self.good_team:
                                self.rewards[agent_name] = -1
                            else:
                                self.rewards[agent_name] = 1
                    else:
                        # Rotate leader and return to proposal phase
                        self.leader_index = (self.leader_index + 1) % len(self.agents)
                        self.proposed_team = []
                        self.phase = "TEAM_PROPOSAL"
            
            self._accumulate_rewards()
            return
        
        if self.phase == "QUEST":
            # Only mission team members participate in quest
            if agent not in self.current_mission_team:
                # Not on mission team, skip to next agent
                self.agent_selection = self._agent_selector.next()
                return
            
            # Mission team member plays quest card (action_type should be 2)
            if action_type == 2:
                # Good team members MUST play success (1), evil can choose
                if agent in self.good_team:
                    quest_action = 1
                else:
                    quest_action = vote_or_quest
                
                self.quest_buffer[agent] = quest_action
            
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
            
            # Check if all mission team members have played
            if len(self.quest_buffer) == len(self.current_mission_team):
                # Count fails (anonymous)
                fail_count = sum(1 for vote in self.quest_buffer.values() if vote == 0)
                
                # Determine mission success based on Avalon rules
                # Mission 4 with 7+ players requires 2 fails to fail the mission
                # All other missions require only 1 fail to fail the mission
                if len(self.possible_agents) >= 7 and self.mission_number == 4:
                    mission_success = fail_count < 2
                else:
                    mission_success = fail_count == 0
                
                # Record mission results
                self.mission_results.append(mission_success)
                self.mission_fail_counts.append(fail_count)
                self.last_mission_outcome = (self.mission_number, mission_success, fail_count)  # For logging
                
                # Add leader to history now that mission is complete
                self.who_chose_history.append(self.current_mission_leader)
                
                # Check win conditions (first to 3 wins)
                good_wins = sum(1 for result in self.mission_results if result)
                evil_wins = sum(1 for result in self.mission_results if not result)
                
                if good_wins == 3:
                    # Good team wins 3 missions: evil team discusses who Merlin is
                    self.phase = "EVIL_DISCUSSION"
                    self.evil_guesses_buffer = {}
                elif evil_wins == 3:
                    # Evil team wins 3 missions: game over, evil wins
                    for agent_name in self.agents:
                        self.terminations[agent_name] = True
                        if agent_name in self.good_team:
                            self.rewards[agent_name] = -1
                        else:
                            self.rewards[agent_name] = 1
                else:
                    # Continue to next mission
                    self.mission_number += 1
                    self.leader_index = (self.leader_index + 1) % len(self.agents)
                    self.proposed_team = []
                    self.current_mission_team = []
                    self.reject_count = 0
                    self.phase = "TEAM_PROPOSAL"
            
            self._accumulate_rewards()
            return
        
        if self.phase == "EVIL_DISCUSSION":
            # Evil team members submit probability distributions for who they think Merlin is
            if agent not in self.evil_team:
                # Not evil team, skip to next agent
                self.agent_selection = self._agent_selector.next()
                return
            
            # Evil team member submits their probabilities (action_type should be 3)
            # Using team_selection (MultiBinary) to represent probabilities [0.0-1.0] for each player
            if action_type == 3:
                # Convert MultiBinary to float probabilities (values 0 or 1 become 0.0 or 1.0)
                # Note: Neural networks will output these as continuous values between 0 and 1
                probabilities = np.array(team_selection, dtype=np.float32)
                self.evil_guesses_buffer[agent] = probabilities
            
            # Move to next agent
            self.agent_selection = self._agent_selector.next()
            
            # Check if all evil team members have voted
            if len(self.evil_guesses_buffer) == len(self.evil_team):
                # Store each evil player's probabilities in their corresponding row
                for agent_name, probabilities in self.evil_guesses_buffer.items():
                    agent_idx = self.possible_agents.index(agent_name)
                    self.evil_merlin_votes[agent_idx] = probabilities
                
                # Transition to assassination phase
                self.phase = "ASSASSINATION"
            
            self._accumulate_rewards()
            return
        
        if self.phase == "ASSASSINATION":
            # Only the assassin can act in this phase
            agent_role = self.roles.get(agent, "")
            if agent_role != "Assassin":
                # Not assassin's turn, skip to next agent
                self.agent_selection = self._agent_selector.next()
                return
            
            # Assassin attempts to kill Merlin (action_type should be 3)
            if action_type == 3:
                # Check if target is Merlin
                target_agent = self.possible_agents[target_player]
                target_role = self.roles.get(target_agent, "")
                
                if target_role == "Merlin":
                    # Evil wins: assassin successfully killed Merlin
                    for agent_name in self.agents:
                        self.terminations[agent_name] = True
                        if agent_name in self.good_team:
                            self.rewards[agent_name] = -1
                        else:
                            self.rewards[agent_name] = 1
                else:
                    # Good wins: assassin failed to kill Merlin
                    for agent_name in self.agents:
                        self.terminations[agent_name] = True
                        if agent_name in self.good_team:
                            self.rewards[agent_name] = 1
                        else:
                            self.rewards[agent_name] = -1
            
            # Move to next agent (game will end after this)
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()
            return
        
        self._accumulate_rewards()