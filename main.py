from AECSetup import CustomGameEnv
from game_analytics import GameResult, GameAnalytics, MissionRecord
import numpy as np
from datetime import datetime

def play_game(game_id=1, verbose=False, extra_verbose=False):
    """Play a single game and return a GameResult object with complete analytics."""
    env = CustomGameEnv()
    env.reset()
    
    # Save game setup (env.agents gets cleared after game ends)
    agent_order = env.agents[:]
    roles = env.roles.copy()
    good_team = env.good_team[:]
    evil_team = env.evil_team[:]
    
    # Track missions as they happen
    missions_log = []
    current_proposal = None
    vote_buffer = {}
    
    # Clear mission logging tracker
    if hasattr(play_game, '_logged_missions'):
        play_game._logged_missions.clear()
    
    if verbose:
        print(f"\n=== New Game ===")
        print(f"Agent order: {agent_order}")
        print(f"Roles: {roles}")
        print(f"Good team: {good_team}")
        print(f"Evil team: {evil_team}")
    
    turn_count = 0
    final_rewards = {} 
    assassination_target = None
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        # Capture rewards as they come in
        if reward != 0:
            final_rewards[agent] = reward
        
        if termination or truncation:
            action = None
            if extra_verbose:
                print(f"  {agent} terminated, reward: {reward}")
        else:
            phase = env.phase
            is_leader = agent == env.possible_agents[env.leader_index]
            
            if extra_verbose:
                print(f"\n[Turn {turn_count}] Agent: {agent}, Phase: {phase}, Leader: {is_leader}")
            elif verbose and turn_count % 6 == 0:
                print(f"\nPhase: {phase}, Mission: {env.mission_number}, Leader: {env.possible_agents[env.leader_index]}")
            
            if phase == "TEAM_PROPOSAL":
                action_type = 0
                required_size = env.mission_sizes[env.mission_number - 1]
                team_selection = np.zeros(6, dtype=np.int8)
                team_selection[:required_size] = 1
                vote_or_quest = 0
                target_player = 0
                
                if is_leader:
                    proposed_team_list = [env.possible_agents[i] for i in range(required_size)]
                    current_proposal = MissionRecord(
                        mission_number=env.mission_number,
                        leader=agent,
                        proposed_team=proposed_team_list,
                        votes={},
                        approved=False,
                        reject_count=env.reject_count
                    )
                    vote_buffer = {}
                    
                    if verbose or extra_verbose:
                        print(f"  {agent} (leader) proposes team: {proposed_team_list}")
                elif extra_verbose:
                    print(f"  {agent} (not leader) skips")
                
            elif phase == "TEAM_VOTE":
                action_type = 1
                team_selection = np.zeros(6, dtype=np.int8)
                vote_or_quest = 1
                target_player = 0
                
                # Track vote
                vote_buffer[agent] = vote_or_quest
                
                # Check if voting is complete and log mission
                if len(vote_buffer) == len(env.agents) and current_proposal:
                    current_proposal.votes = vote_buffer.copy()
                    approvals = sum(vote_buffer.values())
                    majority = len(env.agents) // 2 + 1
                    current_proposal.approved = approvals >= majority
                    
                    if not current_proposal.approved:
                        missions_log.append(current_proposal)
                        current_proposal = None
                
                # Log votes aggregated at end of phase
                if extra_verbose:
                    print(f"  {agent} votes: {'approve' if vote_or_quest else 'reject'}")
                
            elif phase == "QUEST":
                action_type = 2
                team_selection = np.zeros(6, dtype=np.int8)
                vote_or_quest = 1 if agent in env.good_team else 0
                target_player = 0
                
                on_mission = agent in env.current_mission_team
                if extra_verbose:
                    if on_mission:
                        print(f"  {agent} plays: {'success' if vote_or_quest else 'fail'}")
                    else:
                        print(f"  {agent} not on mission, skips")
                
                # After all quest actions, log mission result
                if verbose and hasattr(env, 'last_mission_outcome') and env.last_mission_outcome:
                    mission_num, success, fails = env.last_mission_outcome
                    if mission_num not in getattr(play_game, '_logged_missions', set()):
                        if not hasattr(play_game, '_logged_missions'):
                            play_game._logged_missions = set()
                        play_game._logged_missions.add(mission_num)
                        
                        # Complete the mission record
                        if current_proposal:
                            current_proposal.quest_results = env.quest_buffer.copy()
                            current_proposal.fail_count = fails
                            current_proposal.mission_success = success
                            missions_log.append(current_proposal)
                            current_proposal = None
                        
                        outcome = "SUCCESS" if success else "FAILED"
                        good_wins = sum(1 for r in env.mission_results if r)
                        evil_wins = sum(1 for r in env.mission_results if not r)
                        print(f"\n  >>> MISSION {mission_num} {outcome} ({fails} fails) | Score: Good {good_wins} - Evil {evil_wins} <<<")
                
            elif phase == "EVIL_DISCUSSION":
                action_type = 3
                team_selection = np.random.rand(6)
                vote_or_quest = 0
                target_player = 0
                
                if agent in env.evil_team:
                    if extra_verbose:
                        print(f"  {agent} (evil) submits Merlin guesses: {team_selection}")
                elif extra_verbose:
                    print(f"  {agent} (good) skips evil discussion")
                
            elif phase == "ASSASSINATION":
                action_type = 3
                team_selection = np.zeros(6, dtype=np.int8)
                vote_or_quest = 0
                target_player = np.random.randint(0, 6)
                
                if env.roles.get(agent) == "Assassin":
                    assassination_target = env.possible_agents[target_player]
                    if verbose or extra_verbose:
                        print(f"  {agent} (Assassin) targets: {assassination_target} (role: {env.roles[assassination_target]})")
                elif extra_verbose:
                    print(f"  {agent} not assassin, skips")
                
            else:
                action_type = 0
                team_selection = np.zeros(6, dtype=np.int8)
                vote_or_quest = 0
                target_player = 0
            
            action = (action_type, team_selection, vote_or_quest, target_player)
            turn_count += 1
        
        env.step(action)
    
    # Determine winner and condition
    if not final_rewards:
        final_rewards = env.rewards
    
    # Check a good team member's reward to determine winner (positive = good wins, negative = evil wins)
    good_agent = good_team[0]
    winner = "good" if final_rewards[good_agent] > 0 else "evil"
    
    # Determine winning condition
    if env.reject_count == 5:
        winning_condition = "5_rejections"
    elif assassination_target:
        if winner == "evil":
            winning_condition = "assassinated_merlin"
        else:
            winning_condition = "failed_assassination"
    else:
        winning_condition = "3_missions"
    
    # Create mission scores
    final_mission_scores = {}
    for i, result in enumerate(env.mission_results):
        final_mission_scores[i + 1] = 2 if result else 0
    
    # Create GameResult object
    game_result = GameResult(
        game_id=game_id,
        timestamp=datetime.now().isoformat(),
        agent_order=agent_order,
        roles=roles,
        good_team=good_team,
        evil_team=evil_team,
        missions=missions_log,
        final_mission_scores=final_mission_scores,
        assassination_target=assassination_target,
        winner=winner,
        winning_condition=winning_condition,
        final_rewards=final_rewards,
        turn_count=turn_count
    )
    
    if verbose or extra_verbose:
        print(game_result)
    
    return game_result

# Play one game with extra verbose output for debugging
print("Playing 1 game with extra details:")
play_game(game_id=1, verbose=True, extra_verbose=True)

# Play games continuously until keyboard interrupt
analytics = GameAnalytics()
errors = 0
game_count = 0
batch_num = 1

print(f"\n\nPlaying games continuously... (Press Ctrl+C to stop)")
print(f"Saving batches every 1000 games to 'game_logs/' folder\n")

try:
    while True:
        try:
            game_count += 1
            game_result = play_game(game_id=game_count, verbose=False)
            analytics.add_game(game_result)
            
        except Exception as e:
            errors += 1
            if errors <= 10:  # Only print first 10 errors
                print(f"Game {game_count} error: {e}")
        
        # Progress update every 100 games
        if game_count % 100 == 0:
            print(f"Completed {game_count} games... (Good: {analytics.good_wins}, Evil: {analytics.evil_wins}, Errors: {errors})")
        
        # Save batch every 1000 games
        if game_count % 1000 == 0:
            analytics.save_batch(batch_num)
            print(analytics.get_summary())
            batch_num += 1

except KeyboardInterrupt:
    print(f"\n\n=== Stopping after {game_count} games ===")
    
    # Save final partial batch if any games recorded
    if len(analytics.games) > 0:
        analytics.save_batch(batch_num)

# Final summary
print(analytics.get_summary())

if errors > 0:
    print(f"\nErrors encountered: {errors}")