import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import os

@dataclass
class MissionRecord:
    """Records details of a single mission."""
    mission_number: int
    leader: str
    proposed_team: List[str]
    votes: Dict[str, int]  # agent -> 1 (approve) or 0 (reject)
    approved: bool
    quest_results: Optional[Dict[str, int]] = None  # agent -> 1 (success) or 0 (fail), only if approved
    fail_count: Optional[int] = None
    mission_success: Optional[bool] = None
    reject_count: int = 0

@dataclass
class PlayerStats:
    """Statistics for a specific player/role."""
    role: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    times_on_missions: int = 0
    missions_proposed: int = 0
    proposals_approved: int = 0
    votes_cast: int = 0
    approve_votes: int = 0
    reject_votes: int = 0
    times_assassinated: int = 0
    successful_assassinations: int = 0  # Only for Assassin role
    
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played * 100
    
    def approval_rate(self) -> float:
        if self.missions_proposed == 0:
            return 0.0
        return self.proposals_approved / self.missions_proposed * 100
    
    def vote_approve_rate(self) -> float:
        if self.votes_cast == 0:
            return 0.0
        return self.approve_votes / self.votes_cast * 100

@dataclass
class GameResult:
    """Complete record of a single game."""
    game_id: int
    timestamp: str
    agent_order: List[str]
    roles: Dict[str, str]
    good_team: List[str]
    evil_team: List[str]
    missions: List[MissionRecord]
    final_mission_scores: Dict[str, int]  # mission_num -> 2 (good won) or 0 (evil won)
    assassination_target: Optional[str]
    winner: str  # "good" or "evil"
    winning_condition: str  # "3_missions", "5_rejections", "assassinated_merlin", "failed_assassination"
    final_rewards: Dict[str, float]
    turn_count: int
    
    def __str__(self) -> str:
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"GAME #{self.game_id} - {self.timestamp}")
        lines.append(f"{'='*80}")
        lines.append(f"Winner: {self.winner.upper()} ({self.winning_condition})")
        lines.append(f"Turn count: {self.turn_count}")
        lines.append(f"\nAgent Order: {', '.join(self.agent_order)}")
        lines.append(f"Good Team: {', '.join(self.good_team)}")
        lines.append(f"Evil Team: {', '.join(self.evil_team)}")
        
        lines.append(f"\n{'='*80}")
        lines.append("MISSION HISTORY")
        lines.append(f"{'='*80}")
        
        for mission in self.missions:
            if mission.approved:
                outcome = "✓ SUCCESS" if mission.mission_success else "✗ FAILED"
                lines.append(f"\nMission {mission.mission_number}: {outcome} ({mission.fail_count} fails)")
                lines.append(f"  Leader: {mission.leader}")
                lines.append(f"  Team: {', '.join(mission.proposed_team)}")
                lines.append(f"  Votes: {sum(mission.votes.values())}/{len(mission.votes)} approved")
            else:
                lines.append(f"\nMission {mission.mission_number} Proposal: REJECTED")
                lines.append(f"  Leader: {mission.leader}")
                lines.append(f"  Team: {', '.join(mission.proposed_team)}")
                lines.append(f"  Votes: {sum(mission.votes.values())}/{len(mission.votes)} approved (needed majority)")
        
        if self.assassination_target:
            lines.append(f"\n{'='*80}")
            lines.append(f"ASSASSINATION: Target = {self.assassination_target} (Role: {self.roles[self.assassination_target]})")
            if self.winning_condition == "assassinated_merlin":
                lines.append("  Result: Evil wins! Merlin was killed.")
            else:
                lines.append("  Result: Good wins! Assassin missed Merlin.")
        
        lines.append(f"\n{'='*80}")
        lines.append("FINAL REWARDS")
        lines.append(f"{'='*80}")
        for agent, reward in self.final_rewards.items():
            lines.append(f"  {agent} ({self.roles[agent]}): {reward:+.1f}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        return data

class GameAnalytics:
    """Aggregates statistics across multiple games."""
    
    def __init__(self):
        self.games: List[GameResult] = []
        self.player_stats: Dict[str, PlayerStats] = {}
        self.good_wins = 0
        self.evil_wins = 0
        self.total_games = 0
        
        # Initialize stats for each role
        roles = ["Merlin", "Morgana", "Percival", "Assassin", "Generic_Good_1", "Generic_Good_2"]
        for role in roles:
            self.player_stats[role] = PlayerStats(role=role)
    
    def add_game(self, game: GameResult):
        """Add a game result and update statistics."""
        self.games.append(game)
        self.total_games += 1
        
        if game.winner == "good":
            self.good_wins += 1
        else:
            self.evil_wins += 1
        
        # Update player statistics
        for agent, role in game.roles.items():
            stats = self.player_stats[role]
            stats.games_played += 1
            
            if game.final_rewards[agent] > 0:
                stats.wins += 1
            else:
                stats.losses += 1
            
            # Count times on missions
            for mission in game.missions:
                if agent in mission.proposed_team:
                    stats.times_on_missions += 1
                
                if mission.leader == agent:
                    stats.missions_proposed += 1
                    if mission.approved:
                        stats.proposals_approved += 1
                
                if agent in mission.votes:
                    stats.votes_cast += 1
                    if mission.votes[agent] == 1:
                        stats.approve_votes += 1
                    else:
                        stats.reject_votes += 1
            
            # Assassination statistics
            if game.assassination_target == agent:
                stats.times_assassinated += 1
            
            if role == "Assassin" and game.assassination_target:
                if game.winning_condition == "assassinated_merlin":
                    stats.successful_assassinations += 1
    
    def get_summary(self) -> str:
        """Generate a comprehensive summary of all games."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"ANALYTICS SUMMARY - {self.total_games} Games")
        lines.append(f"{'='*80}")
        
        lines.append(f"\nOVERALL STATISTICS")
        lines.append(f"  Good Wins: {self.good_wins} ({self.good_wins/self.total_games*100:.1f}%)")
        lines.append(f"  Evil Wins: {self.evil_wins} ({self.evil_wins/self.total_games*100:.1f}%)")
        
        lines.append(f"\n{'='*80}")
        lines.append("PLAYER/ROLE STATISTICS")
        lines.append(f"{'='*80}")
        
        # Group by team
        good_roles = ["Merlin", "Percival", "Generic_Good_1", "Generic_Good_2"]
        evil_roles = ["Morgana", "Assassin"]
        
        lines.append("\nGOOD TEAM:")
        for role in good_roles:
            stats = self.player_stats[role]
            lines.append(f"\n  {role}:")
            lines.append(f"    Games: {stats.games_played} | Win Rate: {stats.win_rate():.1f}%")
            lines.append(f"    Missions: {stats.times_on_missions} times on team")
            lines.append(f"    Leadership: {stats.missions_proposed} proposals, {stats.approval_rate():.1f}% approved")
            lines.append(f"    Voting: {stats.votes_cast} votes, {stats.vote_approve_rate():.1f}% approvals")
            if role == "Merlin":
                lines.append(f"    Assassinated: {stats.times_assassinated} times")
        
        lines.append("\nEVIL TEAM:")
        for role in evil_roles:
            stats = self.player_stats[role]
            lines.append(f"\n  {role}:")
            lines.append(f"    Games: {stats.games_played} | Win Rate: {stats.win_rate():.1f}%")
            lines.append(f"    Missions: {stats.times_on_missions} times on team")
            lines.append(f"    Leadership: {stats.missions_proposed} proposals, {stats.approval_rate():.1f}% approved")
            lines.append(f"    Voting: {stats.votes_cast} votes, {stats.vote_approve_rate():.1f}% approvals")
            if role == "Assassin":
                lines.append(f"    Assassinations: {stats.successful_assassinations}/{stats.games_played} successful")
        
        return '\n'.join(lines)
    
    def save_batch(self, batch_num: int, output_dir: str = "game_logs"):
        """Save a batch of games to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f"batch_{batch_num:04d}.json")
        
        data = {
            "batch_number": batch_num,
            "total_games": len(self.games),
            "good_wins": self.good_wins,
            "evil_wins": self.evil_wins,
            "games": [game.to_dict() for game in self.games],
            "player_stats": {role: asdict(stats) for role, stats in self.player_stats.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved batch {batch_num} to {filename}")
        
        # Reset for next batch
        self.games.clear()
