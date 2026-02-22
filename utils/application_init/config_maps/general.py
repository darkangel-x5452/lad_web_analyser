from dataclasses import dataclass

@dataclass
class QueryMetadata:
    team_statistics: str | None
    player_statistics: str | None
    match_statistics: str | None
    match_commentary: str | None

@dataclass
class MatchupInfo:
    contestant_home: str
    contestant_away: str
    start_time_aest: str
    competition_name: str
    tournament_name: str | None
    sport_name: str
    
    
    