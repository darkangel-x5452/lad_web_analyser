import os

from jinja2 import Template
import yaml

from utils.application_init.config_maps.general import MatchupInfo, QueryMetadata
from utils.application_init.config_variables.base_configs import BaseConfigsCombined


class BaseWebLinkFinder(BaseConfigsCombined):
    def __init__(self, provider: str | None = None):
        super().__init__()
        self.provider = provider
        self.contenstant_main = None
        self.contenstant_home = None
        self.contenstant_away = None
        self.qm = None
        self.matchup_info: MatchupInfo

    def _render_query_yml(self,
                          contestant_main: str | None = None,
                          ):
        print(f"setup called for class {self.__class__.__name__}")
        with open(self.fns.query_metadata, "r") as f:
            self.category_prompts = Template(f.read())
        rendered = self.category_prompts.render(
            CONTESTANT_MAIN=contestant_main,
            CONTESTANT_HOME=self.matchup_info.contestant_home,
            CONTESTANT_AWAY=self.matchup_info.contestant_away,
            DATE=self.matchup_info.start_time_aest,
            COMPETITION_NAME=self.matchup_info.competition_name,
            SPORT_NAME=self.matchup_info.sport_name,
        )

        config = yaml.safe_load(rendered)

        self.qm = QueryMetadata(**config)

    def find_links(self, query: str) -> list[str]:
        """Find relevant links for the given query."""
        raise NotImplementedError("Must be implemented by subclass.")
