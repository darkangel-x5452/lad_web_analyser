import json
import os
import sys
import requests

from utils.application_init.config_maps.general import MatchupInfo
from weblink_finder.base_link_finder import BaseWebLinkFinder


# ── ANSI colour helpers ───────────────────────────────────────────────────────
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


class SerpApiFinder(BaseWebLinkFinder):
    def __init__(
        self,
        matchup_info: MatchupInfo,
    ):
        super().__init__(
            provider="SerpAPI",
        )
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            print(f"{RED}Error: SERPAPI_API_KEY environment variable not set.{RESET}")
            sys.exit(1)
        self.matchup_info = matchup_info
        self.contenstants_dict = {
            "home": self.matchup_info.contestant_home,
            "away": self.matchup_info.contestant_away,
        }
        self.contenstant_home = self.matchup_info.contestant_home
        self.contenstant_away = self.matchup_info.contestant_away
        self.date = self.matchup_info.start_time_aest
        self.query_suffix = f"For the sport '{self.matchup_info.sport_name}' and competition '{self.matchup_info.competition_name}'."

    def get_links_and_markdown(
        self, query: str, query_type: str, contestant_location: str | None = None
    ):

        print(f"Getting links and markdown for query: {query}")
        if contestant_location is not None:
            query_type_new = f"{query_type}_{contestant_location}"
        else:
            query_type_new = f"{query_type}"
        file_name_links = f"data/serpapi/results/links_{query_type_new}.json"
        file_name_markdown = f"data/serpapi/results/markdown_{query_type_new}.md"

        if contestant_location is not None:
            contestant_name = self.contenstants_dict[contestant_location]
            md_header_1 = f"# {query_type.upper()} for {contestant_name}\n\n"
        else:
            md_header_1 = f"# {query_type.upper()}\n\n"

        params = {
            "engine": "google_ai_mode",
            "q": query,
            "api_key": self.api_key,
        }
        resp = requests.get(f"https://serpapi.com/search.json", params=params)
        resp_jn = resp.json()
        ai_markdown = resp_jn["reconstructed_markdown"]
        ai_markdown = md_header_1 + ai_markdown

        keep_keys = [
            "title",
            "link",
            # "displayed_link",
            # "thumbnail",
            # "favicon",
        ]
        cln_refs = [
            dict((k, v) for k, v in _items.items() if k in keep_keys)
            for _items in resp_jn["references"]
        ]
        with open(file_name_markdown, "w", encoding="utf-8") as f:
            f.write(ai_markdown)
        with open(file_name_links, "w", encoding="utf-8") as f:
            json.dump(cln_refs, f, indent=2, ensure_ascii=False)
        print("bye")
        # from serpapi import GoogleSearch

        # search = GoogleSearch(params)
        # results = search.get_dict()

        # text_blocks = results["text_blocks"]
        # https://serpapi.com/search.json?engine=google_ai_mode&q=Coffee

        # s = serpapi.search(q="Coffee", engine="google", location="Austin, Texas", hl="en", gl="us")
        # resp = requests.get(f"https://serpapi.com/search.json?q=Coffee&api_key={YOUR_API_KEY}")
        # resp = requests.get(f"https://serpapi.com/search.json?engine=google_ai_mode&q=Coffee&api_key={YOUR_API_KEY}")
        # return cln_refs, ai_markdown

    def main(self):
        self.get_contestant_info(self.contenstant_home, "home")
        self.get_contestant_info(self.contenstant_away, "away")
        self.get_match_info()

    def get_contestant_info(self, contestant: str, location: str):

        self._render_query_yml(
            contestant_main=contestant,
        )
        all_statistics = {
            "team_statistics": self.qm.team_statistics,
            "player_statistics": self.qm.player_statistics,
        }
        for _key, _stat in all_statistics.items():
            new_query = self.query_suffix.strip() + _stat
            self.get_links_and_markdown(
                query=new_query, query_type=_key, contestant_location=location
            )

    def get_match_info(self):
        self._render_query_yml()
        all_statistics = {
            "match_statistics": self.qm.match_statistics,
            "match_commentary": self.qm.match_commentary,
        }
        for _key, _stat in all_statistics.items():
            new_query = self.query_suffix.strip() + _stat
            self.get_links_and_markdown(query=new_query, query_type=_key)


if __name__ == "__main__":
    with open("configs/examples/basketball.json", "r", encoding="utf-8") as f:
        example_match_ls = json.load(f)
    for _example in example_match_ls:
        matchup_info = {
            "contestant_home": _example["contestants"][0]["full_name"],
            "contestant_away": _example["contestants"][1]["full_name"],
            "start_time_aest": _example["start_time_aest"],
            "competition_name": _example["competition_name"],
            "tournament_name": None,
            "sport_name": _example["sport_name"],
        }
    
    mi = MatchupInfo(**matchup_info)

    saf = SerpApiFinder(matchup_info=mi)
    saf.main()
