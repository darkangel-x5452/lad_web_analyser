import json
import os
import re
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
        env: str,
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
        self.match_name = self.matchup_info.match_name
        self.match_name_shrt = re.sub(r'[^A-Za-z0-9/]', '', self.match_name)
        self.match_name_shrt = self.match_name_shrt.replace("/", "_")
        self.tournament_name = self.matchup_info.tournament_name
        self.date_day = self.date.split("T")[0].replace("-", "")
        self.query_suffix = f"For the sport '{self.matchup_info.sport_name}' and competition '{self.matchup_info.competition_name}'."
        self.env = env
        
        self.date_dir = f"{self.dirns.serpapi_results}/{self.env}/{self.date_day}"
        self.match_dir = f"{self.dirns.serpapi_results}/{self.env}/{self.date_day}/{self.match_name_shrt}"

        

    def get_match_dir(self) -> str:
        return self.match_dir
    
    def get_links_and_markdown(
        self, query: str, query_type: str, contestant_location: str | None = None
    ) -> None:

        print(f"Getting links and markdown for query: {self.match_name}")
        if contestant_location is not None:
            query_type_new = f"{query_type}_{contestant_location}"
        else:
            query_type_new = f"{query_type}"
        file_name_links = f"{self.match_dir}/links_{query_type_new}.json"
        file_name_markdown = f"{self.match_dir}/markdown_{query_type_new}.md"

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
        if "reconstructed_markdown" not in resp_jn:
            print(f"{YELLOW}Warning: 'reconstructed_markdown' not found in SerpAPI response for query: '{self.match_name}', '{query_type_new}'. Skipping markdown generation.{RESET}")
            return None
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
        # from serpapi import GoogleSearch

        # search = GoogleSearch(params)
        # results = search.get_dict()

        # text_blocks = results["text_blocks"]
        # https://serpapi.com/search.json?engine=google_ai_mode&q=Coffee

        # s = serpapi.search(q="Coffee", engine="google", location="Austin, Texas", hl="en", gl="us")
        # resp = requests.get(f"https://serpapi.com/search.json?q=Coffee&api_key={YOUR_API_KEY}")
        # resp = requests.get(f"https://serpapi.com/search.json?engine=google_ai_mode&q=Coffee&api_key={YOUR_API_KEY}")
        # return cln_refs, ai_markdown

    def main(self) -> None:
        if os.path.exists(self.match_dir):
            print(f"{YELLOW}Warning: Match directory '{self.match_dir}' already exists. Skipping link finding for this match to avoid overwriting existing data.{RESET}")
            return None
        os.makedirs(self.match_dir, exist_ok=True)
        self.get_contestant_info(self.contenstant_home, "home")
        self.get_contestant_info(self.contenstant_away, "away")
        self.get_match_info()

    def get_contestant_info(self, contestant: str, location: str) -> None:

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

    def get_match_info(self) -> None:
        self._render_query_yml()
        all_statistics = {
            "match_statistics_and_commentary": self.qm.match_statistics_and_commentary,
            # "match_commentary": self.qm.match_commentary,
        }
        for _key, _stat in all_statistics.items():
            new_query = self.query_suffix.strip() + _stat
            self.get_links_and_markdown(query=new_query, query_type=_key)


if __name__ == "__main__":
    print("hi analyser")
    # with open("configs/examples/basketball.json", "r", encoding="utf-8") as f:
    #     example_match_ls = json.load(f)
    matches_fp = os.environ("GAME_MATCHES_FP")
    with open(matches_fp, "r", encoding="utf-8") as f:
        example_match_ls = json.load(f)

    match_dirs = []
    for _match in example_match_ls:
        tournament_name = _match.get("tournamentName", None)
        sport_name = _match["sport_name"]
        if sport_name.lower() != "basketball":
            print(f"{YELLOW}Skipping match '{_match['match_name']}' due to unsupported sport '{sport_name}'.{RESET}")
            continue
        matchup_info = {
            "contestant_home": _match["contestants"][0]["full_name"],
            "contestant_away": _match["contestants"][1]["full_name"],
            "start_time_aest": _match["start_time_aest"],
            "competition_name": _match["competition_name"],
            "tournament_name": tournament_name,
            "sport_name": _match["sport_name"],
            "match_name": _match["match_name"],
        }
        mi = MatchupInfo(**matchup_info)
        saf = SerpApiFinder(matchup_info=mi)
        saf.main()
        match_dir = saf.get_match_dir
        match_dirs.append(match_dir)
    print(f"{GREEN}Finished processing matches. Data saved in the following directories:{RESET}")
    print("bye analyser")