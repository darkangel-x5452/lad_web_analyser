import json
import os
import time

from utils.tools import iso_to_date_string
from weblink_finder.ddgs.v1.simple_api import ddgs_main
from utils.logger import logger
from webpage_analyser.v6.webpage_query_extractor_free_image import link_analyser_app


_logger = logger(__name__)


class MainAppRunner:
    def __init__(self):
        self.env = "prod"
        self.match_dirs_fp = "configs/match_dirs_ddgs.json"
        self.extract_info_prompt = os.environ["DEMO_URL_QUERY"]

    def get_links(self) -> list[str]:
        matches_fp = os.environ["GAME_MATCHES_FP"]
        with open(matches_fp, "r", encoding="utf-8") as f:
            match_ls = json.load(f)

        matches_metadata = []
        match_ls_len = len(match_ls)
        for _idx, _match in enumerate(match_ls):
            tournament_name = _match.get("tournamentName", None)
            sport_name: str = _match["sport_name"]
            if sport_name.lower() != "basketball":
                print(
                    f"Skipping match '{_match['match_name']}' due to unsupported sport '{sport_name}'."
                )
                continue
            print(f"{_idx}:{match_ls_len}, Processing match: {_match['match_name']}...")
            contestant_home = _match["contestants"][0]["full_name"]
            contestant_away = _match["contestants"][1]["full_name"]
            competition_name = _match["competition_name"]
            matchup_info = {
                "contestant_names": _match["contestant_names"],
                "contestant_home": contestant_home,
                "contestant_away": contestant_away,
                "start_time_aest": _match["start_time_aest"],
                "competition_name": competition_name,
                "tournament_name": tournament_name,
                "sport_name": sport_name,
                "match_name": _match["match_name"],
            }
            if competition_name == "NCAA Basketball":
                comp_url = "mens-college-basketball"
            elif competition_name == "NBA":
                comp_url = "nba"
            elif competition_name == "NCAA Basketball Women":
                comp_url = "womens-college-basketball"
            else:
                _logger.warning(f"Unknown {sport_name} competition '{competition_name}' for match '{_match['match_name']}'. Skipping link search.")
                continue
            query = f"{os.environ['LINK_QUERY_PREFIX']}/{comp_url}/game/_/gameId/ live coverage {contestant_home} {contestant_away}"
            counter = 0
            while counter < 3:
                try:
                    links = ddgs_main(query=query)
                    break
                except Exception as e:
                    counter += 1
                    _logger.error(f"Error during DuckDuckGo search for match '{competition_name}', '{_match['match_name']}' (attempt {counter}/3): {e}")
                    time.sleep(5)  # wait before retrying
                    links = []
            matchup_info.update({"links": links})
            matches_metadata.append(matchup_info)
        with open(self.match_dirs_fp, "w", encoding="utf-8") as f:
            json.dump(matches_metadata, f, indent=4)
        print(f"Finished processing matches. Data saved in the following directories:")
        return matches_metadata

    def analyser_app(self, matches_metadata: dict[str, str] | None = None):
        if matches_metadata is None:
            with open(self.match_dirs_fp, "r") as f:
                matches_metadata = json.load(f)
        for _match_metadata in matches_metadata:
            competition_name = _match_metadata["competition_name"]
            match_name = _match_metadata["match_name"]
            start_time_aest = _match_metadata["start_time_aest"]

            if len(_match_metadata["links"]) == 0:
                _logger.error(f"No links found for match '{competition_name}', '{match_name}'. Skipping analysis.")
                continue
            link = _match_metadata["links"][0]["href"]
            start_time_aest_str = iso_to_date_string(start_time_aest)

            save_dir = f"data/site_analyser/site1"
            file_name = f"{start_time_aest_str}.json"
            os.makedirs(save_dir, exist_ok=True)
            save_fp = f"{save_dir}/{file_name}"

            if os.path.exists(save_fp):
                with open(save_fp, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            if any(
                item["competition_name"] == competition_name and
                item["match_name"] == match_name
                for item in existing_data
            ):
                print(f"Data for match '{match_name}' in competition '{competition_name}' already exists. Skipping analysis.")
                continue
            counter = 0
            while counter < 3:
                try:
                    prediction_jn = link_analyser_app(
                        link_input=link,
                        query_input=self.extract_info_prompt,
                    )
                    break
                except Exception as e:
                    _logger.error(f"Error during analysis for match '{competition_name}', '{match_name}': {e}")
                    counter += 1
                    time.sleep(5)  # wait before retrying
                    prediction_jn = {}

            _match_metadata.update({"prediction": prediction_jn})
            existing_data.append(_match_metadata)
            with open(save_fp, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4)

            print("bye")


def run_app():
    mar = MainAppRunner()
    mar.get_links()
    mar.analyser_app()


if __name__ == "__main__":
    _logger.info("Starting main application runner...")
    run_app()
    _logger.info("Finished main application runner...")
