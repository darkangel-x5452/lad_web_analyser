import json
import os

from reasoning_model.v2.md_reasoning_engine import main
from utils.application_init.config_maps.general import MatchupInfo
from weblink_finder.serpapi.run_app import SerpApiFinder


class MainAppRunner:
    def __init__(self):
        self.env = "prod"
        self.match_dirs_fp = "configs/match_dirs.json"

    def serpapi_app(self) -> list[str]:
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
            matchup_info = {
                "contestant_names": _match["contestant_names"],
                "contestant_home": _match["contestants"][0]["full_name"],
                "contestant_away": _match["contestants"][1]["full_name"],
                "start_time_aest": _match["start_time_aest"],
                "competition_name": _match["competition_name"],
                "tournament_name": tournament_name,
                "sport_name": sport_name,
                "match_name": _match["match_name"],
            }
            mi = MatchupInfo(**matchup_info)
            saf = SerpApiFinder(env=self.env, matchup_info=mi)
            match_dir = saf.get_match_dir()
            saf.main()
            match_metadata = matchup_info.copy()
            match_metadata["match_dir"] = match_dir
            matches_metadata.append(match_metadata)
        with open(self.match_dirs_fp, "w", encoding="utf-8") as f:
            json.dump(matches_metadata, f, indent=4)
        print(f"Finished processing matches. Data saved in the following directories:")
        return matches_metadata

    def reasoning_app(self, matches_metadata: dict[str, str] | None = None):
        if matches_metadata is None:
            with open(self.match_dirs_fp, "r") as f:
                matches_metadata = json.load(f)
        for _match_metadata in matches_metadata:
            dir = _match_metadata["match_dir"]
            print(dir)
            md_dir = dir
            question_prompt = "\nBased on all the statistical data provided, who will win the game between the two contestants? Do not use any bookie odds information or prediction information. Output the result as percentage chance of winning for each team, and a final prediction of the winner."
            md_dir_split = md_dir.split("/")
            date_dir = md_dir_split[-2]
            match_name = md_dir_split[-1]

            save_dir = (
                f"data/reasoning_engine/results/{self.env}/{date_dir}/{match_name}"
            )
            os.makedirs(save_dir, exist_ok=True)
            main(
                md_dir=md_dir,
                question_prompt=question_prompt,
                save_dir=save_dir,
                match_metadata=_match_metadata,
            )


def run_app():
    mar = MainAppRunner()
    mar.serpapi_app()
    mar.reasoning_app()


if __name__ == "__main__":
    run_app()
