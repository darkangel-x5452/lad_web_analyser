from dataclasses import dataclass


@dataclass
class FileNames():
    query_metadata = "configs/basketball/query_metadata.yml"

@dataclass
class DirectoryNames():
    serpapi_results = "data/serpapi/results"
    reasoning_results = "data/reasoning_engine/results"
