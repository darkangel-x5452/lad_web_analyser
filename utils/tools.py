import re
from datetime import datetime


def clean_directory_name(name: str) -> str:
    # Remove everything except letters and numbers
    clean = re.sub(r'[^a-zA-Z0-9]', '', name)
    return clean

def iso_to_date_string(date_str: str):
    try:
        dt = datetime.fromisoformat(date_str)  # validates ISO date
        return dt.strftime("%Y-%m-%d")         # convert to date string
    except ValueError:
        return None  # not a valid date