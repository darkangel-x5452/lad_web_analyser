

from utils.application_init.config_variables.file_names import DirectoryNames, FileNames


class BaseConfigsCombined(
):
    def __init__(self):
        self.filns = FileNames()
        self.dirns = DirectoryNames()
