class RequestConfig:
    def __init__(self, coords, start_date, end_date, download_checked, selected_file_type, download_directory, import_checked, ndvi_checked):
        self.coords = coords
        self.start_date = start_date
        self.end_date = end_date
        self.download_checked = download_checked
        self.selected_file_type = selected_file_type
        self.download_directory = download_directory
        self.import_checked = import_checked
        self.ndvi_checked = ndvi_checked

