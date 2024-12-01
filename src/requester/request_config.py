class RequestConfig:
    def __init__(self, coords, start_date, end_date, download_checked, selected_file_type, download_directory, import_checked, plot_checked, additional_options):
        self._coords = coords
        self._start_date = start_date
        self._end_date = end_date
        self._download_checked = download_checked
        self._selected_file_type = selected_file_type
        self._download_directory = download_directory
        self._import_checked = import_checked
        self._plot_checked = plot_checked
        self._additional_options = additional_options

    @property
    def coords(self):
        return self._coords

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def download_checked(self):
        return self._download_checked

    @property
    def selected_file_type(self):
        return self._selected_file_type

    @property
    def download_directory(self):
        return self._download_directory

    @property
    def import_checked(self):
        return self._import_checked

    @property
    def plot_checked(self):
        return self._plot_checked

    @property
    def additional_options(self):
        return self._additional_options