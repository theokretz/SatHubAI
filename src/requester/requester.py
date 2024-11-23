# requester.py
import traceback
from abc import ABC, abstractmethod

from ..exceptions.missing_credentials_exception import MissingCredentialsException
from ..exceptions.ndvi_calculation_exception import NDVICalculationError
from ..utils import display_error_message


class Requester:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def request_data(self):
        """abstract method to request data from provider"""
        pass

    def execute_request(self):
        """executes the request and handles errors"""
        try:
            self.request_data()
        except NDVICalculationError as e:
            display_error_message(str(e), "NDVI Calculation Error!")
        except MissingCredentialsException as e:
            display_error_message(str(e), "Credentials are missing")
        except Exception as e:
            error_message = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            display_error_message(error_message)
