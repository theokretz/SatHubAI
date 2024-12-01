from .landsat_processor import LandsatProcessor
from .sentinel_processor import SentinelProcessor


class ProcessorFactory:
    @staticmethod
    def create_processor(config, provider, collection):
        """factory method to create the appropriate processor based on the collection"""
        if collection.startswith("sentinel"):
            return SentinelProcessor(config, provider, collection)
        elif collection.startswith("landsat"):
            return LandsatProcessor(config, provider, collection)
        else:
            raise ValueError(f"Unsupported collection: {collection}")