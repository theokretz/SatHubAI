class OptionsConfig:
    def __init__(self, provider, collection, ndvi_checked):
        self._provider = provider
        self._collection = collection
        self._ndvi_checked = ndvi_checked


    @property
    def collection(self):
        return self._collection

    @property
    def ndvi_checked(self):
        return self._ndvi_checked