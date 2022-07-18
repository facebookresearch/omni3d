# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.utils.file_io import PathHandler, PathManager

__all__ = ["CubeRCNNHandler"]

class CubeRCNNHandler(PathHandler):
    """
    Resolves CubeRCNN's model zoo files. 
    """

    PREFIX = "cubercnn://"
    CUBERCNN_PREFIX = "https://dl.fbaipublicfiles.com/cubercnn/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.CUBERCNN_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(CubeRCNNHandler())