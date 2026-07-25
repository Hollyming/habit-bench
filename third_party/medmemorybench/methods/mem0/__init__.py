import importlib.metadata

try:
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    # MedMemoryBench vendors the Mem0 source tree.  Importing that source must
    # not require an independently installed mem0ai wheel (which could also
    # silently replace the implementation being benchmarked).
    __version__ = "0+medmemorybench-vendored"

from methods.mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from methods.mem0.memory.main import Memory  # noqa
