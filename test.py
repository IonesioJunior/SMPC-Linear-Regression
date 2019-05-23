import syft as sy
import torch as th
from syft.workers import WebsocketClientWorker



hook = sy.TorchHook( th )
kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": True}
bob = WebsocketClientWorker( id="bob", port=8777, **kwargs_websocket )

while 1:
    continue
