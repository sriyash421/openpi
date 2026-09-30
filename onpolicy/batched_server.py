"""Batched drop-in for scripts/serve_policy.py: many concurrent clients share one model call.

openpi's server runs Policy.infer one request at a time, and exclusive-process GPUs allow only one server per GPU,
so with ~23 clients per GPU inference was the rollout bottleneck. Here requests arriving within `window_ms` are
batched (up to `max_batch`, padded to a fixed bucket so only a few programs are compiled) into one sample_actions
call. Per-sample input/output transforms, the jitted sample_actions and the websocket/msgpack protocol are openpi's
own, so clients are unchanged.

    python -m onpolicy.batched_server --config pi05_libero_ctx --dir <ckpt> --port 8000
"""
import asyncio
import concurrent.futures as cf
import dataclasses
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _config

BUCKETS = (1, 4, 8, 16, 32)


@dataclasses.dataclass
class Args:
    config: str
    dir: str
    port: int = 8000
    max_batch: int = 32
    window_ms: float = 5.0


class BatchedPolicy:
    def __init__(self, policy, max_batch):
        self.p = policy
        self.buckets = [b for b in BUCKETS if b <= max_batch]

    def infer(self, obs_list):
        """Same steps as openpi Policy.infer, applied to a list of observations with one model call."""
        inputs = [self.p._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]
        n = len(inputs)
        size = next(b for b in self.buckets if b >= n)
        inputs += [inputs[-1]] * (size - n)  # pad to a fixed bucket: one compiled program per bucket
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        self.p._rng, rng = jax.random.split(self.p._rng)
        start = time.monotonic()
        actions = np.asarray(self.p._sample_actions(rng, _model.Observation.from_dict(batch), **self.p._sample_kwargs))
        ms = (time.monotonic() - start) * 1000
        states = np.asarray(batch["state"])
        out = []
        for i in range(n):
            o = self.p._output_transform({"state": states[i], "actions": actions[i]})
            o["policy_timing"] = {"infer_ms": ms, "batch": n}
            out.append(o)
        return out


async def serve(args: Args):
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.dir)
    batched = BatchedPolicy(policy, args.max_batch)
    queue = asyncio.Queue()
    executor = cf.ThreadPoolExecutor(1)  # model calls run off the event loop, one batch at a time
    loop = asyncio.get_running_loop()

    async def batcher():
        while True:
            items = [await queue.get()]
            deadline = loop.time() + args.window_ms / 1000
            while len(items) < args.max_batch and (left := deadline - loop.time()) > 0:
                try:
                    items.append(await asyncio.wait_for(queue.get(), left))
                except asyncio.TimeoutError:
                    break
            try:
                results = await loop.run_in_executor(executor, batched.infer, [obs for obs, _ in items])
                for (_, fut), r in zip(items, results):
                    fut.set_result(r)
            except Exception as e:  # fail the waiting clients rather than the server
                for _, fut in items:
                    fut.set_exception(e)

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(policy.metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
            except Exception:
                return
            fut = loop.create_future()
            await queue.put((obs, fut))
            await ws.send(packer.pack(await fut))

    asyncio.create_task(batcher())
    async with _server.serve(handler, "0.0.0.0", args.port, compression=None, max_size=None) as server:
        await server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(tyro.cli(Args)))
