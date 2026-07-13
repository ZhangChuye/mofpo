"""
DDP-specific AsyncVectorEnv variant.

Keeps robust close behavior for multi-process async eval workers where partial
initialization/teardown can happen more often.
"""

from mof.gym_util.async_vector_env import (
    AsyncVectorEnv,
    AsyncState,
    logger,
    mp,
)


class AsyncVectorEnvDDP(AsyncVectorEnv):
    def close_extras(self, timeout=None, terminate=False):
        # Be tolerant to partially constructed instances (e.g. init failed early)
        # where __del__ may still call close().
        state = getattr(self, "_state", AsyncState.DEFAULT)
        parent_pipes = getattr(self, "parent_pipes", None)
        processes = getattr(self, "processes", None)
        if parent_pipes is None or processes is None:
            return

        timeout = 0 if terminate else timeout
        try:
            if state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending "
                    "call to `{0}` to complete.".format(state.value)
                )
                function = getattr(self, "{0}_wait".format(state.value))
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in processes:
                if process is None:
                    continue
                if getattr(process, "_popen", None) is None:
                    continue
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in processes:
            if process is None:
                continue
            if getattr(process, "_popen", None) is None:
                continue
            process.join()
