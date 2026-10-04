"""Wait for both Cube cache shards to encode 100 episodes, then profile once."""
import time
from . import core,telemetry

if __name__=='__main__':
    if core.ROOT!=core.Path('/workspace/x1_cube'):raise RuntimeError('Cube root required')
    print('WAITING_BOTH_CUBE_CACHE_SHARDS_100_EPISODES',flush=True)
    while True:
        paths=[core.ROOT/'state'/f'cube_cache_progress_{s}.json' for s in (0,1)]
        if all(p.exists() and core.read(p)['complete']>=100 for p in paths):break
        time.sleep(10)
    telemetry.run('cube',samples=7,interval=10)
