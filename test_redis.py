import asyncio
from redis.asyncio.cluster import RedisCluster


async def main():
    rc = RedisCluster(
        host="redis-node-0",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    print(await rc.ping())
    await rc.close()


asyncio.run(main())
