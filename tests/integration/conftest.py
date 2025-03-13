import redis
from pytest_docker_tools import container, fetch, wrappers


class RedisContainer(wrappers.Container):
    def ready(self) -> bool:
        if super().ready() and len(self.ports["6379/tcp"]) > 0:
            print(f"Redis using port:{self.ports['6379/tcp'][0]}")
            # Perform a simple ping to check if the server is ready
            r = redis.Redis(host="localhost", port=self.ports["6379/tcp"][0])
            try:
                return r.ping()
            except redis.ConnectionError:
                return False

        return False

    def connection_url(self, database: int = 0) -> str:
        port = self.ports["6379/tcp"][0]
        return f"redis://localhost:{port}/{database}"


redis_image = fetch(repository="redis:7")

redis_container = container(
    image="{redis_image.id}",
    scope="session",
    ports={
        "6379/tcp": None,
    },
    wrapper_class=RedisContainer,
)
