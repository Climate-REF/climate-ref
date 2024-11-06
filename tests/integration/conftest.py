import pytest
from pytest_celery import CeleryBrokerCluster, RedisTestBroker


@pytest.fixture
def celery_broker_cluster(celery_redis_broker: RedisTestBroker) -> CeleryBrokerCluster:
    cluster = CeleryBrokerCluster(celery_redis_broker)
    yield cluster
    cluster.teardown()
