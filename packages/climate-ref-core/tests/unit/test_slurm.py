from unittest.mock import MagicMock

import pytest

from climate_ref_core.slurm import SlurmChecker


@pytest.fixture
def create_mock_association():
    """Factory fixture for creating mock objects"""

    def _create_mock_association(**kwargs):
        """
        Creates a mock PySlurm association object with configurable attributes.

        Args:
            **kwargs: Key-value pairs to set as attributes on the mock association

        Returns:
            MagicMock: Configured mock association object
        """
        # Default values for association attributes
        defaults = {
            "account": "test_account",
            "cluster": "test_cluster",
            "comment": None,
            "group_jobs": "UNLIMITED",
            "group_jobs_accrue": "UNLIMITED",
            "group_submit_jobs": "UNLIMITED",
            "group_tres": None,
            "group_tres_mins": None,
            "group_tres_run_mins": None,
            "group_wall_time": "UNLIMITED",
            "is_default": False,
            "max_jobs_accrue": 4,
            "max_submit_jobs": "UNLIMITED",
            "max_tres_mins_per_job": None,
            "max_tres_per_job": None,
            "max_tres_per_node": None,
            "max_tres_run_mins_per_user": None,
            "max_wall_time_per_job": "UNLIMITED",
            "max_jobs": 10,
            "max_wall_pj": "1-00:00:00",
            "min_priority_threshold": "UNLIMITED",
            "partition": "test_partition",
            "shares": 100,
            "qos": ["normal"],
            "priority": 1000,
            "id": 1,
            "lft": 0,
            "parent_acct": None,
            "parent_account_id": 4,
            "rgt": 0,
            "user": "test_user",
        }

        # Update defaults with any provided kwargs
        defaults.update(kwargs)

        # Create the mock association
        mock_assoc = MagicMock()

        # Set attributes on the mock
        for key, value in defaults.items():
            setattr(mock_assoc, key, value)

        return mock_assoc

    return _create_mock_association


@pytest.fixture
def create_mock_partition():
    """Factory fixture for creating mock objects"""

    def _create_mock_partition(**kwargs):
        """
        Creates a mock PySlurm partition object with configurable attributes.

        Args:
            **kwargs: Key-value pairs to set as attributes on the mock partition

        Returns:
            MagicMock: Configured mock partition object
        """

        # Default values for partition attributes
        defaults = {
            "allow_root_jobs": False,
            "allowed_accounts": ["ALL"],
            "allowed_groups": ["ALL"],
            "allowed_qos": ["ALL"],
            "allowed_submit_nodes": ["ALL"],
            "alternate": None,
            "cluster": "test_cluster",
            "cpu_binding": None,
            "default_cpus_per_gpu": None,
            "default_memory_per_cpu": None,
            "default_memory_per_gpu": None,
            "default_memory_per_node": "UNLIMITED",
            "default_time": 720,
            "denied_accounts": ["ALL"],
            "denied_qos": ["ALL"],
            "is_default": False,
            "is_hidden": False,
            "is_root_only": False,
            "is_user_exclusive": False,
            "least_loaded_nodes_scheduling": False,
            "max_cpus_per_node": "UNLIMITED",
            "max_cpus_per_socket": "UNLIMITED",
            "max_memory_per_cpu": None,
            "max_memory_per_node": "UNLIMITED",
            "max_nodes": "UNLIMITED",
            "max_time": 7200,
            "min_nodes": 0,
            "name": "batch",
            "nodes": "baseline[3-140]",
            "nodesets": [],
            "over_time_limit": None,
            "oversubscribe": "EXCLUSIVE",
            "preempt_mode": "OFF",
            "preemption_grace_time": None,
            "priority_job_factor": 1,
            "priority_tier": 1,
            "qos": None,
            "requires_reservation": False,
            "select_type_parameters": [],
            "state": "UP",
            "total_cpus": 17664,
            "total_nodes": 138,
        }

        # Update defaults with any provided kwargs
        defaults.update(kwargs)

        # Create the mock partition
        mock_part = MagicMock()

        # Set attributes on the mock
        for key, value in defaults.items():
            setattr(mock_part, key, value)

        return mock_part

    return _create_mock_partition


@pytest.fixture
def create_mock_qos():
    """Factory fixture for creating mock objects"""

    def _create_mock_qos(**kwargs):
        """
        Creates a mock PySlurm QoS object with configurable attributes.

        Args:
            **kwargs: Key-value pairs to set as attributes on the mock QoS

        Returns:
            MagicMock: Configured mock QoS object
        """
        # Default values for QoS attributes
        defaults = {
            "name": "normal",
            "description": "Default Quality of Service",
            "flags": 0,
            "grace_time": 0,
            "grp_jobs": 4294967295,
            "grp_submit_jobs": 4294967295,
            "grp_tres": None,
            "grp_tres_mins": None,
            "grp_tres_run_mins": None,
            "grp_wall": 4294967295,
            "max_jobs_pu": 4294967295,
            "max_submit_jobs_pu": 4294967295,
            "max_tres_mins_pj": None,
            "max_tres_pj": None,
            "max_tres_pn": None,
            "max_tres_pu": None,
            "max_tres_run_mins_pu": None,
            "max_wall_pj": 4294967295,
            "min_tres_pj": None,
            "preempt_mode": "OFF",
            "priority": 0,
            "usage_factor": 1.0,
            "usage_thres": 4294967295.0,
        }

        # Update defaults with any provided kwargs
        defaults.update(kwargs)

        # Create the mock QoS
        mock_qos = MagicMock()

        # Set attributes on the mock
        for key, value in defaults.items():
            setattr(mock_qos, key, value)

        return mock_qos

    return _create_mock_qos


@pytest.fixture
def create_mock_node():
    """Factory fixture for creating mock objects"""

    def _create_mock_node(**kwargs):
        """
        Creates a mock PySlurm node object with configurable attributes.

        Args:
            **kwargs: Key-value pairs to set as attributes on the mock node

        Returns:
            MagicMock: Configured mock node object
        """
        # Default values for node attributes
        defaults = {
            "active_features": [],
            "address": "test_address",
            "allocated_cpus": 0,
            "allocated_gres": {},
            "allocated_memory": 0,
            "allocated_tres": {},
            "architecture": "x86_64",
            "available_features": [],
            "avg_watts": 0,
            "bcast_address": None,
            "boards": 1,
            "boot_time": 1747765065,
            "cap_watts": 0,
            "cluster": "baseline",
            "comment": None,
            "configured_gres": {},
            "cores_per_socket": 64,
            "cores_reserved_for_system": None,
            "cpu_binding": None,
            "cpu_load": 0.18,
            "current_watts": 0,
            "effective_cpus": 128,
            "external_sensors": {"joules_total": None, "current_watts": None, "temperature": None},
            "extra": None,
            "free_memory": 231147,
            "hostname": "baseline101",
            "idle_cpus": 128,
            "idle_memory": 256000,
            "last_busy_time": 1747789216,
            "mcs_label": None,
            "memory_reserved_for_system": None,
            "name": "baseline101",
            "next_state": None,
            "operating_system": "Linux 4.18.0-553.46.1.el8_10.x86_64 #1 SMP Sat Mar 15 01:37:33 EDT 2025",
            "owner": None,
            "partitions": ["batch", "batch_low_memory"],
            "real_memory": 256000,
            "reason": None,
            "reason_time": None,
            "reason_user": None,
            "slurm_version": "23.02.7",
            "slurmd_port": 6818,
            "slurmd_start_time": 1747765241,
            "sockets": 2,
            "state": "IDLE",
            "temporary_disk": None,
            "threads_per_core": 1,
            "total_cpus": 128,
            "weight": 1,
        }

        # Update defaults with any provided kwargs
        defaults.update(kwargs)

        # Create the mock node
        mock_node = MagicMock()

        # Set attributes on the mock
        for key, value in defaults.items():
            setattr(mock_node, key, value)

        return mock_node

    return _create_mock_node


def test_slurm_checker(
    create_mock_association, create_mock_partition, create_mock_qos, create_mock_node, mocker
):
    # Setup mock data
    associations = {
        1: create_mock_association(account="climate_ref1", partition="cpu"),
        2: create_mock_association(account="climate_ref2", partition="gpu"),
    }

    partitions = {
        "normal": create_mock_partition(name="nomral"),
        "batch": create_mock_partition(name="batch"),
    }

    qoss = {
        "normal": create_mock_qos(name="normal"),
    }
    nodes = {
        "node0001": create_mock_node(name="node0001"),
    }

    # Patch the Slurm dependencies
    mocker.patch("pyslurm.db.Associations.load", return_value=associations)
    mocker.patch("pyslurm.Partitions.load", return_value=partitions)
    mocker.patch("pyslurm.Nodes.load", return_value=nodes)

    mocker.patch("pyslurm.qos")  # First mock the qos class
    mock_qos_instance = MagicMock()
    mock_qos_instance.get.return_value = qoss  # Your mock qos dictionary
    mocker.patch("pyslurm.qos", return_value=mock_qos_instance)

    # Create and test the checker
    checker = SlurmChecker()

    print(checker.can_account_use_partition("climate_ref1", "cpu"), "xxxx")
    assert checker.can_account_use_partition("climate_ref1", "cpu") is True
    assert checker.can_account_use_partition("climate_ref2", "gpu") is True
    assert checker.can_account_use_partition("climate", "nonexistent") is False
