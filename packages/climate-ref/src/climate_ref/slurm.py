import importlib.util
import os
from typing import Any

HAS_REAL_SLURM = importlib.util.find_spec("pyslurm") is not None


class SlurmChecker:
    """Check and get slurm settings."""

    def __init__(self, intest: bool = False) -> None:
        if HAS_REAL_SLURM:
            import pyslurm  # type: ignore

            self.slurm_association: dict[int, Any] | None = pyslurm.db.Associations.load()
            self.slurm_partition: dict[str, Any] | None = pyslurm.Partitions.load()
            self.slurm_qos: dict[str, Any] | None = pyslurm.qos().get()
            self.slurm_node: dict[str, Any] | None = pyslurm.Nodes.load()
        elif intest:
            import pyslurm

            self.slurm_association = pyslurm.db.Associations.load()  # dict [num -> Association]
            self.slurm_partition = pyslurm.Partitions.load()  # collection
            self.slurm_qos = pyslurm.qos().get()  # dict
            self.slurm_node = pyslurm.Nodes.load()  # dict
        else:
            print("Warning: pyslurm not found. Skipping HPCExecutor config validations")
            self.slurm_association = None
            self.slurm_partition = None
            self.slurm_qos = None
            self.slurm_node = None

    def get_partition_info(self, partition_name: str) -> Any:
        """Check if a partition exists in the Slurm configuration."""
        return self.slurm_partition.get(partition_name) if self.slurm_partition else None

    def get_qos_info(self, qos_name: str) -> Any:
        """Check if a qos exists in the Slurm configuration."""
        return self.slurm_qos.get(qos_name) if self.slurm_qos else None

    def get_account_info(self, account_name: str) -> list[Any]:
        """Get all associations for an account"""
        if self.slurm_association:
            return [a for a in self.slurm_association.values() if a.account == account_name]
        else:
            return [None]

    def can_account_use_partition(self, account_name: str, partition_name: str) -> bool:
        """
        Check if an account has access to a specific partition.

        Returns
        -------
            bool: True if accessible, False if not accessible or error occurred
        """
        account_info = self.get_account_info(account_name)
        if not account_info:
            return False

        partition_info = self.get_partition_info(partition_name)

        if not partition_info:
            return False

        allowed_partitions = account_info[0].partition
        if allowed_partitions is None:
            return True
        else:
            return partition_name in allowed_partitions

    def can_account_use_qos(self, account_name: str, qos_name: str) -> bool:
        """
        Check if an account has access to a specific qos.

        Returns
        -------
            bool: True if accessible, False if not accessible or error occurred
        """
        account_info = self.get_account_info(account_name)

        if not account_info:
            return False

        qos_info = self.get_qos_info(qos_name)
        if not qos_info:
            return False

        sample_acc = account_info[0]
        user_name = os.environ["USER"]

        if user_name:
            for acc in account_info:
                if acc.user == user_name:
                    sample_acc = acc
                    break

        allowed_qoss = sample_acc.qos
        if allowed_qoss is None:
            return True
        else:
            return qos_name in allowed_qoss

    def get_partition_limits(self, partition_name: str) -> dict[str, str | int] | None:
        """
        Get time limits for a specific partition.

        Returns
        -------
            Dict with 'max_time' and 'default_time' (strings or UNLIMITED)
            or None if partition doesn't exist or error occurred
        """
        partition_info = self.get_partition_info(partition_name)
        if not partition_info:
            return None

        return {
            "max_time_minutes": partition_info.to_dict().get("max_time", 0),  # in minutes
            "default_time_minutes": partition_info.to_dict().get("default_time", 30),  # in minutes
            "max_nodes": partition_info.to_dict().get("max_node", 1),
            "total_nodes": partition_info.to_dict().get("total_nodes", 0),
            "total_cpus": partition_info.to_dict().get("total_cpus", 0),
        }

    def get_node_from_partition(self, partition_name: str) -> dict[str, str | int] | None:
        """
        Get the node information for a specific partition.

        Returns
        -------
            Dicts
        """
        partition_info = self.get_partition_info(partition_name)
        if not partition_info:
            return None

        sample_node = None

        if self.slurm_node:
            for node in self.slurm_node.values():
                if partition_name in node.partitions and "cpu" in node.available_features:
                    sample_node = node
                    break

        return {
            "cpus": int(sample_node.total_cpus) if sample_node is not None else 1,
            "cores_per_socket": int(sample_node.cores_per_socket) if sample_node is not None else 1,
            "sockets": int(sample_node.sockets) if sample_node is not None else 1,
            "threads_per_core": int(sample_node.threads_per_core) if sample_node is not None else 1,
            "real_memory": int(sample_node.real_memory) if sample_node is not None else 215,
            "node_names": sample_node.name if sample_node is not None else "unknown",
        }

    def get_qos_limits(self, qos_name: str) -> dict[str, str | int]:
        """
        Get time limits for a specific qos.

        Returns
        -------
            Dict with 'max_time' and 'default_time' (strings or UNLIMITED)
            or None if partition doesn't exist or error occurred
        """
        qos_info = self.get_qos_info(qos_name)

        return {
            "max_time_minutes": qos_info.get("max_wall_pj", 1.0e6),
            "max_jobs_pu": qos_info.get("max_jobs_pu", 1.0e6),
            "max_submit_jobs_pu": qos_info.get("max_submit_jobs_pu", 1.0e6),
            "max_tres_pj": qos_info.get("max_tres_pj").split("=")[0],
            "default_time_minutes": 120,
        }

    def check_account_partition_access_with_limits(
        self, account_name: str, partition_name: str
    ) -> dict[str, Any]:
        """
        Comprehensive check of account access and partition limits.

        Returns dictionary with all relevant information.
        """
        result = {
            "account_exists": True if self.get_account_info(account_name) else False,
            "partition_exists": True if self.get_partition_info(partition_name) else False,
            "has_access": False,
            "time_limits": None,
            "error": "none",
        }

        try:
            if result["account_exists"] and result["partition_exists"]:
                result["has_access"] = self.can_account_use_partition(account_name, partition_name)
                if result["has_access"]:
                    result["time_limits"] = self.get_partition_info(partition_name).to_dict().get("max_time")
        except Exception as e:
            result["error"] = str(e)

        return result
