"""Data generators and warehouse bootstrap helpers."""

from .synthetic import build_demo_warehouse, build_experiment_users, build_orders_daily

__all__ = ["build_demo_warehouse", "build_experiment_users", "build_orders_daily"]
