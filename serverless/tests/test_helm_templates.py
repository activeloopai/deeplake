#!/usr/bin/env python3
"""Helm template tests for things that can silently break production."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "pg-deeplake"
SECURE_ARGS = ["--set", "postgres.password=test-secure-password-123"]


def _render(extra_args=None):
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")
    cmd = ["helm", "template", "test", str(CHART_DIR)] + SECURE_ARGS
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.check_output(cmd, text=True)


def _render_docs(extra_args=None):
    return list(yaml.safe_load_all(_render(extra_args)))


def _find_doc(docs, kind, name_contains=None):
    for doc in docs:
        if doc and doc.get("kind") == kind:
            if name_contains is None or name_contains in doc["metadata"]["name"]:
                return doc
    return None


# Deploying with default password = production incident
def test_secure_defaults_block_empty_password():
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")
    result = subprocess.run(
        ["helm", "template", "test", str(CHART_DIR)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


# Without startup probe, liveness kills slow-booting database pods
def test_startup_probe_exists():
    docs = _render_docs()
    ss = _find_doc(docs, "StatefulSet")
    container = ss["spec"]["template"]["spec"]["containers"][0]
    assert "startupProbe" in container
    assert container["startupProbe"]["failureThreshold"] >= 20


# Without HAProxy PDB, node drain kills both proxies = total outage
def test_haproxy_pdb_when_multi_replica():
    docs = _render_docs(["--set", "haproxy.replicaCount=2"])
    pdb = _find_doc(docs, "PodDisruptionBudget", "haproxy")
    assert pdb is not None
    assert pdb["spec"]["minAvailable"] == 1


def test_haproxy_pdb_skipped_when_single_replica():
    docs = _render_docs(["--set", "haproxy.replicaCount=1"])
    pdb = _find_doc(docs, "PodDisruptionBudget", "haproxy")
    assert pdb is None


# HAProxy must know about all possible backends for scale-up to work
def test_haproxy_backends_match_max_replicas():
    rendered = _render([
        "--set", "autoscaling.enabled=true",
        "--set", "autoscaling.maxReplicas=8",
    ])
    servers = re.findall(r"server test-pg-deeplake-\d+ ", rendered)
    assert len(servers) == 8


# PG tuning params must actually flow from values, not be hardcoded
def test_pg_config_from_values():
    rendered = _render([
        "--set", "postgresConfig.sharedBuffers=4GB",
        "--set", "postgresConfig.maxConnections=500",
    ])
    assert "shared_buffers = 4GB" in rendered
    assert "max_connections = 500" in rendered
