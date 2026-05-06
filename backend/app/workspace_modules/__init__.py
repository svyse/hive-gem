"""Workspace Modules

This package implements a lightweight "modules" system stored under the
configured WORKSPACE_ROOT (default: ~/.agentic_hive_studio/workspace).

Goals:
- allow the app (and orchestrators) to create small, reusable utilities
  without modifying the main repo
- keep the feature safe by constraining writes to WORKSPACE_ROOT/modules
"""
