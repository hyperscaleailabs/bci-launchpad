"""Ray runtime: cluster connection, resource selection, generic tasks/actors.

Submodules are imported lazily by callers (``from bci_platform.ray_runtime.tasks
import ...``) so importing this package stays cheap.
"""
