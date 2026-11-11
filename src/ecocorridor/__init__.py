"""EcoCorridor — modelling of ecological corridors between protected areas.

The package is organised as a small, dependency-light modelling toolkit plus a
FastAPI application:

``dataset``        project loading (rasters, vectors, land-cover classes)
``resistance``     resistance / permeability surface
``costpath``       least-cost paths on a raster graph
``fragmentation``  landscape structure and connectivity indices
``corridors``      corridor network, flows and pinch points
``scenarios``      urbanisation and restoration scenarios
``analysis``       orchestrates a full run and caches the results
``render``         map layers as PNG for the browser client
``api``            HTTP API and static client
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
