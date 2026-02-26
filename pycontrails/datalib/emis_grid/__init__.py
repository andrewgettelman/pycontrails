"""Gridded aircraft emissions inventory data access.

Provides data source classes for reading gridded aircraft emissions inventories
into :class:`~pycontrails.core.met.MetDataset` objects compatible with the
pycontrails model pipeline.

Supported datasets
------------------
- :class:`AEIC` — Aviation Emissions Inventory Code (monthly gridded netCDF)

Planned datasets
----------------
- GAIA — Global Aviation Integrated Assessment (monthly and hourly gridded)

Typical usage
-------------
>>> from pycontrails.datalib.emis_grid import AEIC
>>> aeic = AEIC(
...     time=("2019-01", "2019-03"),
...     variables=["FUELBURN", "BC", "DISTANCE"],
...     data_dir="/path/to/AEIC/monthly",
... )
>>> emis = aeic.open_metdataset()
"""

from __future__ import annotations

from pycontrails.datalib.emis_grid.aeic import AEIC, DEFAULT_PRESSURE_LEVELS
from pycontrails.datalib.emis_grid.variables import (
    AEIC_DEFAULT_VARIABLES,
    AEIC_VARIABLES,
    AEICBC,
    AEICCO,
    AEICDistance,
    AEICFuelBurn,
    AEICHC,
    AEICHONO,
    AEICNO,
    AEICNO2,
    AEICOC,
)

__all__ = [
    "AEIC",
    "DEFAULT_PRESSURE_LEVELS",
    "AEIC_VARIABLES",
    "AEIC_DEFAULT_VARIABLES",
    "AEICFuelBurn",
    "AEICBC",
    "AEICDistance",
    "AEICCO",
    "AEICHC",
    "AEICHONO",
    "AEICNO",
    "AEICNO2",
    "AEICOC",
]