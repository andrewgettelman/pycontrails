"""AEIC (Aviation Emissions Inventory Code) gridded monthly data access.

Provides a :class:`AEIC` data source class that reads monthly mean gridded
aircraft emissions from local netCDF files and returns them as a
:class:`~pycontrails.core.met.MetDataset`, compatible with the rest of the
pycontrails model pipeline (e.g. UPCoM).

File naming convention expected on disk::

    <data_dir>/AEIC_monmean_YYYYMM.nc

Examples
--------
Load three months with the default pressure range (150–400 hPa):

>>> from pycontrails.datalib.emis_grid import AEIC
>>> aeic = AEIC(
...     time=("2019-01", "2019-03"),
...     variables=["FUELBURN", "BC", "DISTANCE"],
...     data_dir="/path/to/AEIC/monthly",
... )
>>> emis = aeic.open_metdataset()

Restrict to a specific list of pressure levels (hPa):

>>> aeic = AEIC(
...     time=("2019-01", "2019-03"),
...     variables=["FUELBURN", "BC", "DISTANCE"],
...     data_dir="/path/to/AEIC/monthly",
...     pressure_levels=[200, 250, 300],
... )
>>> emis = aeic.open_metdataset()
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import sys
from datetime import datetime
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails.core import cache as cache_module
from pycontrails.core.met import MetDataset
from pycontrails.datalib._met_utils import metsource
from pycontrails.datalib.emis_grid.variables import (
    AEIC_DEFAULT_VARIABLES,
    AEIC_VARIABLES,
)

LOG = logging.getLogger(__name__)

#: Pressure levels (hPa) selected by default — aviation cruise band.
DEFAULT_PRESSURE_LEVELS: list[int] = [150, 175, 200, 225, 250, 275, 300, 350, 400]


def _parse_monthly_timesteps(time: metsource.TimeInput) -> list[datetime]:
    """Parse *time* input into a list of month-start :class:`datetime` objects.

    Unlike :func:`~pycontrails.datalib._met_utils.metsource.parse_timesteps`, this
    function uses the ``"MS"`` (MonthBegin) pandas offset which is a non-fixed
    frequency and cannot be used with ``Timestamp.floor()``.

    Parameters
    ----------
    time : TimeInput
        A single date-like value or a (start, end) pair.  The start is
        snapped to the first day of its month; months are enumerated up to
        and including the month containing the end value.

    Returns
    -------
    list[datetime]
        Sorted list of month-start datetimes, e.g.
        ``[datetime(2019, 1, 1), datetime(2019, 2, 1), datetime(2019, 3, 1)]``.

    Raises
    ------
    ValueError
        If *time* cannot be parsed or has more than two elements.
    """
    if isinstance(time, str | datetime | pd.Timestamp | np.datetime64):
        t0 = t1 = pd.to_datetime(time)
    else:
        seq = list(time)
        if len(seq) == 1:
            t0 = t1 = pd.to_datetime(seq[0])
        elif len(seq) == 2:
            t0, t1 = pd.to_datetime(seq[0]), pd.to_datetime(seq[1])
        else:
            raise ValueError(
                f"time must be a single value or a (start, end) pair, got {time!r}"
            )

    # Snap each bound to the first day of its month, then enumerate
    date_range = pd.date_range(
        start=t0.to_period("M").to_timestamp(),
        end=t1.to_period("M").to_timestamp(),
        freq="MS",
    )
    return date_range.to_pydatetime().tolist()


class AEIC(metsource.MetDataSource):
    """Data source for AEIC monthly gridded aircraft emissions.

    Reads local netCDF files produced by the Aviation Emissions Inventory Code
    (AEIC, Eastham & Wofsy) and returns a :class:`MetDataset` with dimensions
    ``(longitude, latitude, level, time)`` compatible with the pycontrails
    interpolation and model infrastructure.

    Parameters
    ----------
    time : metsource.TimeInput | None
        The time range for data retrieval.  Pass a single month string
        (``"2019-01"``) or a (start, end) pair (``("2019-01", "2019-12")``).
        Months are enumerated at monthly ("MS") frequency.
        If *None*, ``paths`` must be provided.
    variables : metsource.VariableInput
        Variable names to load.  Accepts short names (``"FUELBURN"``,
        ``"BC"``, ``"DISTANCE"``), standard names, or
        :class:`~pycontrails.core.met.MetVariable` instances.
        Supported variables: FUELBURN, BC, DISTANCE, CO, HC, HONO, NO,
        NO2, OC.
    pressure_levels : metsource.PressureLevelInput | None, optional
        Pressure levels (hPa) to retain from the 36-level AEIC grid.
        Pass a single value or a list.  If *None* (default), the range
        150–400 hPa is used.  Pass the special value ``-1`` for
        surface/single-level data (not typical for AEIC).
    data_dir : str | pathlib.Path
        Directory containing the AEIC monthly netCDF files.
        Files are expected to be named ``AEIC_monmean_YYYYMM.nc``.
    paths : str | list[str] | pathlib.Path | list[pathlib.Path] | None, optional
        Explicit file paths to load, overriding the ``data_dir`` /
        time-based file discovery.  Supports glob patterns.
        Defaults to *None*.
    cachestore : cache.CacheStore | None, optional
        Unused for AEIC (data is local).  Retained for API compatibility
        with other :class:`~pycontrails.datalib._met_utils.metsource.MetDataSource`
        subclasses.  Defaults to *None*.

    Raises
    ------
    FileNotFoundError
        Raised at construction time if any expected monthly file is not
        present in ``data_dir``.
    ValueError
        Raised if ``time`` is *None* and ``paths`` is also *None*.

    Examples
    --------
    >>> aeic = AEIC(
    ...     time=("2019-01", "2019-03"),
    ...     variables=["FUELBURN", "BC", "DISTANCE"],
    ...     data_dir="/data/AEIC/monthly",
    ... )
    >>> emis = aeic.open_metdataset()
    >>> emis.data.dims
    Frozen({'longitude': 576, 'latitude': 361, 'level': 9, 'time': 3})
    """

    __slots__ = ("cachestore", "data_dir")

    #: Root directory containing AEIC monthly netCDF files.
    data_dir: pathlib.Path

    def __init__(
        self,
        time: metsource.TimeInput | None,
        variables: metsource.VariableInput = AEIC_DEFAULT_VARIABLES,
        pressure_levels: metsource.PressureLevelInput | None = None,
        data_dir: str | pathlib.Path = ".",
        paths: str | list[str] | pathlib.Path | list[pathlib.Path] | None = None,
        cachestore: cache_module.CacheStore | None = None,
    ) -> None:
        if time is None and paths is None:
            raise ValueError("Parameter 'time' must be provided when 'paths' is None.")

        # Store data directory
        self.data_dir = pathlib.Path(data_dir)

        # Cachestore (unused, kept for API compatibility)
        self.cachestore = cachestore

        # Paths override
        self.paths = paths

        # Grid spacing — not fixed for AEIC, set to None
        self.grid = None

        # Parse timesteps at monthly (month-start) frequency.
        # parse_timesteps() uses Timestamp.floor() which doesn't support the
        # non-fixed "MS" offset, so we use _parse_monthly_timesteps instead.
        self.timesteps = _parse_monthly_timesteps(time) if time is not None else []

        # Parse variables against supported list
        self.variables = metsource.parse_variables(variables, AEIC_VARIABLES)

        # Parse pressure levels
        if pressure_levels is None:
            self.pressure_levels = list(DEFAULT_PRESSURE_LEVELS)
        else:
            self.pressure_levels = metsource.parse_pressure_levels(pressure_levels)

        # Validate that all expected files exist (only when using data_dir)
        if self.paths is None:
            self._validate_files()

    def _validate_files(self) -> None:
        """Check that all expected monthly files are present in :attr:`data_dir`.

        Raises
        ------
        FileNotFoundError
            If any expected file is missing.
        """
        missing = []
        for t in self.timesteps:
            fp = pathlib.Path(self.create_cachepath(t))
            if not fp.exists():
                missing.append(str(fp))

        if missing:
            missing_str = "\n  ".join(missing)
            raise FileNotFoundError(
                f"The following AEIC monthly files were not found:\n  {missing_str}"
            )

    # ------------------------------------------------------------------
    # MetDataSource interface
    # ------------------------------------------------------------------

    @property
    def hash(self) -> str:
        """Generate a unique hash for this datasource.

        Returns
        -------
        str
            SHA-1 hash string.
        """
        hashstr = (
            f"{self.__class__.__name__}"
            f"{self.timesteps}"
            f"{self.variable_shortnames}"
            f"{self.pressure_levels}"
            f"{self.data_dir}"
        )
        return hashlib.sha1(bytes(hashstr, "utf-8")).hexdigest()

    @property
    def pressure_level_variables(self) -> list:
        """Return all AEIC pressure-level variables.

        Returns
        -------
        list[MetVariable]
            All variables defined in :data:`AEIC_VARIABLES`.
        """
        return AEIC_VARIABLES

    @override
    def create_cachepath(self, t: datetime) -> str:
        """Return the expected path to the AEIC monthly file for datetime *t*.

        Parameters
        ----------
        t : datetime
            Month represented by the file.  Only ``year`` and ``month`` are used.

        Returns
        -------
        str
            Absolute path to the file, e.g.
            ``/data/AEIC/monthly/AEIC_monmean_201901.nc``.
        """
        fname = f"AEIC_monmean_{t.year}{t.month:02d}.nc"
        return str(self.data_dir / fname)

    @override
    def download_dataset(self, times: list[datetime]) -> None:
        """No-op: AEIC data is sourced from local files only.

        Parameters
        ----------
        times : list[datetime]
            Ignored.
        """
        LOG.debug(
            "AEIC.download_dataset called but AEIC data is local only — no download performed."
        )

    @override
    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """No-op: AEIC data is sourced from local files only.

        Parameters
        ----------
        dataset : xr.Dataset
            Ignored.
        """
        LOG.debug("AEIC.cache_dataset called but caching is not supported for local AEIC data.")

    @override
    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set AEIC metadata attributes on *ds*.

        Parameters
        ----------
        ds : xr.Dataset | MetDataset
            Dataset to annotate in-place.
        """
        ds.attrs.update(
            provider="AEIC",
            dataset="AEIC",
            product="monthly",
        )

    @override
    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open AEIC emissions as a :class:`MetDataset`.

        The returned dataset has dimensions ``(longitude, latitude, level, time)``
        and contains only the requested variables at the requested pressure levels.
        Coordinate names are normalised from the AEIC native names
        (``lon``, ``lat``, ``lev``) to pycontrails conventions
        (``longitude``, ``latitude``, ``level``).

        Parameters
        ----------
        dataset : xr.Dataset | None, optional
            Pre-loaded xarray Dataset.  If provided, file discovery is
            skipped and this dataset is processed directly.
        xr_kwargs : dict[str, Any] | None, optional
            Extra keyword arguments forwarded to :func:`xarray.open_mfdataset`
            when opening files.
        **kwargs : Any
            Additional keyword arguments passed to the :class:`MetDataset`
            constructor (e.g. ``wrap_longitude``).

        Returns
        -------
        MetDataset
            Gridded emissions dataset with dimensions
            ``(longitude, latitude, level, time)``.
        """
        xr_kwargs = xr_kwargs or {}

        if dataset is not None:
            ds = dataset
        elif self.paths is not None:
            LOG.debug("Opening AEIC data from explicit paths: %s", self.paths)
            ds = self.open_dataset(self.paths, **xr_kwargs)
        else:
            file_paths = [self.create_cachepath(t) for t in self.timesteps]
            LOG.debug("Opening AEIC data from files: %s", file_paths)
            ds = self.open_dataset(file_paths, **xr_kwargs)

        ds = self._process_dataset(ds)

        mds = MetDataset(ds, **kwargs)
        self.set_metadata(mds)
        return mds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalise an AEIC :class:`xr.Dataset` for use with :class:`MetDataset`.

        Steps performed:

        1. Rename native AEIC dimensions to pycontrails conventions.
        2. Select requested variables only.
        3. Select requested pressure levels.
        4. Ensure time is encoded as ``datetime64[ns]``.
        5. Drop extraneous coordinate variables (``time_bnds``, ``bnds``).

        Parameters
        ----------
        ds : xr.Dataset
            Raw dataset as loaded from AEIC netCDF files.

        Returns
        -------
        xr.Dataset
            Normalised dataset ready for :class:`MetDataset` construction.
        """
        # 1. Rename dimensions/coordinates to pycontrails conventions
        rename_map: dict[str, str] = {}
        if "lon" in ds.dims:
            rename_map["lon"] = "longitude"
        if "lat" in ds.dims:
            rename_map["lat"] = "latitude"
        if "lev" in ds.dims:
            rename_map["lev"] = "level"
        if rename_map:
            ds = ds.rename(rename_map)

        # 2. Select requested variables (by short_name)
        var_names = self.variable_shortnames
        missing = set(var_names) - set(ds.data_vars)
        if missing:
            raise KeyError(
                f"The following requested variables are not present in the AEIC dataset: "
                f"{sorted(missing)}. Available variables: {sorted(ds.data_vars)}"
            )
        ds = ds[var_names]

        # 3. Select pressure levels
        #    The AEIC 'lev' coordinate is in hPa; after renaming it is 'level'.
        #    We use a boolean mask to support both ascending and descending level coords.
        if self.pressure_levels != [-1]:
            pl_min = float(min(self.pressure_levels))
            pl_max = float(max(self.pressure_levels))
            level_vals = ds["level"].values
            requested = np.array(self.pressure_levels, dtype=float)
            # Try exact selection first
            available = set(level_vals.tolist())
            exact_match = all(float(p) in available for p in requested)
            if exact_match:
                ds = ds.sel(level=requested)
            else:
                # Fall back: keep all levels within [pl_min, pl_max] (inclusive)
                mask = (level_vals >= pl_min) & (level_vals <= pl_max)
                if not mask.any():
                    raise ValueError(
                        f"No AEIC levels found in the range [{pl_min}, {pl_max}] hPa. "
                        f"Available levels: {sorted(level_vals.tolist())}"
                    )
                LOG.warning(
                    "Some requested pressure levels are not present exactly in the AEIC "
                    "dataset.  Selecting all levels in the range [%s, %s] hPa instead.",
                    pl_min,
                    pl_max,
                )
                ds = ds.isel(level=mask)

        # 4. Ensure time is datetime64[ns]
        if "time" in ds.coords:
            ds["time"] = ds["time"].astype("datetime64[ns]")

        # 5. Drop auxiliary coordinate variables that are not needed downstream.
        for drop_var in ("time_bnds", "bnds"):
            if drop_var in ds:
                ds = ds.drop_vars(drop_var)

        return ds