"""Tests for local monthly and daily AEIC inventory access."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pycontrails.datalib.emis_grid import AEIC
from pycontrails.datalib.emis_grid.aeic import AEIC_PRESSURE_LEVELS


@pytest.fixture
def aeic_files(tmp_path: Path) -> Path:
    """Create small inventories with indexed daily and pressure-valued monthly levels."""
    for frequency, dates in (
        ("daily", ["2019-03-01", "2019-03-02"]),
        ("monthly", ["2019-03-16T12", "2019-04-16"]),
    ):
        for date in dates:
            timestamp = np.datetime64(date, "ns")
            level = np.arange(1, 37) if frequency == "daily" else np.array(AEIC_PRESSURE_LEVELS)
            values = np.broadcast_to(np.arange(1, 37)[None, :, None, None], (1, 36, 2, 2))
            dataset = xr.Dataset(
                {"FUELBURN": (("time", "lev", "lat", "lon"), values.astype(float))},
                coords={"time": [timestamp], "lev": level, "lat": [40.0, 41.0], "lon": [0.0, 1.0]},
            )
            dataset["lev"].attrs["units"] = "level" if frequency == "daily" else "hPa"
            dataset["FUELBURN"].attrs["units"] = "kg/m2/s"
            datetime_value = datetime.fromisoformat(date)
            filename = (
                f"AEIC_{datetime_value:%Y%m%d}.0.5x0.625.36L.nc"
                if frequency == "daily" else f"AEIC_monmean_{datetime_value:%Y%m}.nc"
            )
            dataset.to_netcdf(tmp_path / filename)
    return tmp_path


@pytest.mark.parametrize(
    ("time", "days"),
    [
        ("2019-03-01", [1]),
        (("2019-03-01", "2019-03-02"), [1, 2]),
        (("2019-03-01T08", "2019-03-02T23"), [1, 2]),
    ],
)
def test_aeic_daily_files(aeic_files: Path, time: str | tuple[str, str], days: list[int]) -> None:
    """Daily requests include endpoint days, not the following day, and remain lazy."""
    source = AEIC(
        time, variables=["FUELBURN"], pressure_levels=[85, 1020],
        data_dir=aeic_files, frequency="daily",
    )
    assert source.timesteps == [datetime(2019, 3, day) for day in days]
    result = source.open_metdataset()
    assert result.attrs["product"] == "daily"
    assert result.data.sizes["time"] == len(days)
    np.testing.assert_array_equal(
        result.data.time.values,
        np.array([f"2019-03-{day:02d}" for day in days], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(result.data.level, sorted(AEIC_PRESSURE_LEVELS))
    np.testing.assert_allclose(result.data.FUELBURN.sel(level=1005.6505).compute(), 1.0)
    np.testing.assert_allclose(result.data.FUELBURN.sel(level=85.439).compute(), 36.0)
    assert result.data.FUELBURN.attrs["units"] == "kg/m2/s"
    assert hasattr(result.data.FUELBURN.data, "__dask_graph__")


def test_aeic_monthly_default(aeic_files: Path) -> None:
    """Existing callers retain monthly filenames, internal times, and metadata."""
    source = AEIC(
        ("2019-03", "2019-04"), variables=["FUELBURN"],
        pressure_levels=[85, 1020], data_dir=aeic_files,
    )
    assert source.frequency == "monthly"
    assert source.timesteps == [datetime(2019, 3, 1), datetime(2019, 4, 1)]
    assert source.create_cachepath(source.timesteps[0]).endswith("AEIC_monmean_201903.nc")
    result = source.open_metdataset()
    assert result.attrs["product"] == "monthly"
    np.testing.assert_array_equal(
        result.data.time, np.array(["2019-03-16T12", "2019-04-16"], dtype="datetime64[ns]")
    )


def test_aeic_daily_pressure_selection(aeic_files: Path) -> None:
    """Select by pressure after converting daily level indices."""
    source = AEIC(
        "2019-03-01", variables=["FUELBURN"], pressure_levels=[1005, 1006],
        data_dir=aeic_files, frequency="daily",
    )
    result = source.open_metdataset()
    np.testing.assert_allclose(result.data.level, [1005.6505])
    np.testing.assert_allclose(result.data.FUELBURN.compute(), 1.0)


def test_aeic_daily_explicit_paths(aeic_files: Path) -> None:
    """Explicit paths continue to bypass date-based discovery."""
    source = AEIC(
        None, variables=["FUELBURN"], frequency="daily",
        paths=str(aeic_files / "AEIC_2019030*.0.5x0.625.36L.nc"),
        pressure_levels=[85, 1020],
    )
    assert source.open_metdataset().data.sizes["time"] == 2


def test_aeic_daily_missing_file(aeic_files: Path) -> None:
    """Missing daily-file errors identify the requested date and product."""
    with pytest.raises(FileNotFoundError, match="daily files.*") as error:
        AEIC(("2019-03-01", "2019-03-03"), data_dir=aeic_files, frequency="daily")
    assert "AEIC_20190303.0.5x0.625.36L.nc" in str(error.value)


def test_aeic_frequency_validation() -> None:
    """Reject unknown frequencies and distinguish product hashes."""
    with pytest.raises(ValueError, match="frequency"):
        AEIC("2019-03-01", paths=[], **{"frequency": "hourly"})
    monthly = AEIC("2019-03-01", paths=[])
    daily = AEIC("2019-03-01", paths=[], frequency="daily")
    assert monthly.hash != daily.hash


def test_aeic_daily_unknown_vertical_grid(aeic_files: Path) -> None:
    """Do not reinterpret unknown indexed vertical coordinates as pressure."""
    path = aeic_files / "AEIC_20190301.0.5x0.625.36L.nc"
    with xr.open_dataset(path) as dataset:
        source = AEIC(None, variables=["FUELBURN"], paths=str(path), frequency="daily")
        with pytest.raises(ValueError, match="standard 1..36"):
            source.open_metdataset(dataset=dataset.isel(lev=slice(0, 35)))