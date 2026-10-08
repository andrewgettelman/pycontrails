"""Test models and methods pertaining to ISSR, SAC, and PCR."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pycontrails import Flight, GeoVectorDataset, JetA, MetDataArray, MetDataset
from pycontrails.core.met import originates_from_ecmwf
from pycontrails.models import sac
from pycontrails.models.humidity_scaling import (
    ConstantHumidityScaling,
    ExponentialBoostHumidityScaling,
)
from pycontrails.models.issr import ISSR
from pycontrails.models.pcr import PCR
from pycontrails.models.sac import SAC
from pycontrails.models.upcom import UPCOM
from pycontrails.physics import constants, thermo, units
from tests import _deprecated

# TODO: Include tests for all simple models here
# 1) Downselect met results in copy
# 2) Check downselect met bounds
# 4) Default params for SAC, etc


@pytest.fixture
def upcom_met() -> MetDataset:
    """Small grid for emission ingestion tests without external meteorology."""
    coords = {
        "longitude": [0.0, 1.0],
        "latitude": [40.0, 41.0],
        "level": [200.0, 250.0],
        "time": np.array(["2022-03-01T00", "2022-03-01T01"], dtype="datetime64[ns]"),
    }
    return MetDataset(
        xr.Dataset(
            {
                "air_temperature": (tuple(coords), np.full((2, 2, 2, 2), 225.0)),
                "specific_humidity": (tuple(coords), np.full((2, 2, 2, 2), 1e-4)),
            },
            coords=coords,
        )
    )


def test_upcom_emissions_mapping(upcom_met: MetDataset) -> None:
    """Mapped rates preserve values, units, NaNs, laziness, and input data."""
    template = upcom_met.data["air_temperature"]
    fuel = xr.full_like(template, 2.0)
    fuel.values[0, 0, 0, 0] = np.nan
    fuel.attrs = {"units": "kg/m2/s", "long_name": "Inventory fuel burn"}
    emissions = MetDataset(
        xr.Dataset(
            {
                "FUELBURN": fuel.chunk({"longitude": 1}),
                "DISTANCE": xr.full_like(template, 3.0).assign_attrs(units="km/m2/s"),
                "BC": xr.full_like(template, 4.0).assign_attrs(units="kg/m2/s"),
            }
        ).assign_coords(altitude=("level", [10000.0, 9000.0]))
    )
    original_met = upcom_met.data.copy(deep=True)
    original_emissions = emissions.data.copy(deep=True)
    mapping = {"fuel_rate": "FUELBURN", "distance_rate": "DISTANCE", "particle_rate": "BC"}
    model = UPCOM(upcom_met, emissions=emissions, emissions_mapping=mapping)
    mapping["fuel_rate"] = "changed_after_construction"
    result = model.eval()

    for canonical_name, input_name in model.emissions_mapping.items():
        np.testing.assert_allclose(result.data[canonical_name], emissions.data[input_name])
        assert result.data[canonical_name].attrs["units"] == emissions.data[input_name].attrs["units"]
        assert result.data[canonical_name].attrs["emissions_source_variable"] == input_name
    assert hasattr(result.data["fuel_rate"].data, "__dask_graph__")
    assert np.isnan(result.data["fuel_rate"].isel(longitude=0, latitude=0, level=0, time=0))
    xr.testing.assert_identical(upcom_met.data, original_met)
    xr.testing.assert_identical(emissions.data, original_emissions)
    baseline = UPCOM(upcom_met).eval()
    for variable in ("sac", "issr", "potential_persistent_contrail"):
        xr.testing.assert_identical(result.data[variable], baseline.data[variable])


@pytest.mark.parametrize("mapping", [None, {}, {"fuel_rate": "fuel_rate"}])
def test_upcom_emissions_defaults(upcom_met: MetDataset, mapping: dict | None) -> None:
    """Canonical lookup and partial mappings produce the expected zero defaults."""
    emissions = MetDataset(
        xr.Dataset({"fuel_rate": xr.full_like(upcom_met.data["air_temperature"], 2.0)})
    )
    result = UPCOM(upcom_met, emissions=emissions, emissions_mapping=mapping).eval()
    np.testing.assert_allclose(result.data["fuel_rate"], 0.0 if mapping == {} else 2.0)
    for name in ("distance_rate", "particle_rate"):
        xr.testing.assert_equal(result.data[name], xr.zeros_like(result.data["air_temperature"]).rename(name))


def test_upcom_without_emissions(upcom_met: MetDataset) -> None:
    """All rates are zero when no inventory is supplied."""
    result = UPCOM(upcom_met).eval()
    for name in ("fuel_rate", "distance_rate", "particle_rate"):
        np.testing.assert_array_equal(result.data[name], 0.0)
    np.testing.assert_array_equal(result.data["contrail_flag"], 0.0)
    np.testing.assert_array_equal(result.data["persistent_contrail"], 0.0)


@pytest.mark.parametrize("issr_holds", [False, True])
@pytest.mark.parametrize(
    ("fuel_rate", "distance_rate", "sac_holds", "expected"),
    [
        (1.0, 1.0, True, 1.0),
        (1.0, 0.0, True, 0.0),
        (0.0, 1.0, True, 0.0),
        (0.0, 0.0, True, 0.0),
        (-1.0, 1.0, True, 0.0),
        (1.0, -1.0, True, 0.0),
        (np.nan, 1.0, True, 0.0),
        (1.0, np.nan, True, 0.0),
        (1.0, 1.0, False, 0.0),
    ],
)
def test_upcom_contrail_flag(
    upcom_met: MetDataset,
    fuel_rate: float,
    distance_rate: float,
    sac_holds: bool,
    expected: float,
    issr_holds: bool,
) -> None:
    """Emission-gated formation requires SAC; persistence additionally requires ISSR."""
    template = upcom_met.data["air_temperature"]
    upcom_met.data["air_temperature"] = xr.full_like(template, 200.0 if sac_holds else 260.0)
    emissions = MetDataset(
        xr.Dataset(
            {
                "fuel_rate": xr.full_like(template, fuel_rate),
                "distance_rate": xr.full_like(template, distance_rate),
            }
        ).chunk({"longitude": 1})
    )
    result = UPCOM(
        upcom_met, emissions=emissions, rhi_threshold=1.0 if issr_holds else 1e6
    ).eval()
    np.testing.assert_array_equal(result.data["sac"], float(sac_holds))
    np.testing.assert_array_equal(result.data["issr"], float(issr_holds and sac_holds))
    np.testing.assert_array_equal(result.data["particle_rate"], 0.0)
    np.testing.assert_array_equal(result.data["contrail_flag"], expected)
    assert hasattr(result.data["contrail_flag"].data, "__dask_graph__")
    assert result.data["contrail_flag"].dims == result.data["sac"].dims
    assert result.data["contrail_flag"].dtype == result.data["sac"].dtype
    assert result.data["contrail_flag"].attrs["units"] == "dimensionless"
    np.testing.assert_array_equal(result.data["persistent_contrail"], expected if issr_holds else 0.0)
    assert hasattr(result.data["persistent_contrail"].data, "__dask_graph__")
    assert result.data["persistent_contrail"].dims == result.data["contrail_flag"].dims
    assert result.data["persistent_contrail"].dtype == result.data["contrail_flag"].dtype
    assert result.data["persistent_contrail"].attrs["units"] == "dimensionless"


@pytest.mark.parametrize("dimension", ["longitude", "latitude", "level", "time"])
def test_upcom_emissions_grid_mismatch(upcom_met: MetDataset, dimension: str) -> None:
    """Reject shifted coordinates instead of silently interpolating or aligning."""
    emissions = xr.Dataset({"fuel_rate": xr.zeros_like(upcom_met.data["air_temperature"])})
    offset = np.timedelta64(1, "h") if dimension == "time" else 0.01
    emissions = emissions.assign_coords({dimension: emissions[dimension] + offset})
    with pytest.raises(ValueError, match=dimension):
        UPCOM(upcom_met, emissions=MetDataset(emissions))


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ({"fuelmass": "FUELBURN"}, ValueError, "Unknown canonical"),
        ({"fuel_rate": "typo"}, KeyError, "missing"),
        ({"fuel_rate": ""}, ValueError, "Invalid input"),
    ],
)
def test_upcom_emissions_invalid_mapping(
    upcom_met: MetDataset, mapping: dict, error: type[Exception], message: str
) -> None:
    """Explicit mapping errors must not silently produce zero emissions."""
    emissions = MetDataset(
        xr.Dataset({"FUELBURN": xr.zeros_like(upcom_met.data["air_temperature"])})
    )
    with pytest.raises(error, match=message):
        UPCOM(upcom_met, emissions=emissions, emissions_mapping=mapping)


def test_upcom_emissions_missing_dataset(upcom_met: MetDataset) -> None:
    """Explicit mappings require an inventory."""
    with pytest.raises(KeyError, match="missing"):
        UPCOM(upcom_met, emissions_mapping={"fuel_rate": "FUELBURN"})


def test_upcom_emissions_variable_dimensions(upcom_met: MetDataset) -> None:
    """Reject fields missing a grid dimension even when dataset coordinates match."""
    emissions = MetDataset(
        xr.Dataset(
            {"fuel_rate": upcom_met.data["air_temperature"].isel(time=0, drop=True)},
            coords=upcom_met.data.coords,
        )
    )
    with pytest.raises(ValueError, match="span all grid dimensions"):
        UPCOM(upcom_met, emissions=emissions).eval()


def test_upcom_emissions_evaluation_grid(upcom_met: MetDataset) -> None:
    """Reject a different source grid or a trajectory when an inventory is supplied."""
    emissions = MetDataset(
        xr.Dataset({"fuel_rate": xr.zeros_like(upcom_met.data["air_temperature"])})
    )
    model = UPCOM(upcom_met, emissions=emissions)
    source = MetDataset(upcom_met.data.isel(longitude=slice(0, 1)))
    with pytest.raises(ValueError, match="longitude"):
        model.eval(source)
    trajectory = GeoVectorDataset(
        longitude=[0.0],
        latitude=[40.0],
        level=[200.0],
        time=np.array(["2022-03-01T00"], dtype="datetime64[ns]"),
    )
    with pytest.raises(NotImplementedError, match="gridded"):
        model.eval(trajectory)


def test_ISSR_met_source(met_issr: MetDataset) -> None:
    """Test ISSR model evaluated on a MetDataset source."""

    with pytest.warns(UserWarning, match="originated from ECMWF"):
        ISSR(met=met_issr)

    model = ISSR(met=met_issr, humidity_scaling=ExponentialBoostHumidityScaling())
    out1 = model.eval()

    assert isinstance(out1, MetDataset)
    assert isinstance(out1.data, xr.Dataset)
    assert isinstance(model.met, MetDataset)
    assert set(out1.data) == {"air_temperature", "specific_humidity", "rhi", "issr"}
    assert all(v == np.float32 for v in out1.data.dtypes.values())
    assert out1["issr"].name == "issr"

    assert out1.attrs["humidity_scaling_name"] == "exponential_boost"
    assert out1.attrs["humidity_scaling_formula"] == "rhi -> (rhi / rhi_adj) ^ rhi_boost_exponent"
    assert out1.attrs["humidity_scaling_rhi_adj"] == 0.97
    assert out1.attrs["humidity_scaling_rhi_boost_exponent"] == 1.7
    assert out1.attrs["humidity_scaling_clip_upper"] == 1.7

    assert "pycontrails_version" in out1.attrs
    assert out1.attrs["met_source_provider"] == "ECMWF"
    assert out1.attrs["met_source_dataset"] == "ERA5"
    assert out1.attrs["met_source_product"] == "reanalysis"

    # Pin some counts
    vals, counts = np.unique(out1.data["issr"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [563, 157])

    met_copy = met_issr.copy()
    del met_copy.data["air_temperature"]
    out2 = model.eval(source=met_copy)

    # Important: if specific_humidity included in model input, it will NOT
    # undergo humidity scaling. This is why the counts below are lower.
    assert "humidity_scaling" not in out2.attrs
    assert isinstance(out2, MetDataset)
    assert all(v == np.float32 for v in out2.data.dtypes.values())

    # Pin some counts
    vals, counts = np.unique(out2.data["issr"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [587, 133])


def test_SAC_met_source(met_issr: MetDataset) -> None:
    """Test the SAC model evaluated in a MetDataset source."""

    with pytest.warns(UserWarning, match="originated from ECMWF"):
        SAC(met=met_issr)

    model = SAC(met_issr, humidity_scaling=ConstantHumidityScaling())
    out1 = model.eval()

    assert isinstance(out1, MetDataset)
    assert isinstance(out1.data, xr.Dataset)
    assert isinstance(model.met, MetDataset)
    assert all(v == np.float32 for v in out1.data.dtypes.values())
    assert set(out1.data) == {
        "G",
        "T_sat_liquid",
        "rhi",
        "rh_critical_sac",
        "specific_humidity",
        "rh",
        "sac",
        "air_temperature",
    }

    # Pin some counts
    vals, counts = np.unique(out1.data["sac"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [351, 369])

    met_copy = met_issr.copy()
    del met_copy.data["air_temperature"]
    met_copy.data["engine_efficiency"] = 0.35
    out2 = model.eval(source=met_copy)

    assert isinstance(out2, MetDataset)
    assert np.all(out2.data["engine_efficiency"] == 0.35)
    assert all(v == np.float32 for v in out1.data.dtypes.values())

    # Pin some counts
    # Values are now different because humidity was not scaled
    vals, counts = np.unique(out2.data["sac"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [338, 382])


def test_SAC_with_nan(met_issr: MetDataset) -> None:
    """Check that NaN values persist after SAC calculation."""
    met = MetDataset(met_issr.data.copy(deep=True))
    met.data["specific_humidity"][4, 3, 2, 1] = np.nan

    model = SAC(met, humidity_scaling=ExponentialBoostHumidityScaling())
    out = model.eval()["sac"]
    assert out.data.dtype == np.float32
    np.testing.assert_array_equal(np.nonzero(np.isnan(out.values)), [[4], [3], [2], [1]])

    actual = np.unique(out.data)
    expected = np.array([0.0, 1.0, np.nan])
    np.testing.assert_array_equal(actual, expected)


def test_PCR_grid(met_issr: MetDataset) -> None:
    """Test PCRGrid model."""

    with pytest.warns(UserWarning, match="originated from ECMWF"):
        PCR(met=met_issr)

    model = PCR(met_issr, humidity_scaling=ConstantHumidityScaling())
    out1 = model.eval()

    assert isinstance(out1, MetDataset)
    assert isinstance(out1.data, xr.Dataset)
    assert isinstance(model.met, MetDataset)
    assert all(v == np.float32 for v in out1.data.dtypes.values())

    assert out1.attrs.pop("pycontrails_version")
    assert out1.attrs == {
        "humidity_scaling_name": "constant_scale",
        "humidity_scaling_formula": "rhi -> rhi / rhi_adj",
        "humidity_scaling_rhi_adj": 0.97,
        "met_source_provider": "ECMWF",
        "met_source_dataset": "ERA5",
        "met_source_product": "reanalysis",
        "engine_efficiency": 0.3,
        "rhi_adj": 0.97,
    }

    # Pin some counts
    vals, counts = np.unique(out1.data["pcr"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [607, 113])

    met_copy = met_issr.copy()
    del met_copy.data["air_temperature"]
    met_copy.data["engine_efficiency"] = (
        met_copy.data.dims,
        np.full(met_copy.shape, 0.35, dtype=np.float32),
    )
    out2 = model.eval(source=met_copy)

    assert isinstance(out2, MetDataset)
    assert all(v == np.float32 for v in out2.data.dtypes.values())
    assert np.all(out2.data["engine_efficiency"] == 0.35)

    # Pin some counts
    vals, counts = np.unique(out2.data["pcr"], return_counts=True)
    np.testing.assert_array_equal(vals, [0, 1])
    np.testing.assert_array_equal(counts, [622, 98])


def test_era5_as_expected(met_era5_fake: MetDataset) -> None:
    """Test ERA5 data with all models."""

    # created with exactly 20% is ISSR
    assert not originates_from_ecmwf(met_era5_fake)
    issr = ISSR(met=met_era5_fake).eval()["issr"]
    assert isinstance(issr, MetDataArray)
    assert issr.proportion == 0.2
    assert issr.data.dtype == np.result_type(*met_era5_fake.data.dtypes.values())

    # ISSR exist exactly when longitude is a multiple of 5
    # demonstrating this behavior on a few arbitrary selections
    assert np.all(issr.data.sel(longitude=123).values == 0)
    assert np.all(issr.data.sel(longitude=124).values == 0)
    assert np.all(issr.data.sel(longitude=125).values == 1)
    assert np.all(issr.data.sel(longitude=126).values == 0)
    assert np.all(issr.data.sel(longitude=127).values == 0)

    # ISSR and PCR do not completely agree for this data
    # In particular, on levels 150, 175, and 200, the two diverge
    pcr = PCR(met_era5_fake).eval()["pcr"]
    assert isinstance(pcr, MetDataArray)
    assert pcr.data.dtype == np.result_type(*met_era5_fake.data.dtypes.values())
    pcr_low = pcr.data.isel(level=slice(3, None))
    issr_low = issr.data.isel(level=slice(3, None))
    xr.testing.assert_equal(pcr_low, issr_low)

    # But pcr is identically 0 on the top three levels
    pcr_high = pcr.data.isel(level=slice(0, 3))
    assert (pcr_high == 0).all()


def test_ISSR_flight(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR model."""
    out = ISSR(met=met_era5_fake).eval(source=flight_fake)

    assert isinstance(out, Flight)
    assert np.nansum(out["issr"]) == 73.0


def test_ISSR_flight_high_altitude(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR model with high altitudes."""
    # modifying fixture
    fl = flight_fake.copy()
    fl["altitude"] *= 2.5

    with pytest.raises(ValueError, match="One of the requested xi is out of bounds"):
        ISSR(met=met_era5_fake, interpolation_bounds_error=True).eval(source=fl)

    fl = ISSR(met=met_era5_fake).eval(source=fl)

    # waypoints above highest-altitude level in era5 get filled with NaN
    assert isinstance(fl, Flight)
    low_level_waypoints = fl.level < met_era5_fake.data["level"].min().item()
    assert np.isnan(fl["issr"][low_level_waypoints]).all()
    assert ~np.isnan(fl["issr"][~low_level_waypoints]).all()


@pytest.mark.parametrize("interpolation_use_indices", [True, False])
def test_ISSR_flight_outside_met(
    met_era5_fake: MetDataset, flight_fake: Flight, interpolation_use_indices: bool
) -> None:
    """Test ISSR model with flight data outside bounds of met data.

    The flight trajectory leaves the met domain in two places:

    - the initial flight waypoints are at low altitude (below the max pressure level in met)
    - the later flight waypoints have times beyond the max time in met
    """

    # Alter the flight time
    fl = flight_fake.copy()
    with pytest.warns(UserWarning, match="Overwriting data in key `time`."):
        fl["time"] += pd.Timedelta(hours=1)

    model = ISSR(met_era5_fake, interpolation_use_indices=interpolation_use_indices)

    with pytest.raises(ValueError, match="One of the requested xi is out of bounds in dimension 2"):
        model.eval(source=fl, interpolation_bounds_error=True)

    # Evaluate the model
    fl = model.eval(source=fl, interpolation_bounds_error=False)
    assert isinstance(fl, Flight)
    issr = fl["issr"]
    assert issr.dtype == np.float64

    # Waypoints with timestamp beyond the max time are nan-filled
    late_waypoints = fl["time"] > met_era5_fake["time"].values[-1]
    assert np.sum(late_waypoints) == 130
    assert np.isnan(issr[late_waypoints]).all()

    # Waypoints with low altitude are also nan-filled
    low_waypoints = fl.altitude < met_era5_fake["altitude"].values.min()
    assert np.sum(low_waypoints) == 257
    assert np.isnan(fl["issr"][late_waypoints]).all()

    # Anything remaining is non-nan
    remaining = ~late_waypoints & ~low_waypoints
    assert np.sum(remaining) == 113
    assert np.all(np.isfinite(issr[remaining]))


def test_ISSR_flight_span_antimeridian(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR model with flight data spanning anti-meridian."""
    # modifying fixture
    met_era5_fake_wrapped = MetDataset(met_era5_fake.data, wrap_longitude=True)
    fl = flight_fake.copy()
    n_waypoints = len(fl)
    assert n_waypoints % 2 == 0  # needed below
    longitude = np.concatenate(
        [
            np.linspace(150, 180, n_waypoints // 2, endpoint=False),
            np.linspace(-180, -140, n_waypoints // 2),
        ]
    )
    altitude = np.full(shape=fl.shape, fill_value=10000)
    fl.update(longitude=longitude, altitude=altitude)

    fl = ISSR(met_era5_fake_wrapped).eval(source=fl)
    assert isinstance(fl, Flight)
    assert np.all(~np.isnan(fl["issr"]))

    # assert output
    assert fl["issr"].sum() == 158.0


def test_ISSR_flight_fill_value(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR model with nan fill value."""

    # Waypoints below lowest-altitude level in era5 get filled with nan
    low_altitude_waypoints = flight_fake["altitude"] < met_era5_fake.data["altitude"].min().item()

    # null result (default fill value)
    out = ISSR(met_era5_fake).eval(source=flight_fake)
    assert np.all(np.isnan(out["issr"][low_altitude_waypoints]))
    assert np.all(np.isfinite(out["issr"][~low_altitude_waypoints]))

    # When we pass in a custom fill_value here, we no longer encounter nan
    # keep the interpolation fill value reasonable here; else some of our
    # thermo functions will encounter divide by zero issues
    out2 = ISSR(met_era5_fake, interpolation_fill_value=180).eval(source=flight_fake)
    assert np.all(np.isfinite(out2["issr"]))


def test_ISSR_flight_interp(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR model with interpolation methods."""

    # test fake method
    with pytest.raises(ValueError, match="Method 'fake' is not defined"):
        ISSR(met_era5_fake, interpolation_method="fake").eval(source=flight_fake)

    # test bounds error
    with pytest.raises(ValueError, match="xi is out of bounds"):
        ISSR(met_era5_fake, interpolation_bounds_error=True).eval(source=flight_fake)

    # default interpolation method is linear
    out2 = ISSR(met_era5_fake).eval(source=flight_fake)
    out3 = ISSR(met_era5_fake, interpolation_method="linear").eval(source=flight_fake)

    assert isinstance(out2, Flight)
    assert isinstance(out3, Flight)

    # linear fill is default, so these must be equal (outside nan)
    assert np.all(out2["issr"][~np.isnan(out2["issr"])] == out3["issr"][~np.isnan(out3["issr"])])

    # nearest interpolation without extrapolation
    out4 = ISSR(met_era5_fake, interpolation_method="nearest").eval(source=flight_fake)
    # nearest interpolation with extrapolation
    out5 = ISSR(met_era5_fake, interpolation_method="nearest", interpolation_fill_value=None).eval(
        source=flight_fake
    )

    assert isinstance(out4, Flight)
    assert isinstance(out5, Flight)

    # out4 has missing values, out5 has no missing values
    assert np.isnan(out4["issr"]).any()
    assert not np.isnan(out5["issr"]).any()

    # AND, because fl4 is grabbing the closest era5 value
    # it contains positive ISSR waypoints at very low altitudes
    assert out5["issr"][0] == 1

    # expect an issr value exactly when flight waypoint
    # longitude value is within 1/2 of a multiple of 5
    expect_issr = np.abs((flight_fake["longitude"] + 2.5) % 5 - 2.5) < 1 / 2
    actual_issr = out5["issr"].astype(bool)
    assert np.all(expect_issr == actual_issr)


def test_ISSR_flight_incomplete_era5(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR flight with incomplete met data."""

    da = met_era5_fake["air_temperature"].data
    ds = da.to_dataset()
    mds = MetDataset(ds)
    with pytest.raises(KeyError, match="Dataset does not contain variable `specific_humidity`"):
        ISSR(mds)


def test_ISSR_flight_missing_waypoint(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test ISSR flight with missing waypoints met data."""

    # modifying fl fixture to avoid NaN caused by low altitude waypoints
    fl = flight_fake.copy()
    fl.update(altitude=np.full(shape=fl.shape, fill_value=10000.0))

    # and adding some additional NaN
    fl["altitude"][200] = np.nan
    fl["longitude"][210] = np.nan
    fl["latitude"][212] = np.nan
    fl["time"][217] = np.datetime64("NaT")

    out2 = ISSR(met_era5_fake).eval(source=fl)

    # expect issr column to have exactly at waypoints with NaN
    for idx in [200, 210, 212, 217]:
        assert np.isnan(out2["issr"][idx])


def test_SAC_flight(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test SAC flight model."""

    fl = flight_fake.copy()

    prev_sac_sum = float("inf")
    for altitude, sac_sum in [(9000, 160), (10000, 150), (11000, 139), (12000, 128), (13000, 0)]:
        # modifying flight_fake fixture by setting constant altitude
        fl.update(altitude=np.full(shape=fl.shape, fill_value=altitude))
        fl.attrs["engine_efficiency"] = 0.4

        model = SAC(met_era5_fake, humidity_scaling=ExponentialBoostHumidityScaling())
        out = model.eval(source=fl)
        assert isinstance(out, Flight)
        assert np.nansum(out["sac"]) == sac_sum

        # NOTE: as altitudes go up, we get fewer SAC
        # this is caused by the `air_pressure` term in the function `G` in formula.py
        cur_sac_sum = np.nansum(out["sac"])
        assert cur_sac_sum < prev_sac_sum
        prev_sac_sum = cur_sac_sum


def test_SAC_flight_with_nan(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Check that NaN persist in SAC calculation."""

    # Original flight has no NaN
    for key in flight_fake:
        assert not np.isnan(flight_fake[key]).any()

    # So we put some in
    fl = flight_fake.copy()
    fl["longitude"][5] = np.nan

    # But, the flight is out of bounds of the met data
    with pytest.raises(ValueError, match="bounds"):
        SAC(met_era5_fake, interpolation_bounds_error=True).eval(source=fl)

    # So modify the level to keep flight in the met bounds
    fl.update(altitude=np.full(shape=fl.shape, fill_value=units.pl_to_m(275)))

    # And confirm that the NaN at position 5 persists in the new SAC column
    # Note that "nan" is considered out of bounds, so we need to set bounds error to False
    out = SAC(met_era5_fake, interpolation_bounds_error=False).eval(source=fl)
    sac = out["sac"]
    actual = np.isnan(sac)
    expected = np.zeros_like(sac).astype(bool)
    expected[5] = True
    np.testing.assert_array_equal(actual, expected)


def test_PCR_flight(met_era5_fake: MetDataset, flight_fake: Flight) -> None:
    """Test PCR Flight model."""

    model = PCR(met_era5_fake, humidity_scaling=ExponentialBoostHumidityScaling())

    out1 = model.eval(source=flight_fake)
    assert isinstance(out1, Flight)
    assert "engine_efficiency" in out1.attrs  # params transferred from model
    assert np.nansum(out1["pcr"]) == 56

    # override engine efficiency via attributes
    flight_fake.attrs["engine_efficiency"] = 0.4
    out2 = model.eval(source=flight_fake)

    assert isinstance(out2, Flight)
    assert out2.attrs["engine_efficiency"] == 0.4
    assert np.nansum(out2["pcr"]) == 71


@pytest.mark.parametrize("step", [1e-3, 1e-4, 1e-5, 1e-6])
def test_e_sat_liquid_prime(step: float) -> None:
    """Confirm `thermo.e_sat_liquid_prime` formula agrees with secant approximation.

    This test as well as the `thermo.e_sat_liquid_prime` function will need to be
    updated if `thermo.e_sat_liquid` changes.
    """
    T0 = np.linspace(-60.0, -20.0, 200) - constants.absolute_zero
    assert np.all(T0 > 0)

    T1 = T0 + step * np.ones_like(T0)

    e_sat0 = thermo.e_sat_liquid(T0)
    e_sat1 = thermo.e_sat_liquid(T1)

    secant_slope = (e_sat1 - e_sat0) / (T1 - T0)
    tangent_slope = thermo.e_sat_liquid_prime(T0)

    error = np.abs(secant_slope - tangent_slope)
    assert np.all(error < 2 * step)  # approximation is << step


def test_T_critical_sac_implementation() -> None:
    """Show agreement between current scipy-based approach with depreciated version."""
    rng = np.random.default_rng(987654)
    size = 10000
    p = rng.uniform(20000, 40000, size)
    q = rng.uniform(0, 0.0001, size)
    t = rng.uniform(-70, -10, size) - constants.absolute_zero
    eta = rng.uniform(0.2, 0.5, size)
    rh = thermo.rh(q, t, p)

    jetA = JetA()
    ei_h2o = jetA.ei_h2o
    q_fuel = jetA.q_fuel

    G = sac.slope_mixing_line(q, p, eta, ei_h2o, q_fuel)
    t_lm = sac.T_sat_liquid(G)
    t_crit1 = sac.T_critical_sac(t_lm, rh, G)

    # We get a bunch of warnings here
    # We acknowledge them and continue
    with pytest.warns(RuntimeWarning):
        t_crit2 = _deprecated.T_critical_sac(t, q, p, eta)  # type: ignore[arg-type]

    assert np.all(np.isfinite(t_crit1))
    assert np.all(np.isfinite(t_crit2))

    # Close agreement
    np.testing.assert_allclose(t_crit1, t_crit2, atol=0.15)


def test_T_sat_liquid() -> None:
    """Confirm close agreement between `T_sat_liquid` and `T_sat_liquid_high_accuracy`."""
    rng = np.random.default_rng(737)
    size = 10000
    G = rng.uniform(low=1, high=4, size=size)
    t1 = sac.T_sat_liquid(G)
    t2 = sac.T_sat_liquid_high_accuracy(G)

    # Close agreement
    np.testing.assert_allclose(t1, t2, atol=0.05)


def test_T_critical_sac_schumann() -> None:
    """Confirm Schumann's statements from his 1996 paper."""

    with (
        patch("pycontrails.physics.thermo.e_sat_liquid", thermo.sonntag_e_sat_liquid),
        patch("pycontrails.physics.thermo.e_sat_liquid_prime", thermo.sonntag_e_sat_liquid_prime),
    ):
        rng = np.random.default_rng(737)
        size = 10000
        G = rng.uniform(low=1, high=4, size=size)

        # Intentionally going above 1 for rh here; these supersaturated values
        # get clipped in sac.T_critical_sac
        rh = rng.uniform(low=0, high=1.01, size=size)
        t_LM = sac.T_sat_liquid(G)
        t_LC = sac.T_critical_sac(t_LM, rh, G)

        # Critical temperature is always below tangent line temperature value
        assert np.all(t_LC <= t_LM)
        # Equality when U >= 1
        idx = np.nonzero(rh >= 1)
        np.testing.assert_array_equal(t_LC[idx], t_LM[idx])

        # For U = 0, equation (11) gives explicit answer
        t_LC_U0 = sac.T_critical_sac(t_LM, np.zeros_like(rh), G)

        eqn11 = t_LM - thermo.sonntag_e_sat_liquid(t_LM) / G
        np.testing.assert_array_equal(t_LC_U0, eqn11)


@pytest.mark.parametrize("in_bounds", [False, True])
def test_issr_source_grid_different_resolution(met_issr: MetDataset, in_bounds: bool):
    """Confirm the ISSR model can handle gridded source with different resolution from met."""
    model = ISSR(met=met_issr, humidity_scaling=ConstantHumidityScaling(rhi_adj=0.95))

    # Create source
    level = [240, 250, 271]
    if not in_bounds:
        level[0] = 221

    source = MetDataset.from_coords(
        longitude=[43, 44, 45],
        latitude=[-80, -55, 13],
        level=level,
        time=np.datetime64("2019-05-31 05:00:00"),
    )

    if in_bounds:
        out = model.eval(source, interpolation_bounds_error=True)
        assert np.all(np.isfinite(out.data))
        return

    with pytest.raises(ValueError, match="221"):
        model.eval(source, interpolation_bounds_error=True)


@pytest.mark.filterwarnings(r"ignore:\sMet data appears to have originated from ECMWF")
@pytest.mark.parametrize("q_method", [None, "cubic-spline"])
def test_ISSR_grid_variable_q_method(met_issr: MetDataset, q_method: None | str) -> None:
    """Test ISSR model with variable q_method."""

    source = MetDataset.from_coords(
        longitude=met_issr.coords["longitude"],
        latitude=met_issr.coords["latitude"],
        level=[240, 250, 260],  # different from met_issr
        time=met_issr.coords["time"],
    )

    model = ISSR(met_issr, interpolation_q_method=q_method)
    out = model.eval(source)["issr"]

    assert isinstance(out, MetDataArray)
    if q_method is None:
        assert out.data.mean().item() == pytest.approx(0.16805, abs=1e-5)
    else:
        assert q_method == "cubic-spline"
        assert out.data.mean().item() == pytest.approx(0.15555, abs=1e-5)
