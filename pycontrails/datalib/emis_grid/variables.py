"""Gridded aircraft emissions inventory variable definitions.

Variables follow the CF-like conventions used across pycontrails, defined
as :class:`MetVariable` frozen dataclasses.
"""

from __future__ import annotations

from pycontrails.core.met import MetVariable

# ---------------------------------------------------------------------------
# AEIC variables
# ---------------------------------------------------------------------------

AEICFuelBurn = MetVariable(
    short_name="FUELBURN",
    standard_name="aircraft_fuel_burn",
    long_name="aircraft fuelburn in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description=(
        "Gridded aircraft fuel burn rate per unit area at each pressure level. "
        "Non-LTO (Landing and Take-Off) operations only."
    ),
)

AEICBC = MetVariable(
    short_name="BC",
    standard_name="aircraft_black_carbon_emission",
    long_name="aircraft black carbon emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description=(
        "Gridded aircraft black carbon (soot) emission rate per unit area "
        "at each pressure level."
    ),
)

AEICDistance = MetVariable(
    short_name="DISTANCE",
    standard_name="aircraft_distance_flown",
    long_name="Non-LTO total flown distance in km/m2/s",
    units="km m**-2 s**-1",
    level_type="isobaricInhPa",
    description=(
        "Gridded aircraft distance flown per unit area at each pressure level. "
        "Non-LTO (Landing and Take-Off) operations only."
    ),
)

AEICCO = MetVariable(
    short_name="CO",
    standard_name="aircraft_co_emission",
    long_name="aircraft CO emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description="Gridded aircraft carbon monoxide emission rate per unit area at each pressure level.",
)

AEICHC = MetVariable(
    short_name="HC",
    standard_name="aircraft_hydrocarbon_emission",
    long_name="aircraft hydrocarbon emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description="Gridded aircraft hydrocarbon emission rate per unit area at each pressure level.",
)

AEICHONO = MetVariable(
    short_name="HONO",
    standard_name="aircraft_hono_emission",
    long_name="aircraft HONO emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description=(
        "Gridded aircraft nitrous acid (HONO) emission rate per unit area "
        "at each pressure level."
    ),
)

AEICNO = MetVariable(
    short_name="NO",
    standard_name="aircraft_no_emission",
    long_name="aircraft NO emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description="Gridded aircraft nitric oxide emission rate per unit area at each pressure level.",
)

AEICNO2 = MetVariable(
    short_name="NO2",
    standard_name="aircraft_no2_emission",
    long_name="aircraft NO2 emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description="Gridded aircraft nitrogen dioxide emission rate per unit area at each pressure level.",
)

AEICOC = MetVariable(
    short_name="OC",
    standard_name="aircraft_organic_carbon_emission",
    long_name="aircraft organic carbon emissions in kg/m2/s",
    units="kg m**-2 s**-1",
    level_type="isobaricInhPa",
    description=(
        "Gridded aircraft organic carbon emission rate per unit area at each pressure level."
    ),
)

#: All AEIC variables available in monthly netCDF files
AEIC_VARIABLES: list[MetVariable] = [
    AEICFuelBurn,
    AEICBC,
    AEICDistance,
    AEICCO,
    AEICHC,
    AEICHONO,
    AEICNO,
    AEICNO2,
    AEICOC,
]

#: Default AEIC variables requested when none are specified
AEIC_DEFAULT_VARIABLES: list[MetVariable] = [
    AEICFuelBurn,
    AEICBC,
    AEICDistance,
]