import jax.numpy as jnp
from astropy.constants import pc  # type: ignore TODO: fix astropy stubs
import astropy.units as u  # type: ignore
G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C_SI = 299792458.0
""" Speed of light, m/s """
C = C_SI
MSUN = 1.988409870698050731911960804878414216e30
""" Nominal solar mass, kg """

MTSUN = 4.925490947641266978197229498498379006e-6
""" Geometrised Nominal solar mass, s """

MRSUN = 1.476625061404649406193430731479084713e3
""" Geometrised Nominal solar mass, m """

year = (1 * u.yr).cgs.value  # type: ignore
Mpc = 1e6 * pc.value  # m
euler_gamma = 0.577215664901532860606512090082
EulerGamma = euler_gamma
EARTH_SEMI_MAJOR_AXIS = 6378137.0  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

DAYSID_SI = 86164.09053133354
DAYJUL_SI: int = 86400

DEG_TO_RAD = jnp.pi / 180

HR_TO_RAD = 2 * jnp.pi / 24
HR_TO_SEC: int = 3600
SEC_TO_RAD = HR_TO_RAD / HR_TO_SEC

"""Pi"""
PI = 3.141592653589793238462643383279502884

TWO_PI = 6.283185307179586476925286766559005768

gt = G * MSUN / (C**3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

clightGpc = C / 3.0856778570831e22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second
"""
