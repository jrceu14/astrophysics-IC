import scipy.integrate as integrate
import numpy as np

# Metric disclaimer: by convention, all distances are measured in kpc and
# all densities in GeV/cm^3. Detours from the convention must be explicitly
# noted. 

class MassProfile:
    """Generic class to specify dark matter profiles.
    
    :param r_s: scale radius.
    :param rho_s: scale density.
    """
    def __init__(self, r_s, rho_s):
        self.r_s = r_s
        self.rho_s = rho_s
        self._r_sun = 8.33 # kpc
        self._rho_sun = 0.3 # GeV/cm^3
    
    def j_factor(self, theta, factor_type='annihilation', **kwargs):
        """J-factor integral on the observer l.o.s.
        
        :param theta: angle between GC and the observer line of sight (los).
        :param factor_type: j-factor type: annihilation or decay.
        :param *args: scipy integrate.quad arugments
        :return: integrated annihilation J-factor on los.
        """
        if factor_type == "annihilation":
            expr = lambda s: (1/self._r_sun)*(self.density(s, theta, geocentric=True)/self._rho_sun)**2
        elif factor_type == "decay":
            expr = lambda s: (1/self._r_sun)*(self.density(s, theta, geocentric=True)/self._rho_sun)
        j_factor, err = integrate.quad(expr, 0, np.inf, **kwargs)
        return j_factor, err
    
    def j_factor_map(self, L_coord, B_coord, delta_l, delta_b, factor_type='annihilation', **kwargs):
        """Bidimensional map of J-factors. 
        
        :param L_coord: grid of galactic polar longitudes.
        :param B_coord: grid of galactic polar latitudes.
        :param delta_l: spacing of galactic polar longitudes.
        :param delta_b: spacing of galactic polar longitudes.
        :param factor_type: j-factor type: annihilation or decay.
        :return: integrated j_factor at each grid position.
        """
        factors = np.zeros_like(L_coord)
        for i in range(L_coord.shape[0]):
            for j in range(B_coord.shape[0]):
                l, b = L_coord[i, j], B_coord[i, j]
                theta = np.arccos(np.cos(l)*np.cos(b))
                aperture = delta_l*(np.sin(B_coord[i, j]+delta_b/2) - np.sin(B_coord[i, j]-delta_b/2))
                factors[i, j] = aperture*self.j_factor(theta, factor_type, **kwargs)[0] 
        return factors
        
    def apertures_map(self, L_coord, B_coord, delta_l, delta_b):
        """Bidimensional map of apertures. All angles should be
        in rad.
        
        :param L_coord: grid of galactic polar longitudes.
        :param B_coord: grid of galactic polar latitudes.
        :param delta_l: spacing of galactic polar longitudes.
        :param delta_b: spacing of galactic polar longitudes.
        :return: apertures at each grid cell.
        """
        apertures = np.zeros_like(L_coord)
        for i in range(L_coord.shape[0]):
            for j in range(B_coord.shape[0]):
                aperture = delta_l*(np.sin(B_coord[i, j]+delta_b/2) - np.sin(B_coord[i, j]-delta_b/2))
                apertures[i, j] = aperture
        return apertures
    
    def _geocentric_to_galactocentric(self, s, theta):
        """Convert between geocentric and galactocentric distances. 

        :param r: galactocentric distance.
        :param theta: geocentric angle positive from the GC.
        """
        return np.sqrt(self._r_sun**2 + s**2 - 2*s*self._r_sun*np.cos(theta))

    
class MassProfileNFW(MassProfile):
    """Navarro-Frenk-White dark matter density profile."""
    
    def __init__(self, r_s=24.42, rho_s=0.184):
        super().__init__(r_s, rho_s)
        
    def density(self, r, theta=None, geocentric=False):
        if geocentric:
            r = self._geocentric_to_galactocentric(r, theta)
        return self.rho_s/((r/self.r_s)*(1+r/self.r_s)**2)

    
class MassProfileEinasto(MassProfile):
    """Einasto dark matter density profile """
    def __init__(self, r_s=28.44, rho_s=0.033, alpha=0.17):
        super().__init__(r_s, rho_s)
        self.alpha = alpha
        
    def density(self, r, theta=None, geocentric=False):
        if geocentric:
            r = self._geocentric_to_galactocentric(r, theta)
        return self.rho_s * np.exp(-2/self.alpha*((r/self.r_s)**self.alpha-1))      

    
class MassProfileIsothermal(MassProfile):
    """Isothermal dark matter denstiy profile"""
    def __init__(self, r_s=4.38, rho_s=1.387):
        super().__init__(r_s, rho_s)
    
    def density(self, r, theta=None, geocentric=False):
        if geocentric:
            r = self._geocentric_to_galactocentric(r, theta)
        return self.rho_s/(1+(r/self.r_s)**2)
    

class MassProfileBurkert(MassProfile):
    """Burkert dark matter denstiy profile."""
    def __init__(self, r_s=12.67, rho_s=0.712):
        super().__init__(r_s, rho_s)
        
    def density(self, r, theta=None, geocentric=False):
        if geocentric:
            r = self._geocentric_to_galactocentric(r, theta)
        return self.rho_s/((1+r/self.r_s)*(1+(r/self.r_s)**2))
    

class MassProfileMoore(MassProfile):
    """Moore dark matter denstiy profile."""
    def __init__(self, r_s=30.28, rho_s=0.105):
        super().__init__(r_s, rho_s)
    
    def density(self, r, theta=None, geocentric=False):
        if geocentric:
            r = self._geocentric_to_galactocentric(r, theta)
        return self.rho_s*(self.r_s/r)**(1.16) * (1 + r/self.r_s)**(-1.84)
 
