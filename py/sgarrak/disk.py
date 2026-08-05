# disk module

import numpy as np


### EMERGE
## EMERGE is an empirical model for star formation. see Moster et al. 2018
#  For the host disk, it uses the instanious baryon conversion efficiency.
#  For the satellite stellar mass, it uses the integrated baryonic conversion
#  efficiency (SMHM relation).
## NOTE: EMERGE only use 'forward' method.
#        (Although the logic of tdyn is current time - tdyn.)
#        EMERGE uses h=0.6781
def mstar(delta_t,z,
          M_dot_dyn,Rv_dot_dyn,e,dens_profile,
          h=0.7):
    """
    The main part of the equations. It calculates the stellar mass growth
    between each time step.

    Syntax: stellar mass growth = (in-situ formation - dying stars)

    f_loss is from Chabrier 2003 IMF····

    Parameters: delta_t: t-t'
                M_dot_dyn: halo growth rate over a dynamical time
                Rv_dot_dyn: virial radius growth rate over a dynamical time
                e: instantanious baryon conversion efficiency
                dens_profile: halo density profile

    Return: Stellar mass increased in delta_t

    NOTE:·
    m_loss_dot does not include the instantaneous recycling.

    macc is not needed for our purpose.

    cww:
    The model only fit to z=10?
    """

    mstar_dot = sfr(M_dot_dyn,Rv_dot_dyn,e,dens_profile,z)
    f_loss = 0.05 * np.log(1+delta_t/(1.4/1e3)) # eq.12 in Gyr
    delta_mstar = delta_t * (mstar_dot * (1 - f_loss)) # eq.11
    return delta_mstar

def sfr(M_dot_dyn,Rv_dot_dyn,e,dens_profile,z):
    """
    The halo mass growth due to accretions is the accreted masses - pseudo evolution of
    the virial radius growth over time as the background density decrease.

    Syntax: M_dot = <M_dot>_dyn - 4 * pi * Rv^2 * rho(Rv) * <Rv_dot>_dyn

    The baryonic growth rate at two time steps is the baryons brought from mergers

    Syntax: mb_dot = fb * M_acc

    The stellar mass forms between two time steps is the mb_dot times the instantaneous·
    baryonic conversion efficiency.

    Syntax: mstar_dot = mb_dot * e

    Parameter: M_dot_dyn: halo growth rate over a dynamical time
               Rv_dot_dyn: virial radius growth rate over a dynamical time
               e: instantanious baryon conversion efficiency
               dens_profile: halo density profile
               z: redshift

    NOTE: The cosmology in our model does not set Ob0. Maybe this needs to be changed.

    cww:
    This method acctually calculate the baryonic budget from mergers, which solves the·
    ambiguity of gas brought by accreted subhalos when using a purely SMHM relation.

    """

    # Cosmology we use
    Omega_b = 0.
    Omega_m = 0.25
    # Omega_b/Omega_m from Moster18
    fb = 0.156·
    #Omega_b = 0.0484
    #Omega_m = 0.308

    # virial radius
    Rv = dens_profile.rh
    # density at the virial radius.
    rho_Rv = dens_profile.rho(Rv,z)

    # halo growth rate due to accretion and pseudo virial radius growth
    M_dot = M_dot_dyn - 4 * np.pi * Rv**2 * rho_Rv * Rv_dot_dyn # eq.4

    # baryonic growth rate
    mb_dot = fb * M_dot # eq.2

    mstar_dot = mb_dot * e # eq.1
    return mstar_dot

def instant_baryon_conversion_eff(M,z,h=0.7):
    """
    Instantaneous baryon conversion efficiency e

    Table 6 is the fitting result from MCMC.
    """

    # Table 6; best fit
    M0 = 11.339
    Mz = 0.692
    e0 = 0.005
    ez = 0.689
    beta0   = 3.344
    betaz   = -2.079
    gamma0  = 0.966
    h_model = 0.6781

    # h correction
    M0 += np.log10(h_model/h)
    Mz += np.log10(h_model/h)

    a = 1/(z+1)
    lgM1 = M0 + Mz*(1-a)          # eq.7
    en = e0 + ez * (1-a)          # eq.8
    beta  = beta0 + betaz * (1-a) # eq.9
    gamma = gamma0                # eq.10
    M1 = 10**lgM1
    return 2 * en / ((M/M1)**-beta + (M/M1)**gamma) # eq.5

def calculate_tdyn(age,mass,conc,zred,h=0.7,return_lgmass=False):
    """

    Syntax:
    tdyn = (Rv^3/GM)^0.5

    Parameters: age: age of main halos in Gyr
                mass: linear mass of main halos
                conc: concentration from NFW
                zred: redshift
                return_lgmass: This gives mass and Rv in log10 for
                               analysis.

    cww:
    Maybe h affects the mass and the virial radius here?

    """

    #G = const.G.to(u.kpc**3/u.M_sun/u.Gyr**2).value

    halo_profile = NFW(mass,conc,Delta=200.,z=zred,sf=1.)
    Rv = halo_profile.rh
    tdyn_pf = halo_profile.tdyn(Rv)

    # Interpolate in log mass
    lgmass = np.log10(mass)
    fm = interp1d(age,lgmass,fill_value='extrapolate')
    fr = interp1d(age,Rv,fill_value='extrapolate')

    m_dot_dyn  = []
    Rv_dot_dyn = []
    lgm_dot_dyn = []
    tdyn_cal = []
    for rv,m,ag in zip(Rv,mass,age):
        tdyn = (rv**3/cfg.G/m)**0.5 # Gyr
        tdyn_cal.append(tdyn)
        if ag<tdyn:
            raise ValueError(f"At the age '{ag}', the tdyn is larger")
        m_dot_dyn.append((m-10**fm(ag-tdyn))/tdyn) #eq.3
        Rv_dot_dyn.append((rv-fr(ag-tdyn))/tdyn)
        lgm_dot_dyn.append(np.log10(m-10**fm(ag-tdyn))/tdyn)

    if return_lgmass:
        return lgm_dot_dyn,Rv_dot_dyn,tdyn_pf,tdyn_cal,Rv

    return m_dot_dyn,Rv_dot_dyn,tdyn_pf,tdyn_cal,Rv


def disk(choice):
    """
    """

    if choice==fd:
    return

