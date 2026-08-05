# stellar mass halo mass relation module
# Models are h dependent.

import numpy as np

# This is for m18
def integrated_baryon_conversion_eff(zred,mass,h=0.7):
    """
    Integrated baryon conversion efficiency in Moster18, for computing the
    EMERGE SMHM relation.

    Parameters
    ----------
    zred: redshift. scalar
    mass: halo mass. array or scalar

    Returns
    -------
    integrated baryonic efficiency, scatter
    """

    # coefficients, table 8 all centrals
    coef = dict()
    z               = np.array([0.1,0.5,1.0,2.0,4.0,8.0])
    coef['M1']      = np.array([11.80,11.85,11.95,12.,12.05,12.10])
    coef['en']      = np.array([0.14,0.16,0.18,0.18,0.19,0.24])
    coef['beta']    = np.array([1.75,1.70,1.60,1.55,1.50,1.30])
    coef['gamma']   = np.array([0.57,0.58,0.60,0.62,0.64,0.64])
    coef['M_sigma'] = np.array([10.80,10.70,10.60,10.50,10.40,10.30])
    coef['sigma0']  = np.array([0.16,0.14,0.12,0.10,0.08,0.02])
    coef['alpha']   = np.array([1.0,0.90,0.75,0.50,0.40,0.10])

    m18_h = 0.6781

    # h correction
    coef['M1'] += np.log10(m18_h/h)
    coef['M_sigma'] += np.log10(m18_h/h)

    # interpolate, masses are in log.
    for k,d in coef.items():
        f = interp1d(z,d,fill_value='extrapolate')
        coef[k] = f(zred)

    # uncertainty, eq.25
    sigma = coef['sigma0'] + np.log10((mass/10**coef['M_sigma'])**(-coef['alpha']) + 1)

    # eq.5
    e = 2 * coef['en'] / ((mass/10**coef['M1'])**(-coef['beta']) + (mass/10**coef['M1'])**coef['gamma'])
    return e,sigma


def mod_Mstar(hm,h=0.7,z=0.,model='m18',task='Mstar',**kwargs):
    """
    Customized Mstar·
    The halo mass from our eps tree already dealt with h
····
    B13 assumed parameters: Omega_M = 0.27
                            Omega_lambda = 0.73
                            h = 0.7
                            ns = 0.95
                            sigma8 = 0.82

    B19 assumed parameters: h = 0.678

    m18 adjusts h in the parameters. delta_t is now total age.
    It only gives Mstar for the smhm relation.
    m18 assume a baryonic fraction value, but satgen uses cfg.Ob/cfg.Om.

    Parameters: h: Hubble parameter assumed in the merger tree
                hm: linear halo mass

    Return: linear stellar mass

    """

    if model == 'B13':
        h_model = 0.7
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_B13':
            sm = 10**gh.lgMs_B13(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'B13'")

    elif model == 'RP17':
        h_model = 0.7 # Not used, needs to check
        hm *= (h/h_model)
        if task=='Mstar':
            sm = init.Mstar(hm,z,choice=choice)
        elif task=='lgMs_RP17':
            sm = 10**gh.lgMs_RP17(np.log10(hm),z)
        else:
            raise ValueError(f"Invalid task '{task}' for choice 'RP17'")

    elif model == 'B19':
        h_model = 0.678
        hm *= (h/h_model)
        raise NotImplementedError('B19: Not provided yet')

    elif model == 'm18':
        h_model = 0.678
        fb = 0.156 # cfg.Ob/cfg.Om
        e,sigma = integrated_baryon_conversion_eff(z,hm,h=h_model)
        # Should it have a lower limit of hm*fb?
        sm = np.minimum(fb*hm,10**(np.random.normal(np.log10(fb*hm*e),sigma)))

    sm *= (h_model/h)
    return sm
