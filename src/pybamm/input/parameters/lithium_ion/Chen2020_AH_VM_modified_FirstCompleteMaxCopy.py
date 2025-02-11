import pybamm
import numpy as np

def graphite_PAT_ocp_Hebenbrock2025(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    # u_eq = (
    #     57194.9119279 * np.exp(-591.59463163221 * sto)
    #     + 0.8837919007067767
    #     - 0.7181358535* np.tanh(35.60941598 * (sto - 0.017213))
    #     - 0.01926420 * np.tanh(21.10531107 * (sto - 0.516856234))
    #     - 0.057250544 * np.tanh(11.2003946103 * (sto - 0.16110909577420))
    # )

    #Result from mapping of HC features on FC data and to determine sto range
    #x-range: 0.0320 to 0.885 (extended to get right capacity)
    u_eq = (
        -0.002978020577506 * np.exp(-1.845583194414669 * sto)
        + 3.493148782732111e+02
        - 0.043780374257110 * np.tanh(23.752790174953677 * (sto - 0.164028413613762))
        - 3.491648127415070e+02 * np.tanh(38.318905955433150 * (sto + 0.070186813756754))
        - 0.020858185322019 * np.tanh(35.358439863485390 * (sto - 0.468183823131845))
    )

    return u_eq

def NMC811_DiffKoeffi_BA_RasmusBewer_Li(sto,T):
    """
    8th degree polnomial fit for GITT measurement with sucessful parameter determination in SoC range betweenn 10 and 90 %
    This corresponds to a stochiometric range of the material from [0.142 0.684]

    Retrieved from charge direction of GITT technique.

    Parameters for 3 electrode cell setup with aGr/Li/NMC811

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        diffusion coefficient
    """

    D_Li = (
        9.698e-11 * sto**8
        - 3.753e-10 * sto**7
        + 6.038e-10 * sto**6
        - 5.255e-10 * sto**5
        + 2.691e-10 * sto**4
        - 8.249e-11 * sto**3
        + 1.468e-11 * sto**2
        - 1.383e-12 * sto
        + 5.541e-14

    )

    return D_Li

def NMC811_DiffKoeffi_BA_RasmusBewer_Au1(sto,T):
    """
    8th degree polnomial fit for GITT measurement with sucessful parameter determination in SoC range betweenn 10 and 90 %

    Retrieved from charge direction of GITT technique.

    Parameters for 3 electrode cell setup (#1) with aGr/Au(Li)/NMC811

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        diffusion coefficient
    """

    D_Li = (
        2.755e-10 * sto**8
        - 9.416e-10 * sto**7
        + 1.358e-09 * sto**6
        - 1.074e-09 * sto**5
        + 5.068e-10 * sto**4
        - 1.45e-10 * sto**3
        + 2.44e-11 * sto**2
        - 2.203e-12 * sto
        + 8.438e-14

    )

    return D_Li

def NMC811_DiffKoeffi_BA_RasmusBewer_Au2(sto,T):
    """
    8th degree polnomial fit for GITT measurement with sucessful parameter determination in SoC range betweenn 10 and 90 %

    Retrieved from charge direction of GITT technique.

    Parameters for 3 electrode cell setup (#2) with aGr/Au(Li)/NMC811

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    D_Li = (
        6.819e-12 * sto**8
        - 6.747e-11 * sto**7
        + 1.48e-10 * sto**6
        - 1.457e-10 * sto**5
        + 7.616e-11 * sto**4
        - 2.203e-11 * sto**3
        + 3.43e-12 * sto**2
        - 2.632e-13 * sto
        + 1.037e-14

    )

    return D_Li


def graphite_exchange_current_density_CircularLIB(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    k=1.44832e-8 # m/s # from Ecker.2015 * c_e_ref**0.5
    m_ref = k*pybamm.constants.F/1000**0.5 #Chen.2020 Eq. (22) solved for k and then in (19)
    # m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 53400
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def nmc_PAT_ocp_Hebenbrock2025(sto):
    """
    LG M50 NMC open circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    # accurate polynomical fit with extrapolation points to limit errors due small extrapolation
    # x-range: 0.128 to 0.741 - custom cell material
    # u_eq = (
    #     - 1.043048142343884e+04 * sto**9
    #     + 4.962669673525584e+04 * sto**8
    #     - 1.018714014435327e+05 * sto**7
    #     + 1.181473040728662e+05 * sto**6
    #     - 8.512801221395213e+04 * sto**5
    #     + 3.943050938308326e+04 * sto**4
    #     - 1.171024602277544e+04 * sto**3
    #     + 2.142589509608000e+03 * sto**2
    #     - 2.193442437976997e+02 * sto
    #     + 13.871365583439971
    # )

    # CircularLIB Material
    u_eq = (
        - 34.168955877338130 * sto**9
        + 84.919281506762490 * sto**8
        - 13.594007660229918 * sto**7
        - 94.073800375950340 * sto**6
        + 7.770683741187670 * sto**5
        + 1.355413905270537e+02 * sto**4
        - 1.264702926167790e+02 * sto**3
        + 48.222018540864205 * sto**2
        - 9.608924913632150 * sto
        + 5.012572962786598
    )


    return u_eq


def nmc811_exchange_current_density_CircularLIB(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    k=1.12e-9
    m_ref = k*pybamm.constants.F/1000**0.5 #same value as Chen.2020
    # m_ref = 3.42e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations #Chen2020
    E_r = 17800 # J/mol
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def electrolyte_diffusivity_Landesfeind2019(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] J. Landesfeind, H. A. Gasteiger, "Temperature and Concentration Dependence 
    of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes,"
    J. Electrochem. Soc., vol. 166, no. 14, pp. A3079-A3097, 2019.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    p1=1.01e3
    p2=1.01e0
    p3=-1.56e3
    p4=-4.87e2

    c_e_inp=c_e/1000
    D_c_e = p1 * np.exp(p2*c_e_inp) * np.exp(p3/T) * np.exp(p4/T*c_e_inp) * 10e-10 #m/s**2

    return D_c_e


def electrolyte_conductivity_Landesfeind2019(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] J. Landesfeind, H. A. Gasteiger, "Temperature and Concentration Dependence 
    of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes,"
    J. Electrochem. Soc., vol. 166, no. 14, pp. A3079-A3097, 2019.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1=5.21e-1
    p2=2.28e2
    p3=-1.06e0
    p4=3.53e-1
    p5=-3.59e-3
    p6=1.48e-3

    c_e_inp=c_e/1000 #conversion from "mol/m**3" in "mol/L"
    sigma_e = (
        p1*(1+(T-p2))*c_e_inp*((1+p3*c_e_inp**0.5 + p4*(1+p5*np.exp(1000/T))*c_e_inp)/(1+c_e_inp**4*(p6*np.exp(1000/T))))/10
    ) #S/m

    return sigma_e
def electrolyte_thermodynamic_factor_Landesfeind2019(c_e, T):
    """
    Thermodynamic factor of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] J. Landesfeind, H. A. Gasteiger, "Temperature and Concentration Dependence 
    of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes,"
    J. Electrochem. Soc., vol. 166, no. 14, pp. A3079-A3097, 2019.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1=2.57e1
    p2=-4.51e1
    p3=-1.77e-1
    p4=1.94e0
    p5=2.95e-1
    p6=3.08e-4
    p7=2.59e-1
    p8=-9.46e-3
    p9=-4.54e-4

    c_e_inp=c_e/1000
    TDF = (
        p1+p2*c_e_inp+p3*T+p4*c_e_inp**2+p5*c_e_inp*T+p6*T**2+p7*c_e_inp**3+p8*c_e_inp**2*T+p9*c_e_inp*T**2
    )

    return TDF


def electrolyte_tansference_number_Landesfeind2019(c_e, T):
    """
    Transference number of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] J. Landesfeind, H. A. Gasteiger, "Temperature and Concentration Dependence 
    of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes,"
    J. Electrochem. Soc., vol. 166, no. 14, pp. A3079-A3097, 2019.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol` # input: mol/m**3 - need: mol/L
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`   # input: K - need: K
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    p1=-1.28e1
    p2=-6.12e0
    p3=8.21e-2
    p4=9.04e-1
    p5=3.18e-2
    p6=-1.27e-4
    p7=1.75e-2
    p8=-3.12e-3
    p9=-3.96e-5

    c_e_inp=c_e/1000
    tranfer_number = (
        p1+p2*c_e_inp+p3*T+p4*c_e_inp**2+p5*c_e_inp*T+p6*T**2+p7*c_e_inp**3+p8*c_e_inp**2*T+p9*c_e_inp*T**2
    )

    return tranfer_number


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an LG M50 cell, from the paper

        Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
        Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques for
        Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
        Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.

    and references therein.

    SEI parameters are example parameters for SEI growth from the papers:

        Ramadass, P., Haran, B., Gomadam, P. M., White, R., & Popov, B. N. (2004).
        Development of first principles capacity fade model for Li-ion cells. Journal of
        the Electrochemical Society, 151(2), A196-A203.

        Ploehn, H. J., Ramadass, P., & White, R. E. (2004). Solvent diffusion model for
        aging of lithium-ion battery cells. Journal of The Electrochemical Society,
        151(3), A456-A462.

        Single, F., Latz, A., & Horstmann, B. (2018). Identifying the mechanism of
        continued growth of the solid-electrolyte interphase. ChemSusChem, 11(12),
        1950-1955.

        Safari, M., Morcrette, M., Teyssot, A., & Delacour, C. (2009). Multimodal
        Physics- Based Aging Model for Life Prediction of Li-Ion Batteries. Journal of
        The Electrochemical Society, 156(3),

        Yang, X., Leng, Y., Zhang, G., Ge, S., Wang, C. (2017). Modeling of lithium
        plating induced aging of lithium-ion batteries: Transition from linear to
        nonlinear aging. Journal of Power Sources, 360, 28-40.

    Note: this parameter set does not claim to be representative of the true parameter
    values. Instead these are parameter values that were used to fit SEI models to
    observed experimental data in the referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # cell
        "Negative current collector thickness [m]": 1.0e-05,
        "Negative electrode thickness [m]": 8.93e-05,
        "Separator thickness [m]": 2.5e-05,
        "Positive electrode thickness [m]": 6.56e-05,
        "Positive current collector thickness [m]": 1.5e-05, #no value yet
        "Electrode height [m]": 0.065, #limits by positive electrode
        "Electrode width [m]": 0.045*29, #limits by positive electrode
        # "Cell cooling surface area [m2]": 0.00531,
        # "Cell volume [m3]": 2.42e-05,
        # "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        # "Negative current collector conductivity [S.m-1]": 58411000.0,
        # "Positive current collector conductivity [S.m-1]": 36914000.0,
        # "Negative current collector density [kg.m-3]": 8960.0,
        # "Positive current collector density [kg.m-3]": 2700.0,
        # "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        # "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        # "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        # "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 3.0,
        "Current function [A]": 3.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 29871,
        "Negative electrode diffusivity [m2.s-1]": 2.29e-15,
        "Negative electrode OCP [V]": graphite_PAT_ocp_Hebenbrock2025,
        "Negative electrode porosity": 0.364,
        "Negative electrode active material volume fraction": 0.636*0.94, #mutliplied by active material fraction # seems to be the way it is handeled
        "Negative particle radius [m]": 1.6e-05,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        # "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_CircularLIB,
        # "Negative electrode density [kg.m-3]": 1657.0,
        # "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        # "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 67116.0,
        "Positive electrode diffusivity [m2.s-1]": 4e-15, #NMC811_DiffKoeffi_BA_RasmusBewer_Au1, # NMC811_DiffKoeffi_BA_RasmusBewer_Au1, # Chen2020 constant at 4e-15 NMC811_DiffKoeffi_BA_RasmusBewer_Li
        "Positive electrode OCP [V]": nmc_PAT_ocp_Hebenbrock2025,
        "Positive electrode porosity": 0.4784,
        "Positive electrode active material volume fraction": 0.5216*0.94, #mutliplied by active material fraction # seems to be the way it is handeled
        "Positive particle radius [m]": 1.1e-05,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        # "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc811_exchange_current_density_CircularLIB,
        # "Positive electrode density [kg.m-3]": 3262.0,
        # "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        # "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.405,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        # "Separator density [kg.m-3]": 397.0,
        # "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        # "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": electrolyte_tansference_number_Landesfeind2019,
        "Thermodynamic factor": electrolyte_thermodynamic_factor_Landesfeind2019,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Landesfeind2019,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Landesfeind2019,
        # experiment
        "Reference temperature [K]": 298.15,
        # "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 3.0,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 0.9060*29871, # calculated based on limiting NMC (s. matlab) 0.4599 ! exception for trying out
        "Initial concentration in positive electrode [mol.m-3]": 0.2180*67116, # calculated based on limiting NMC (s. matlab) 0.5475
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Chen2020"], #change later
    }

# V_ini_pos=nmc_LGM50_ocp_Chen2020(0.064)

# V_ini_neg=graphite_LGM50_ocp_Chen2020(1)
# print(V_ini_pos,V_ini_neg)

# def load_parameters():
#     """ Load the custom parameter set into PyBaMM """
#     pybamm.ParameterValues.update_from_dict(my_custom_parameters)