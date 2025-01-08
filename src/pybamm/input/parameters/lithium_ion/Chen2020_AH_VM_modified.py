import pybamm
import numpy as np

def graphite_LGM50_ocp_Chen2020(sto):
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
        1.110218413725510 * np.exp(-75.516461602271760 * sto)
        + 0.147841727791915
        - 7.664571570599029e-05* np.tanh(15.146171864949494 * (sto - 0.937038166931668))
        - 0.042765300282118 * np.tanh(23.425076882458310 * (sto - 0.153274739170637))
        - 0.020449578603550 * np.tanh(38.137812807891820 * (sto - 0.459953786691162))
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


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
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
    m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def nmc_LGM50_ocp_Chen2020(sto):
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
    # tried polynomfit - extrapolation went to far off
    # u_eq = (
    #     - 4.399785836560637e+03 * sto**8
    #     + 1.477519507637824e+04 * sto**7
    #     - 2.069917513956780e+04 * sto**6
    #     + 1.570547118537844e+04 * sto**5
    #     - 7.020807789622281e+03 * sto**4
    #     + 1.886330365579380e+03 * sto**3
    #     - 2.960973135106860e+02 * sto**2
    #     + 23.335819747012845 * sto
    #     + 3.489817641449543
    # )

    # fit was done in correctly. 
    # u_eq = ( 
    #     -1.02583318170468 * np.exp(-3.29273222104149 * sto)
    #     + 3.62480991176346
    #     - 0.175189948055655 * np.tanh(8 * (sto + 2))
    #     - 1.03071298522941 * np.tanh(3.20197155934937 * (sto - 0.126648381506623))
    #     -1.37758874029945 * np.tanh(3.66561441475660 * (sto - 1.09106043792285))
    # )

    # polynomical fit with extrapolation points to limit errors due small extrapolation
    # x-range: 0.128 to 0.741
    # The fit is not steep enough at the edges like in Chen, so that the model finds other solutions
    # u_eq = (
    #     - 3.922450690509036e+03 * sto**8
    #     + 1.425016457066832e+04 * sto**7
    #     - 2.172335251120002e+04 * sto**6
    #     + 1.806465191496407e+04 * sto**5
    #     - 8.928062261796630e+03 * sto**4
    #     + 2.679755766826636e+03 * sto**3
    #     - 4.763955221149448e+02 * sto**2
    #     + 44.479420547742215 * sto
    #     + 2.552594308455391
    # )

    # accurate polynomical fit with extrapolation points to limit errors due small extrapolation
    # x-range: 0.128 to 0.741
    u_eq = (
        - 1.043048142343884e+04 * sto**9
        + 4.962669673525584e+04 * sto**8
        - 1.018714014435327e+05 * sto**7
        + 1.181473040728662e+05 * sto**6
        - 8.512801221395213e+04 * sto**5
        + 3.943050938308326e+04 * sto**4
        - 1.171024602277544e+04 * sto**3
        + 2.142589509608000e+03 * sto**2
        - 2.193442437976997e+02 * sto
        + 13.871365583439971
    )

    return u_eq


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
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
    m_ref = 3.42e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 17800
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e


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
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 200000.0,
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "Inner SEI open-circuit potential [V]": 0.1,
        "Outer SEI open-circuit potential [V]": 0.8,
        "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial inner SEI thickness [m]": 2.5e-09,
        "Initial outer SEI thickness [m]": 2.5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 8.52e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 7.56e-05,
        "Positive current collector thickness [m]": 1.6e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        "Negative electrode diffusivity [m2.s-1]": 3.3e-14,
        "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
        "Negative electrode porosity": 0.25,
        "Negative electrode active material volume fraction": 0.75,
        "Negative particle radius [m]": 5.86e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        "Positive electrode diffusivity [m2.s-1]": 4e-15, #NMC811_DiffKoeffi_BA_RasmusBewer_Au1, # NMC811_DiffKoeffi_BA_RasmusBewer_Au1, # Chen2020 constant at 4e-15 NMC811_DiffKoeffi_BA_RasmusBewer_Li
        "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.2594,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
        # experiment
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 3.0,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,
        "Initial concentration in positive electrode [mol.m-3]": 17038.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["Chen2020"],
    }

# V_ini_pos=nmc_LGM50_ocp_Chen2020(0.064)

# V_ini_neg=graphite_LGM50_ocp_Chen2020(1)
# print(V_ini_pos,V_ini_neg)

def load_parameters():
    """ Load the custom parameter set into PyBaMM """
    pybamm.ParameterValues.update_from_dict(my_custom_parameters)