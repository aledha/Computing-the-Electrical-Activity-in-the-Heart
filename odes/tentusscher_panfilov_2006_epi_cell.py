import numpy

parameter = {
    "Buf_c": 0,
    "Buf_sr": 1,
    "Buf_ss": 2,
    "Ca_o": 3,
    "Cm": 4,
    "EC": 5,
    "F": 6,
    "K_NaCa": 7,
    "K_buf_c": 8,
    "K_buf_sr": 9,
    "K_buf_ss": 10,
    "K_mNa": 11,
    "K_mk": 12,
    "K_o": 13,
    "K_pCa": 14,
    "K_sat": 15,
    "K_up": 16,
    "Km_Ca": 17,
    "Km_Nai": 18,
    "Na_o": 19,
    "P_NaK": 20,
    "P_kna": 21,
    "R": 22,
    "T": 23,
    "V_c": 24,
    "V_leak": 25,
    "V_rel": 26,
    "V_sr": 27,
    "V_ss": 28,
    "V_xfer": 29,
    "Vmax_up": 30,
    "alpha": 31,
    "g_CaL": 32,
    "g_K1": 33,
    "g_Kr": 34,
    "g_Ks": 35,
    "g_Na": 36,
    "g_bca": 37,
    "g_bna": 38,
    "g_pCa": 39,
    "g_pK": 40,
    "g_to": 41,
    "gamma": 42,
    "k1_prime": 43,
    "k2_prime": 44,
    "k3": 45,
    "k4": 46,
    "max_sr": 47,
    "min_sr": 48,
    "stim_amplitude": 49,
    "stim_duration": 50,
    "stim_period": 51,
    "stim_start": 52,
}


def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    return parameter[name]


state = {
    "fCass": 0,
    "f": 1,
    "f2": 2,
    "r": 3,
    "s": 4,
    "Na_i": 5,
    "h": 6,
    "j": 7,
    "m": 8,
    "Xr1": 9,
    "Xr2": 10,
    "Xs": 11,
    "d": 12,
    "R_prime": 13,
    "Ca_i": 14,
    "K_i": 15,
    "V": 16,
    "Ca_SR": 17,
    "Ca_ss": 18,
}


def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    return state[name]


monitor = {
    "E_Ca": 0,
    "E_K": 1,
    "E_Ks": 2,
    "E_Na": 3,
    "fCass_inf": 4,
    "tau_fCass": 5,
    "alpha_d": 6,
    "alpha_h": 7,
    "alpha_j": 8,
    "alpha_m": 9,
    "alpha_xr1": 10,
    "alpha_xr2": 11,
    "alpha_xs": 12,
    "beta_d": 13,
    "beta_h": 14,
    "beta_j": 15,
    "beta_m": 16,
    "beta_xr1": 17,
    "beta_xr2": 18,
    "beta_xs": 19,
    "d_inf": 20,
    "f2_inf": 21,
    "f_inf": 22,
    "gamma_d": 23,
    "h_inf": 24,
    "j_inf": 25,
    "m_inf": 26,
    "r_inf": 27,
    "s_inf": 28,
    "tau_f": 29,
    "tau_f2": 30,
    "tau_r": 31,
    "tau_s": 32,
    "xr1_inf": 33,
    "xr2_inf": 34,
    "xs_inf": 35,
    "f_JCa_i_free": 36,
    "f_JCa_sr_free": 37,
    "f_JCa_ss_free": 38,
    "i_CaL": 39,
    "i_NaCa": 40,
    "i_NaK": 41,
    "i_Stim": 42,
    "i_leak": 43,
    "i_p_Ca": 44,
    "i_up": 45,
    "i_xfer": 46,
    "kcasr": 47,
    "i_b_Ca": 48,
    "alpha_K1": 49,
    "beta_K1": 50,
    "i_Kr": 51,
    "i_p_K": 52,
    "i_to": 53,
    "i_Ks": 54,
    "i_Na": 55,
    "i_b_Na": 56,
    "dfCass_dt": 57,
    "tau_h": 58,
    "tau_j": 59,
    "tau_m": 60,
    "tau_xr1": 61,
    "tau_xr2": 62,
    "tau_xs": 63,
    "tau_d": 64,
    "df_dt": 65,
    "df2_dt": 66,
    "dr_dt": 67,
    "ds_dt": 68,
    "k1": 69,
    "k2": 70,
    "ddt_Ca_i_total": 71,
    "xK1_inf": 72,
    "dNa_i_dt": 73,
    "dh_dt": 74,
    "dj_dt": 75,
    "dm_dt": 76,
    "dXr1_dt": 77,
    "dXr2_dt": 78,
    "dXs_dt": 79,
    "dd_dt": 80,
    "O": 81,
    "dR_prime_dt": 82,
    "dCa_i_dt": 83,
    "i_K1": 84,
    "i_rel": 85,
    "dK_i_dt": 86,
    "dV_dt": 87,
    "ddt_Ca_sr_total": 88,
    "ddt_Ca_ss_total": 89,
    "dCa_SR_dt": 90,
    "dCa_ss_dt": 91,
}


def monitor_index(name: str) -> int:
    """Return the index of the monitor with the given name

    Arguments
    ---------
    name : str
        The name of the monitor

    Returns
    -------
    int
        The index of the monitor

    Raises
    ------
    KeyError
        If the name is not a valid monitor
    """

    return monitor[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # Buf_c=0.2, Buf_sr=10.0, Buf_ss=0.4, Ca_o=2.0, Cm=185.0, EC=1.5
    # F=96.485, K_NaCa=1000.0, K_buf_c=0.001, K_buf_sr=0.3
    # K_buf_ss=0.00025, K_mNa=40.0, K_mk=1.0, K_o=5.4, K_pCa=0.0005
    # K_sat=0.1, K_up=0.00025, Km_Ca=1.38, Km_Nai=87.5, Na_o=140.0
    # P_NaK=2.724, P_kna=0.03, R=8.314, T=310.0, V_c=16404.0
    # V_leak=0.00036, V_rel=0.102, V_sr=1094.0, V_ss=54.68
    # V_xfer=0.0038, Vmax_up=0.006375, alpha=2.5, g_CaL=0.0398
    # g_K1=5.405, g_Kr=0.153, g_Ks=0.392, g_Na=14.838
    # g_bca=0.000592, g_bna=0.00029, g_pCa=0.1238, g_pK=0.0146
    # g_to=0.294, gamma=0.35, k1_prime=0.15, k2_prime=0.045
    # k3=0.06, k4=0.005, max_sr=2.5, min_sr=1, stim_amplitude=-52.0
    # stim_duration=1.0, stim_period=1000.0, stim_start=10.0

    parameters = numpy.array(
        [
            0.2,
            10.0,
            0.4,
            2.0,
            185.0,
            1.5,
            96.485,
            1000.0,
            0.001,
            0.3,
            0.00025,
            40.0,
            1.0,
            5.4,
            0.0005,
            0.1,
            0.00025,
            1.38,
            87.5,
            140.0,
            2.724,
            0.03,
            8.314,
            310.0,
            16404.0,
            0.00036,
            0.102,
            1094.0,
            54.68,
            0.0038,
            0.006375,
            2.5,
            0.0398,
            5.405,
            0.153,
            0.392,
            14.838,
            0.000592,
            0.00029,
            0.1238,
            0.0146,
            0.294,
            0.35,
            0.15,
            0.045,
            0.06,
            0.005,
            2.5,
            1,
            -52.0,
            1.0,
            1000.0,
            10.0,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # fCass=0.9953, f=0.7888, f2=0.9755, r=2.42e-08, s=0.999998
    # Na_i=8.604, h=0.7444, j=0.7045, m=0.00172, Xr1=0.00621
    # Xr2=0.4712, Xs=0.0095, d=3.373e-05, R_prime=0.9073
    # Ca_i=0.000126, K_i=136.89, V=-85.23, Ca_SR=3.64
    # Ca_ss=0.00036

    states = numpy.array(
        [
            0.9953,
            0.7888,
            0.9755,
            2.42e-08,
            0.999998,
            8.604,
            0.7444,
            0.7045,
            0.00172,
            0.00621,
            0.4712,
            0.0095,
            3.373e-05,
            0.9073,
            0.000126,
            136.89,
            -85.23,
            3.64,
            0.00036,
        ],
        dtype=numpy.float64,
    )

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Na_i = states[5]
    h = states[6]
    j = states[7]
    m = states[8]
    Xr1 = states[9]
    Xr2 = states[10]
    Xs = states[11]
    d = states[12]
    R_prime = states[13]
    Ca_i = states[14]
    K_i = states[15]
    V = states[16]
    Ca_SR = states[17]
    Ca_ss = states[18]

    # Assign parameters
    Buf_c = parameters[0]
    Buf_sr = parameters[1]
    Buf_ss = parameters[2]
    Ca_o = parameters[3]
    Cm = parameters[4]
    EC = parameters[5]
    F = parameters[6]
    K_NaCa = parameters[7]
    K_buf_c = parameters[8]
    K_buf_sr = parameters[9]
    K_buf_ss = parameters[10]
    K_mNa = parameters[11]
    K_mk = parameters[12]
    K_o = parameters[13]
    K_pCa = parameters[14]
    K_sat = parameters[15]
    K_up = parameters[16]
    Km_Ca = parameters[17]
    Km_Nai = parameters[18]
    Na_o = parameters[19]
    P_NaK = parameters[20]
    P_kna = parameters[21]
    R = parameters[22]
    T = parameters[23]
    V_c = parameters[24]
    V_leak = parameters[25]
    V_rel = parameters[26]
    V_sr = parameters[27]
    V_ss = parameters[28]
    V_xfer = parameters[29]
    Vmax_up = parameters[30]
    alpha = parameters[31]
    g_CaL = parameters[32]
    g_K1 = parameters[33]
    g_Kr = parameters[34]
    g_Ks = parameters[35]
    g_Na = parameters[36]
    g_bca = parameters[37]
    g_bna = parameters[38]
    g_pCa = parameters[39]
    g_pK = parameters[40]
    g_to = parameters[41]
    gamma = parameters[42]
    k1_prime = parameters[43]
    k2_prime = parameters[44]
    k3 = parameters[45]
    k4 = parameters[46]
    max_sr = parameters[47]
    min_sr = parameters[48]
    stim_amplitude = parameters[49]
    stim_duration = parameters[50]
    stim_period = parameters[51]
    stim_start = parameters[52]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    i_leak = V_leak * (Ca_SR - Ca_i)
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    i_b_Ca = g_bca * (-E_Ca + V)
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    i_to = (s * (g_to * r)) * (-E_K + V)
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    i_b_Na = g_bna * (-E_Na + V)
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    values[0] = dfCass_dt
    tau_h = 1 / (alpha_h + beta_h)
    tau_j = 1 / (alpha_j + beta_j)
    tau_m = (1 * alpha_m) * beta_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    df_dt = (-f + f_inf) / tau_f
    values[1] = df_dt
    df2_dt = (-f2 + f2_inf) / tau_f2
    values[2] = df2_dt
    dr_dt = (-r + r_inf) / tau_r
    values[3] = dr_dt
    ds_dt = (-s + s_inf) / tau_s
    values[4] = ds_dt
    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[5] = dNa_i_dt
    dh_dt = (-h + h_inf) / tau_h
    values[6] = dh_dt
    dj_dt = (-j + j_inf) / tau_j
    values[7] = dj_dt
    dm_dt = (-m + m_inf) / tau_m
    values[8] = dm_dt
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    values[9] = dXr1_dt
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    values[10] = dXr2_dt
    dXs_dt = (-Xs + xs_inf) / tau_xs
    values[11] = dXs_dt
    dd_dt = (-d + d_inf) / tau_d
    values[12] = dd_dt
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    values[13] = dR_prime_dt
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[14] = dCa_i_dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[15] = dK_i_dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[16] = dV_dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[17] = dCa_SR_dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[18] = dCa_ss_dt

    return values


def monitor_values(t, states, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Na_i = states[5]
    h = states[6]
    j = states[7]
    m = states[8]
    Xr1 = states[9]
    Xr2 = states[10]
    Xs = states[11]
    d = states[12]
    R_prime = states[13]
    Ca_i = states[14]
    K_i = states[15]
    V = states[16]
    Ca_SR = states[17]
    Ca_ss = states[18]

    # Assign parameters
    Buf_c = parameters[0]
    Buf_sr = parameters[1]
    Buf_ss = parameters[2]
    Ca_o = parameters[3]
    Cm = parameters[4]
    EC = parameters[5]
    F = parameters[6]
    K_NaCa = parameters[7]
    K_buf_c = parameters[8]
    K_buf_sr = parameters[9]
    K_buf_ss = parameters[10]
    K_mNa = parameters[11]
    K_mk = parameters[12]
    K_o = parameters[13]
    K_pCa = parameters[14]
    K_sat = parameters[15]
    K_up = parameters[16]
    Km_Ca = parameters[17]
    Km_Nai = parameters[18]
    Na_o = parameters[19]
    P_NaK = parameters[20]
    P_kna = parameters[21]
    R = parameters[22]
    T = parameters[23]
    V_c = parameters[24]
    V_leak = parameters[25]
    V_rel = parameters[26]
    V_sr = parameters[27]
    V_ss = parameters[28]
    V_xfer = parameters[29]
    Vmax_up = parameters[30]
    alpha = parameters[31]
    g_CaL = parameters[32]
    g_K1 = parameters[33]
    g_Kr = parameters[34]
    g_Ks = parameters[35]
    g_Na = parameters[36]
    g_bca = parameters[37]
    g_bna = parameters[38]
    g_pCa = parameters[39]
    g_pK = parameters[40]
    g_to = parameters[41]
    gamma = parameters[42]
    k1_prime = parameters[43]
    k2_prime = parameters[44]
    k3 = parameters[45]
    k4 = parameters[46]
    max_sr = parameters[47]
    min_sr = parameters[48]
    stim_amplitude = parameters[49]
    stim_duration = parameters[50]
    stim_period = parameters[51]
    stim_start = parameters[52]

    # Assign expressions
    shape = 92 if len(states.shape) == 1 else (92, states.shape[1])
    values = numpy.zeros(shape)
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    values[0] = E_Ca
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    values[1] = E_K
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    values[2] = E_Ks
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    values[3] = E_Na
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    values[4] = fCass_inf
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    values[5] = tau_fCass
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    values[6] = alpha_d
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    values[7] = alpha_h
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    values[8] = alpha_j
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    values[9] = alpha_m
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    values[10] = alpha_xr1
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    values[11] = alpha_xr2
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    values[12] = alpha_xs
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    values[13] = beta_d
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    values[14] = beta_h
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    values[15] = beta_j
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    values[16] = beta_m
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    values[17] = beta_xr1
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    values[18] = beta_xr2
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    values[19] = beta_xs
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    values[20] = d_inf
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    values[21] = f2_inf
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    values[22] = f_inf
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    values[23] = gamma_d
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    values[24] = h_inf
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    values[25] = j_inf
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    values[26] = m_inf
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    values[27] = r_inf
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    values[28] = s_inf
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    values[29] = tau_f
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    values[30] = tau_f2
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    values[31] = tau_r
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    values[32] = tau_s
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    values[33] = xr1_inf
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    values[34] = xr2_inf
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    values[35] = xs_inf
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    values[36] = f_JCa_i_free
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    values[37] = f_JCa_sr_free
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    values[38] = f_JCa_ss_free
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    values[39] = i_CaL
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    values[40] = i_NaCa
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    values[41] = i_NaK
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    values[42] = i_Stim
    i_leak = V_leak * (Ca_SR - Ca_i)
    values[43] = i_leak
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    values[44] = i_p_Ca
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    values[45] = i_up
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    values[46] = i_xfer
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    values[47] = kcasr
    i_b_Ca = g_bca * (-E_Ca + V)
    values[48] = i_b_Ca
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    values[49] = alpha_K1
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    values[50] = beta_K1
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    values[51] = i_Kr
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    values[52] = i_p_K
    i_to = (s * (g_to * r)) * (-E_K + V)
    values[53] = i_to
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    values[54] = i_Ks
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    values[55] = i_Na
    i_b_Na = g_bna * (-E_Na + V)
    values[56] = i_b_Na
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    values[57] = dfCass_dt
    tau_h = 1 / (alpha_h + beta_h)
    values[58] = tau_h
    tau_j = 1 / (alpha_j + beta_j)
    values[59] = tau_j
    tau_m = (1 * alpha_m) * beta_m
    values[60] = tau_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    values[61] = tau_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    values[62] = tau_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    values[63] = tau_xs
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    values[64] = tau_d
    df_dt = (-f + f_inf) / tau_f
    values[65] = df_dt
    df2_dt = (-f2 + f2_inf) / tau_f2
    values[66] = df2_dt
    dr_dt = (-r + r_inf) / tau_r
    values[67] = dr_dt
    ds_dt = (-s + s_inf) / tau_s
    values[68] = ds_dt
    k1 = k1_prime / kcasr
    values[69] = k1
    k2 = k2_prime * kcasr
    values[70] = k2
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    values[71] = ddt_Ca_i_total
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    values[72] = xK1_inf
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[73] = dNa_i_dt
    dh_dt = (-h + h_inf) / tau_h
    values[74] = dh_dt
    dj_dt = (-j + j_inf) / tau_j
    values[75] = dj_dt
    dm_dt = (-m + m_inf) / tau_m
    values[76] = dm_dt
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    values[77] = dXr1_dt
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    values[78] = dXr2_dt
    dXs_dt = (-Xs + xs_inf) / tau_xs
    values[79] = dXs_dt
    dd_dt = (-d + d_inf) / tau_d
    values[80] = dd_dt
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    values[81] = O
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    values[82] = dR_prime_dt
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[83] = dCa_i_dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    values[84] = i_K1
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    values[85] = i_rel
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[86] = dK_i_dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[87] = dV_dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    values[88] = ddt_Ca_sr_total
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    values[89] = ddt_Ca_ss_total
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[90] = dCa_SR_dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[91] = dCa_ss_dt

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    fCass = states[0]
    f = states[1]
    f2 = states[2]
    r = states[3]
    s = states[4]
    Na_i = states[5]
    h = states[6]
    j = states[7]
    m = states[8]
    Xr1 = states[9]
    Xr2 = states[10]
    Xs = states[11]
    d = states[12]
    R_prime = states[13]
    Ca_i = states[14]
    K_i = states[15]
    V = states[16]
    Ca_SR = states[17]
    Ca_ss = states[18]

    # Assign parameters
    Buf_c = parameters[0]
    Buf_sr = parameters[1]
    Buf_ss = parameters[2]
    Ca_o = parameters[3]
    Cm = parameters[4]
    EC = parameters[5]
    F = parameters[6]
    K_NaCa = parameters[7]
    K_buf_c = parameters[8]
    K_buf_sr = parameters[9]
    K_buf_ss = parameters[10]
    K_mNa = parameters[11]
    K_mk = parameters[12]
    K_o = parameters[13]
    K_pCa = parameters[14]
    K_sat = parameters[15]
    K_up = parameters[16]
    Km_Ca = parameters[17]
    Km_Nai = parameters[18]
    Na_o = parameters[19]
    P_NaK = parameters[20]
    P_kna = parameters[21]
    R = parameters[22]
    T = parameters[23]
    V_c = parameters[24]
    V_leak = parameters[25]
    V_rel = parameters[26]
    V_sr = parameters[27]
    V_ss = parameters[28]
    V_xfer = parameters[29]
    Vmax_up = parameters[30]
    alpha = parameters[31]
    g_CaL = parameters[32]
    g_K1 = parameters[33]
    g_Kr = parameters[34]
    g_Ks = parameters[35]
    g_Na = parameters[36]
    g_bca = parameters[37]
    g_bna = parameters[38]
    g_pCa = parameters[39]
    g_pK = parameters[40]
    g_to = parameters[41]
    gamma = parameters[42]
    k1_prime = parameters[43]
    k2_prime = parameters[44]
    k3 = parameters[45]
    k4 = parameters[46]
    max_sr = parameters[47]
    min_sr = parameters[48]
    stim_amplitude = parameters[49]
    stim_duration = parameters[50]
    stim_period = parameters[51]
    stim_start = parameters[52]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    E_Ca = (((0.5 * R) * T) / F) * numpy.log(Ca_o / Ca_i)
    E_K = ((R * T) / F) * numpy.log(K_o / K_i)
    E_Ks = ((R * T) / F) * numpy.log((K_o + Na_o * P_kna) / (K_i + Na_i * P_kna))
    E_Na = ((R * T) / F) * numpy.log(Na_o / Na_i)
    fCass_inf = 0.4 + 0.6 / ((Ca_ss / 0.05) ** 2 + 1)
    tau_fCass = 2 + 80 / ((Ca_ss / 0.05) ** 2 + 1)
    alpha_d = 0.25 + 1.4 / (numpy.exp((-V - 35) / 13) + 1)
    alpha_h = numpy.where((V < -40), 0.057 * numpy.exp((-(V + 80)) / 6.8), 0)
    alpha_j = numpy.where(
        (V < -40),
        (
            (
                (V + 37.78)
                * (
                    (-25428) * numpy.exp(0.2444 * V)
                    - 6.948e-06 * numpy.exp((-0.04391) * V)
                )
            )
            / 1
        )
        / (numpy.exp(0.311 * (V + 79.23)) + 1),
        0,
    )
    alpha_m = 1 / (numpy.exp((-V - 60) / 5) + 1)
    alpha_xr1 = 450 / (numpy.exp((-V - 45) / 10) + 1)
    alpha_xr2 = 3 / (numpy.exp((-V - 60) / 20) + 1)
    alpha_xs = 1400 / numpy.sqrt(numpy.exp((5 - V) / 6) + 1)
    beta_d = 1.4 / (numpy.exp((V + 5) / 5) + 1)
    beta_h = numpy.where(
        (V < -40),
        2.7 * numpy.exp(0.079 * V) + 310000 * numpy.exp(0.3485 * V),
        0.77 / ((0.13 * (numpy.exp((V + 10.66) / ((-11.1))) + 1))),
    )
    beta_j = numpy.where(
        (V < -40),
        (0.02424 * numpy.exp((-0.01052) * V))
        / (numpy.exp((-0.1378) * (V + 40.14)) + 1),
        (0.6 * numpy.exp(0.057 * V)) / (numpy.exp((-0.1) * (V + 32)) + 1),
    )
    beta_m = 0.1 / (numpy.exp((V + 35) / 5) + 1) + 0.1 / (numpy.exp((V - 50) / 200) + 1)
    beta_xr1 = 6 / (numpy.exp((V + 30) / 11.5) + 1)
    beta_xr2 = 1.12 / (numpy.exp((V - 60) / 20) + 1)
    beta_xs = 1 / (numpy.exp((V - 35) / 15) + 1)
    d_inf = 1 / (numpy.exp((-V - 8) / 7.5) + 1)
    f2_inf = 0.33 + 0.67 / (numpy.exp((V + 35) / 7) + 1)
    f_inf = 1 / (numpy.exp((V + 20) / 7) + 1)
    gamma_d = 1 / (numpy.exp((50 - V) / 20) + 1)
    h_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    j_inf = 1 / (numpy.exp((V + 71.55) / 7.43) + 1) ** 2
    m_inf = 1 / (numpy.exp((-V - 56.86) / 9.03) + 1) ** 2
    r_inf = 1 / (numpy.exp((20 - V) / 6) + 1)
    s_inf = 1 / (numpy.exp((V + 20) / 5) + 1)
    tau_f = (
        (
            1102.5 * numpy.exp((-((V + 27) ** 2)) / 225)
            + 200 / (numpy.exp((13 - V) / 10) + 1)
        )
        + 180 / (numpy.exp((V + 30) / 10) + 1)
    ) + 20
    tau_f2 = (
        562 * numpy.exp((-((V + 27) ** 2)) / 240) + 31 / (numpy.exp((25 - V) / 10) + 1)
    ) + 80 / (numpy.exp((V + 30) / 10) + 1)
    tau_r = 9.5 * numpy.exp((-((V + 40) ** 2)) / 1800) + 0.8
    tau_s = (
        85 * numpy.exp((-((V + 45) ** 2)) / 320) + 5 / (numpy.exp((V - 20) / 5) + 1)
    ) + 3
    xr1_inf = 1 / (numpy.exp((-V - 26) / 7) + 1)
    xr2_inf = 1 / (numpy.exp((V + 88) / 24) + 1)
    xs_inf = 1 / (numpy.exp((-V - 5) / 14) + 1)
    f_JCa_i_free = 1 / ((Buf_c * K_buf_c) / (Ca_i + K_buf_c) ** 2 + 1)
    f_JCa_sr_free = 1 / ((Buf_sr * K_buf_sr) / (Ca_SR + K_buf_sr) ** 2 + 1)
    f_JCa_ss_free = 1 / ((Buf_ss * K_buf_ss) / (Ca_ss + K_buf_ss) ** 2 + 1)
    i_CaL = (
        ((F**2 * ((4 * (fCass * (f2 * (f * (d * g_CaL))))) * (V - 15))) / ((R * T)))
        * (-Ca_o + (0.25 * Ca_ss) * numpy.exp((F * (2 * (V - 15))) / ((R * T))))
    ) / (numpy.exp((F * (2 * (V - 15))) / ((R * T))) - 1)
    i_NaCa = (
        K_NaCa
        * (
            Ca_o * (Na_i**3 * numpy.exp((F * (V * gamma)) / ((R * T))))
            - alpha * Ca_i * (Na_o**3 * numpy.exp((F * (V * (gamma - 1))) / ((R * T))))
        )
    ) / (
        (
            ((Ca_o + Km_Ca) * (Km_Nai**3 + Na_o**3))
            * (K_sat * numpy.exp((F * (V * (gamma - 1))) / ((R * T))) + 1)
        )
    )
    i_NaK = ((Na_i * ((K_o * P_NaK) / (K_mk + K_o))) / (K_mNa + Na_i)) / (
        (0.1245 * numpy.exp((F * ((-0.1) * V)) / ((R * T))) + 1)
        + 0.0353 * numpy.exp((F * (-V)) / ((R * T)))
    )
    i_Stim = numpy.where(
        numpy.logical_and(
            (stim_start <= -stim_period * numpy.floor(t / stim_period) + t),
            (
                stim_duration + stim_start
                >= -stim_period * numpy.floor(t / stim_period) + t
            ),
        ),
        stim_amplitude,
        0,
    )
    i_leak = V_leak * (Ca_SR - Ca_i)
    i_p_Ca = (Ca_i * g_pCa) / (Ca_i + K_pCa)
    i_up = Vmax_up / (1 + K_up**2 / Ca_i**2)
    i_xfer = V_xfer * (-Ca_i + Ca_ss)
    kcasr = max_sr - (max_sr - min_sr) / ((EC / Ca_SR) ** 2 + 1)
    i_b_Ca = g_bca * (-E_Ca + V)
    alpha_K1 = 0.1 / (numpy.exp(0.06 * ((-E_K + V) - 200)) + 1)
    beta_K1 = (
        3 * numpy.exp(0.0002 * ((-E_K + V) + 100)) + numpy.exp(0.1 * ((-E_K + V) - 10))
    ) / (numpy.exp((-0.5) * (-E_K + V)) + 1)
    i_Kr = (Xr2 * (Xr1 * ((0.4303314829119352 * numpy.sqrt(K_o)) * g_Kr))) * (-E_K + V)
    i_p_K = (g_pK * (-E_K + V)) / (numpy.exp((25 - V) / 5.98) + 1)
    i_to = (s * (g_to * r)) * (-E_K + V)
    i_Ks = (Xs**2 * g_Ks) * (-E_Ks + V)
    i_Na = (j * (h * (g_Na * m**3))) * (-E_Na + V)
    i_b_Na = g_bna * (-E_Na + V)
    dfCass_dt = (-fCass + fCass_inf) / tau_fCass
    dfCass_dt_linearized = -1 / tau_fCass
    values[0] = (
        dfCass_dt * (numpy.exp(dfCass_dt_linearized * dt) - 1) / dfCass_dt_linearized
        + fCass
    )
    tau_h = 1 / (alpha_h + beta_h)
    tau_j = 1 / (alpha_j + beta_j)
    tau_m = (1 * alpha_m) * beta_m
    tau_xr1 = (1 * alpha_xr1) * beta_xr1
    tau_xr2 = (1 * alpha_xr2) * beta_xr2
    tau_xs = (1 * alpha_xs) * beta_xs + 80
    tau_d = (1 * alpha_d) * beta_d + gamma_d
    df_dt = (-f + f_inf) / tau_f
    df_dt_linearized = -1 / tau_f
    values[1] = df_dt * (numpy.exp(df_dt_linearized * dt) - 1) / df_dt_linearized + f
    df2_dt = (-f2 + f2_inf) / tau_f2
    df2_dt_linearized = -1 / tau_f2
    values[2] = (
        df2_dt * (numpy.exp(df2_dt_linearized * dt) - 1) / df2_dt_linearized + f2
    )
    dr_dt = (-r + r_inf) / tau_r
    dr_dt_linearized = -1 / tau_r
    values[3] = dr_dt * (numpy.exp(dr_dt_linearized * dt) - 1) / dr_dt_linearized + r
    ds_dt = (-s + s_inf) / tau_s
    ds_dt_linearized = -1 / tau_s
    values[4] = ds_dt * (numpy.exp(ds_dt_linearized * dt) - 1) / ds_dt_linearized + s
    k1 = k1_prime / kcasr
    k2 = k2_prime * kcasr
    ddt_Ca_i_total = i_xfer + (
        (Cm * (-(-2 * i_NaCa + (i_b_Ca + i_p_Ca)))) / ((F * (2 * V_c)))
        + (V_sr * (i_leak - i_up)) / V_c
    )
    xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
    dNa_i_dt = Cm * ((-(3 * i_NaCa + (3 * i_NaK + (i_Na + i_b_Na)))) / ((F * V_c)))
    values[5] = Na_i + dNa_i_dt * dt
    dh_dt = (-h + h_inf) / tau_h
    dh_dt_linearized = -1 / tau_h
    values[6] = dh_dt * (numpy.exp(dh_dt_linearized * dt) - 1) / dh_dt_linearized + h
    dj_dt = (-j + j_inf) / tau_j
    dj_dt_linearized = -1 / tau_j
    values[7] = dj_dt * (numpy.exp(dj_dt_linearized * dt) - 1) / dj_dt_linearized + j
    dm_dt = (-m + m_inf) / tau_m
    dm_dt_linearized = -1 / tau_m
    values[8] = dm_dt * (numpy.exp(dm_dt_linearized * dt) - 1) / dm_dt_linearized + m
    dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1
    dXr1_dt_linearized = -1 / tau_xr1
    values[9] = (
        Xr1 + dXr1_dt * (numpy.exp(dXr1_dt_linearized * dt) - 1) / dXr1_dt_linearized
    )
    dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2
    dXr2_dt_linearized = -1 / tau_xr2
    values[10] = (
        Xr2 + dXr2_dt * (numpy.exp(dXr2_dt_linearized * dt) - 1) / dXr2_dt_linearized
    )
    dXs_dt = (-Xs + xs_inf) / tau_xs
    dXs_dt_linearized = -1 / tau_xs
    values[11] = (
        Xs + dXs_dt * (numpy.exp(dXs_dt_linearized * dt) - 1) / dXs_dt_linearized
    )
    dd_dt = (-d + d_inf) / tau_d
    dd_dt_linearized = -1 / tau_d
    values[12] = d + dd_dt * (numpy.exp(dd_dt_linearized * dt) - 1) / dd_dt_linearized
    O = (R_prime * (Ca_ss**2 * k1)) / (Ca_ss**2 * k1 + k3)
    dR_prime_dt = R_prime * (Ca_ss * (-k2)) + k4 * (1 - R_prime)
    dR_prime_dt_linearized = Ca_ss * (-k2) - k4
    values[13] = R_prime + numpy.where(
        numpy.logical_or(
            (dR_prime_dt_linearized > 1e-08), (dR_prime_dt_linearized < -1e-08)
        ),
        dR_prime_dt
        * (numpy.exp(dR_prime_dt_linearized * dt) - 1)
        / dR_prime_dt_linearized,
        dR_prime_dt * dt,
    )
    dCa_i_dt = ddt_Ca_i_total * f_JCa_i_free
    values[14] = Ca_i + dCa_i_dt * dt
    i_K1 = ((0.4303314829119352 * numpy.sqrt(K_o)) * (g_K1 * xK1_inf)) * (-E_K + V)
    i_rel = (O * V_rel) * (Ca_SR - Ca_ss)
    dK_i_dt = Cm * (
        (-(-2 * i_NaK + (i_Stim + (i_p_K + (i_Ks + (i_Kr + (i_K1 + i_to)))))))
        / ((F * V_c))
    )
    values[15] = K_i + dK_i_dt * dt
    dV_dt = -(
        i_Stim
        + (
            i_p_Ca
            + (
                i_p_K
                + (
                    i_b_Ca
                    + (
                        i_NaCa
                        + (
                            i_b_Na
                            + (
                                i_Na
                                + (i_NaK + (i_CaL + (i_Ks + (i_Kr + (i_K1 + i_to)))))
                            )
                        )
                    )
                )
            )
        )
    )
    values[16] = V + dV_dt * dt
    ddt_Ca_sr_total = i_up - (i_leak + i_rel)
    ddt_Ca_ss_total = ((Cm * (-i_CaL)) / ((F * (2 * V_ss))) + (V_sr * i_rel) / V_ss) - (
        V_c * i_xfer
    ) / V_ss
    dCa_SR_dt = ddt_Ca_sr_total * f_JCa_sr_free
    values[17] = Ca_SR + dCa_SR_dt * dt
    dCa_ss_dt = ddt_Ca_ss_total * f_JCa_ss_free
    values[18] = Ca_ss + dCa_ss_dt * dt

    return values
