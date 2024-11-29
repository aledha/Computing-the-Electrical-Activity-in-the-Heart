import numpy

parameter = {
    "a": 0,
    "b": 1,
    "c_1": 2,
    "c_2": 3,
    "c_3": 4,
    "stim_amplitude": 5,
    "stim_duration": 6,
    "stim_period": 7,
    "stim_start": 8,
    "v_peak": 9,
    "v_rest": 10,
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


state = {"s": 0, "v": 1}


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


monitor = {"v_amp": 0, "i_Stim": 1, "ds_dt": 2, "v_th": 3, "I": 4, "dv_dt": 5}


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
    # a=0.13, b=0.013, c_1=0.26, c_2=0.1, c_3=1.0, stim_amplitude=50
    # stim_duration=1, stim_period=1000, stim_start=1, v_peak=40.0
    # v_rest=-85.0

    parameters = numpy.array(
        [0.13, 0.013, 0.26, 0.1, 1.0, 50, 1, 1000, 1, 40.0, -85.0], dtype=numpy.float64
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # s=0.0, v=-85.0

    states = numpy.array([0.0, -85.0], dtype=numpy.float64)

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters
    a = parameters[0]
    b = parameters[1]
    c_1 = parameters[2]
    c_2 = parameters[3]
    c_3 = parameters[4]
    stim_amplitude = parameters[5]
    stim_duration = parameters[6]
    stim_period = parameters[7]
    stim_start = parameters[8]
    v_peak = parameters[9]
    v_rest = parameters[10]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    v_amp = v_peak - v_rest
    i_Stim = (
        stim_amplitude * (1 - 1 / (numpy.exp(-5.0 * stim_start + 5.0 * t) + 1))
    ) / (numpy.exp(-5.0 * stim_duration - 5.0 * stim_start + 5.0 * t) + 1)
    ds_dt = b * (-c_3 * s + (v - v_rest))
    values[0] = ds_dt
    v_th = a * v_amp + v_rest
    I = -s * (c_2 / v_amp) * (v - v_rest) + (
        ((c_1 / v_amp**2) * (v - v_rest)) * (v - v_th)
    ) * (-v + v_peak)
    dv_dt = I + i_Stim
    values[1] = dv_dt

    return values


def monitor_values(t, states, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters
    a = parameters[0]
    b = parameters[1]
    c_1 = parameters[2]
    c_2 = parameters[3]
    c_3 = parameters[4]
    stim_amplitude = parameters[5]
    stim_duration = parameters[6]
    stim_period = parameters[7]
    stim_start = parameters[8]
    v_peak = parameters[9]
    v_rest = parameters[10]

    # Assign expressions
    shape = 6 if len(states.shape) == 1 else (6, states.shape[1])
    values = numpy.zeros(shape)
    v_amp = v_peak - v_rest
    values[0] = v_amp
    i_Stim = (
        stim_amplitude * (1 - 1 / (numpy.exp(-5.0 * stim_start + 5.0 * t) + 1))
    ) / (numpy.exp(-5.0 * stim_duration - 5.0 * stim_start + 5.0 * t) + 1)
    values[1] = i_Stim
    ds_dt = b * (-c_3 * s + (v - v_rest))
    values[2] = ds_dt
    v_th = a * v_amp + v_rest
    values[3] = v_th
    I = -s * (c_2 / v_amp) * (v - v_rest) + (
        ((c_1 / v_amp**2) * (v - v_rest)) * (v - v_th)
    ) * (-v + v_peak)
    values[4] = I
    dv_dt = I + i_Stim
    values[5] = dv_dt

    return values


def forward_explicit_euler(states, t, dt, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters
    a = parameters[0]
    b = parameters[1]
    c_1 = parameters[2]
    c_2 = parameters[3]
    c_3 = parameters[4]
    stim_amplitude = parameters[5]
    stim_duration = parameters[6]
    stim_period = parameters[7]
    stim_start = parameters[8]
    v_peak = parameters[9]
    v_rest = parameters[10]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    v_amp = v_peak - v_rest
    i_Stim = (
        stim_amplitude * (1 - 1 / (numpy.exp(-5.0 * stim_start + 5.0 * t) + 1))
    ) / (numpy.exp(-5.0 * stim_duration - 5.0 * stim_start + 5.0 * t) + 1)
    ds_dt = b * (-c_3 * s + (v - v_rest))
    values[0] = ds_dt * dt + s
    v_th = a * v_amp + v_rest
    I = -s * (c_2 / v_amp) * (v - v_rest) + (
        ((c_1 / v_amp**2) * (v - v_rest)) * (v - v_th)
    ) * (-v + v_peak)
    dv_dt = I + i_Stim
    values[1] = dt * dv_dt + v

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters
    a = parameters[0]
    b = parameters[1]
    c_1 = parameters[2]
    c_2 = parameters[3]
    c_3 = parameters[4]
    stim_amplitude = parameters[5]
    stim_duration = parameters[6]
    stim_period = parameters[7]
    stim_start = parameters[8]
    v_peak = parameters[9]
    v_rest = parameters[10]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    v_amp = v_peak - v_rest
    i_Stim = (
        stim_amplitude * (1 - 1 / (numpy.exp(-5.0 * stim_start + 5.0 * t) + 1))
    ) / (numpy.exp(-5.0 * stim_duration - 5.0 * stim_start + 5.0 * t) + 1)
    ds_dt = b * (-c_3 * s + (v - v_rest))
    ds_dt_linearized = -b * c_3
    values[0] = s + numpy.where(
        numpy.logical_or((ds_dt_linearized > 1e-08), (ds_dt_linearized < -1e-08)),
        ds_dt * (numpy.exp(ds_dt_linearized * dt) - 1) / ds_dt_linearized,
        ds_dt * dt,
    )
    v_th = a * v_amp + v_rest
    I = -s * (c_2 / v_amp) * (v - v_rest) + (
        ((c_1 / v_amp**2) * (v - v_rest)) * (v - v_th)
    ) * (-v + v_peak)
    dv_dt = I + i_Stim
    values[1] = dt * dv_dt + v

    return values
