import numpy

parameter = {}


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


monitor = {"ds_dt": 0, "dv_dt": 1}


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

    parameters = numpy.array([], dtype=numpy.float64)

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # s=2.0, v=1.0

    states = numpy.array([2.0, 1.0], dtype=numpy.float64)

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    ds_dt = v
    values[0] = ds_dt
    dv_dt = -s
    values[1] = dv_dt

    return values


def monitor_values(t, states, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters

    # Assign expressions
    shape = 2 if len(states.shape) == 1 else (2, states.shape[1])
    values = numpy.zeros(shape)
    ds_dt = v
    values[0] = ds_dt
    dv_dt = -s
    values[1] = dv_dt

    return values


def forward_explicit_euler(states, t, dt, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    ds_dt = v
    values[0] = ds_dt * dt + s
    dv_dt = -s
    values[1] = dt * dv_dt + v

    return values


def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    ds_dt = v
    values[0] = ds_dt * dt + s
    dv_dt = -s
    values[1] = dt * dv_dt + v

    return values

def theta_rule(states, t, dt, parameters, theta=0.5):
        # Assign states
    s = states[0]
    v = states[1]

    # Assign parameters

    # Assign expressions
    ds_dt = v
    dv_dt = -s

    A = numpy.array([[1, -dt * theta], [dt * theta, 1]])
    b = numpy.array([s + dt  * (1 - theta) * v, v - dt * (1 - theta) * s])

    values = numpy.linalg.solve(A, b)

    return values

