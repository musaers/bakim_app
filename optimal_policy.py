"""
Optimal Maintenance Policy Calculation Module
This module performs optimal maintenance policy calculations for multi-component systems.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def StateIndex(StateDescV, Dim, SetSize):
    """
    Convert state description vector to linear index.
    
    Arguments:
        StateDescV: State description vector
        Dim: Dimension (number of components)
        SetSize: Set size (K+1)
    
    Returns:
        int: Linear index
    """
    Index = 0
    for i in range(Dim):
        Index += StateDescV[i]*((SetSize)**i)
    return int(Index)

def StateDesc(Ind, Dim, SetSize):
    """
    Convert linear index to state description vector.
    
    Arguments:
        Ind: Linear index
        Dim: Dimension (number of components)
        SetSize: Set size (K+1)
    
    Returns:
        numpy.array: State description vector
    """
    StateDescV = np.zeros(Dim, dtype=int)
    for i in range(Dim):
        StateDescV[i] = np.remainder(Ind, SetSize)
        Ind = np.floor_divide(Ind, SetSize)
    return StateDescV

def PreProcessing(alpha, K, C, Eps):
    """
    Perform preprocessing required before optimal policy calculation.
    This follows the original notebook logic exactly.
    
    Arguments:
        alpha: Probability of no degradation
        K: Maximum deterioration level
        C: Number of components
        Eps: Precision threshold
    
    Returns:
        tuple: (NumberNonGreenV, DistributionV, ObsTransM, U)
    """
    DetSSize = K+1
    ProbReachingT = 1
    
    StateSpacesSize = DetSSize**C  # State space size for joint distribution
    TransitionM = np.zeros([StateSpacesSize, StateSpacesSize])
    DistributionV = np.zeros([1, StateSpacesSize])
    CondDistV = np.zeros(StateSpacesSize)
    ObsTransM = np.zeros([1, 2])  # Transition probabilities from Yellow states at time t to Yellow or Red states
    
    NotRedMaskV = np.zeros(StateSpacesSize, dtype=int)
    NumberNonGreenV = np.zeros(StateSpacesSize, dtype=int)
    StateDescVector = np.zeros(C)
    
    # Create list of Red state indices and mask vector for non-red states
    RedList = []
    for Ind in range(StateSpacesSize):
        StateV = StateDesc(Ind, C, DetSSize)
        
        # Detect Red states
        RedState = 0
        for i in range(C):
            if StateV[i] == K:
                RedState = 1
                break
        
        if RedState == 0:
            NotRedMaskV[Ind] = 1
        else:
            RedList.append(Ind)
        
        # Count non-Green components
        NumNonGreen = 0
        for i in range(C):
            if StateV[i] != 0:
                NumNonGreen += 1
        
        NumberNonGreenV[Ind] = NumNonGreen
    
    # Build transition matrix
    for FromInd in range(StateSpacesSize):
        FromStateV = StateDesc(FromInd, C, DetSSize)
        
        if FromInd in RedList:
            TransitionM[FromInd, FromInd] = 1
        else:
            for IncInd in range(2**C):
                IncV = StateDesc(IncInd, C, 2)
                
                ToStateV = FromStateV + IncV
                ToStateInd = StateIndex(ToStateV, C, DetSSize)
                TransitionM[FromInd, ToStateInd] = (1-alpha)**(sum(IncV))*(alpha**(C-sum(IncV)))
    
    t = 0
    DistributionV[t, :] = TransitionM[0, :]
    CondDistV = 1/(1-DistributionV[0, 0])*DistributionV[t, :]
    CondDistV[0] = 0
    
    ObsTransM[0, :] = [1, 0]
    
    DistributionV[t, :] = CondDistV
    
    # Continue until precision threshold is met (original logic)
    while ProbReachingT > Eps:
        t += 1
        DistributionV = np.append(DistributionV, [np.matmul(CondDistV, TransitionM)], axis=0)
        CondDistV = np.multiply(DistributionV[t, :], NotRedMaskV)
        SumNotRed = np.sum(CondDistV)
        ObsTransM = np.append(ObsTransM, [[SumNotRed, 1-SumNotRed]], axis=0)
        ProbReachingT *= ObsTransM[t, 0]
        CondDistV = 1/SumNotRed*CondDistV
    
    return NumberNonGreenV, DistributionV, ObsTransM, t+1

def ReliabilityLP(Code, K, C, U, alpha, cp, cc, cr, cu, co, ct, NumberNonGreenV, DistributionV, ObsTransM, yellow_threshold=None):
    """
    Solve linear programming model for maintenance policy optimization.
    Modified to respect yellow_threshold constraint while maintaining original logic.
    
    Arguments:
        Code: Policy code (0 for optimal policy)
        K: Maximum deterioration level
        C: Number of components
        U: Time cutoff point (from preprocessing)
        alpha: Probability of no degradation
        cp: Preventive maintenance cost
        cc: Corrective maintenance cost
        cr: Spare part replacement cost
        cu: Shortage cost
        co: Excess cost
        ct: Part transfer cost
        NumberNonGreenV: Non-green component count vector
        DistributionV: Distribution vector
        ObsTransM: Transition matrix
        yellow_threshold: User's yellow signal threshold (optional)
    
    Returns:
        tuple: (SolutionMat, ObjValue)
    """
    DetSSize = K+1
    CounterStuckP = alpha**C
    StateSpacesSize = DetSSize**C  
    
    PeriodCosts = np.zeros((2, U+1, C+1))
    PeriodCosts[0, :, 1:C+1] = cp
    PeriodCosts[1, :, :] = cc
    
    for s in range(2):
        for n in range(0, U+1):  
            for a in range(1, C+1):
                PeriodCosts[s, n, a] += ct*a
                
                if n == 0:
                    PeriodCosts[s, n, a] += (cr*NumberNonGreenV[0] + 
                                           cu*max(NumberNonGreenV[0]-a, 0) + 
                                           co*max(-NumberNonGreenV[0]+a, 0))
                else:
                    for Ind in range(StateSpacesSize):
                        PeriodCosts[s, n, a] += (cr*NumberNonGreenV[Ind] + 
                                               cu*max(NumberNonGreenV[Ind]-a, 0) + 
                                               co*max(-NumberNonGreenV[Ind]+a, 0))*DistributionV[n-1, Ind]
    
    # Create model
    lp = gp.Model("ReliabilityMDP")
    lp.setParam("FeasibilityTol", 1e-9)
    lp.setParam("NumericFocus", 2)
    
    # Variables to optimize
    P = lp.addMVar(shape=(2, U+1, C+1), name="P", obj=PeriodCosts)
    
    # Constraint 0: No intervention at state 0
    lp.addConstr(
        gp.quicksum(P[0, 0, 1:C+1]) == 0
    )
    
    # Constraint 1
    lp.addConstr(
        gp.quicksum(P[0, 0, :]) - 
        gp.quicksum(P[s, n, a] 
            for s in range(2)
            for n in range(U+1)
            for a in range(1, C+1)
        ) - 
        CounterStuckP*P[0, 0, 0] == 0
    )
    
    # Constraint 2
    lp.addConstr(
        gp.quicksum(P[0, 1, :]) - 
        (1-CounterStuckP)*P[0, 0, 0] == 0
    )
    
    # Constraints 3 and 4
    for n in range(2, U+1):
        lp.addConstr(
            gp.quicksum(P[0, n, :]) - 
            ObsTransM[n-1, 0]*P[0, n-1, 0] == 0
        )
        
        lp.addConstr(
            gp.quicksum(P[1, n, :]) - 
            ObsTransM[n-1, 1]*P[0, n-1, 0] == 0
        )
    
    # Constraint 5: Total probability equals 1
    lp.addConstr(
        P.sum() == 1
    )
    
    # Constraint: Mandatory intervention in Red state
    for n in range(U+1):
        lp.addConstr(
            P[1, n, 0] == 0
        )
    
    # Constraint: Cannot reach Red state before K
    for n in range(K):
        for a in range(C+1):
            lp.addConstr(
                P[1, n, a] == 0
            )
    
    # Constraint: Mandatory intervention at cutoff counter (original logic)
    lp.addConstr(
        P[0, U, 0] == 0
    )
    
    # NEW: Yellow threshold constraint - prohibit interventions beyond user's threshold
    if yellow_threshold is not None:
        for n in range(yellow_threshold + 1, U + 1):
            lp.addConstr(
                P[0, n, 0] == 0  # Force intervention at or before yellow_threshold
            )
        
        # If yellow_threshold < U, force intervention at yellow_threshold
        if yellow_threshold < U:
            lp.addConstr(
                P[0, yellow_threshold, 0] == 0
            )
    
    # Additional constraints based on different policy codes
    if Code == 1:  # Intervention at K-1 with optimal number of parts
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
    
    elif Code == 2:  # Intervention at K-1 with only 1 part
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
        for a in range(2, C+1):
            lp.addConstr(P[0, K-1, a] == 0)
    
    elif Code == 3:  # Intervention at K-1 with C parts
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
        for a in range(1, C):
            lp.addConstr(P[0, K-1, a] == 0)
    
    elif Code == 4:  # Only intervention in Red with optimal number of parts
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
    
    elif Code == 5:  # Only intervention in Red with 1 part
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        for n in range(U+1):
            for a in range(2, C+1):
                lp.addConstr(P[1, n, a] == 0)
    
    elif Code == 6:  # Only intervention in Red with C parts
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        for n in range(U+1):
            for a in range(1, C):
                lp.addConstr(P[1, n, a] == 0)
    
    # Solve model
    lp.optimize()
    
    # Get solution
    SolutionMat = P.X
    ObjValue = lp.getObjective().getValue()
    
    return SolutionMat, ObjValue

def calculate_optimal_policy(params):
    """
    Calculate optimal maintenance policy given parameters.
    
    Arguments:
        params: Parameter dictionary (C, K, alpha, cp, cc, cr, cu, co, ct, yellow_threshold)
    
    Returns:
        dict: Optimal policy information
    """
    try:
        # Extract parameters
        C = params['C']
        K = params['K']
        alpha = params['alpha']
        cp = params['c1']
        cc = params['c2']
        cr = params['cr']
        cu = params['cs']
        co = params['ce']
        ct = params['ct']
        yellow_threshold = params.get('yellow_threshold', None)
        
        # Precision threshold
        Eps = 0.01
        
        # Preprocessing (original logic - no modification)
        NumberNonGreenV, DistributionV, ObsTransM, U = PreProcessing(alpha, K, C, Eps)
        
        # Calculate optimal policy with yellow_threshold constraint
        SolutionMat, ObjValue = ReliabilityLP(0, K, C, U, alpha, cp, cc, cr, cu, co, ct, 
                                            NumberNonGreenV, DistributionV, ObsTransM, yellow_threshold)
        
        # Extract policy information
        policy_info = extract_policy_info(SolutionMat, C, U, ObjValue, NumberNonGreenV, DistributionV, yellow_threshold)
        
        return {
            'success': True,
            'policy': policy_info,
            'objective_value': ObjValue,
            'solution_matrix': SolutionMat.tolist()  # Convert numpy array to list
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def extract_policy_info(SolutionMat, C, U, objective, NumberNonGreenV, DistributionV, yellow_threshold=None):
    """
    Extract policy information from solution matrix.
    
    Arguments:
        SolutionMat: Solution matrix
        C: Number of components
        U: Time cutoff point
        objective: Objective function value
        NumberNonGreenV: Non-green component count vector
        DistributionV: Distribution vector
        yellow_threshold: User's yellow threshold
    
    Returns:
        dict: Policy information
    """
    # Find which states require intervention
    yellow_interventions = []
    red_interventions = []
    
    # For yellow states - only consider up to yellow_threshold if specified
    max_counter = yellow_threshold if yellow_threshold is not None else U
    
    for n in range(min(max_counter + 1, U + 1)):
        for a in range(1, C+1):
            if SolutionMat[0, n, a] > 0.001:  # Tolerance for small numerical errors
                yellow_interventions.append({
                    'counter': int(n),
                    'components': int(a),
                    'probability': float(SolutionMat[0, n, a])
                })
    
    # For red states
    for n in range(U+1):
        for a in range(1, C+1):
            if SolutionMat[1, n, a] > 0.001:
                red_interventions.append({
                    'counter': int(n),
                    'components': int(a),
                    'probability': float(SolutionMat[1, n, a])
                })
    
    # System's probability of entering red state
    down_probability = float(np.sum(SolutionMat[1, :, :]))
    
    # System's probability of preventive maintenance
    preventive_probability = float(np.sum(SolutionMat[0, :, 1:C+1]))
    
    # Summary of optimal policy
    if len(yellow_interventions) == 1:
        policy_description = f"Optimal maintenance policy: Perform preventive maintenance with {yellow_interventions[0]['components']} components when yellow signal counter reaches {yellow_interventions[0]['counter']}. Perform corrective maintenance on red signal."
    elif len(yellow_interventions) > 1:
        policy_description = "Mixed maintenance policy: Preventive maintenance should be performed in multiple scenarios."
    elif len(red_interventions) > 0:
        policy_description = "Only corrective maintenance should be performed (on red signal)"
    else:
        policy_description = "No valid maintenance policy found."
    
    # Add constraint information if yellow_threshold was applied
    if yellow_threshold is not None:
        policy_description += f" (Constrained to yellow threshold: {yellow_threshold})"
    
    return {
        'objective_value': float(objective),
        'yellow_interventions': yellow_interventions,
        'red_interventions': red_interventions,
        'down_probability': down_probability,
        'preventive_probability': preventive_probability,
        'policy_description': policy_description
    }