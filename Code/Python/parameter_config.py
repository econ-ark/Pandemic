'''Configurable parameters. This file is edited by run_config.py by putting the new
values at the bottom of the file, superseding the default value.
'''

import numpy as np

# Parameters concerning the distribution of discount factors
DiscFacMeanD = 0.9637   # Mean intertemporal discount factor for dropout types
DiscFacMeanH = 0.9705   # Mean intertemporal discount factor for high school types
DiscFacMeanC = 0.97557  # Mean intertemporal discount factor for college types
DiscFacSpread = 0.0253  # Half-width of uniform distribution of discount factors

# Parameters concerning what happens when the pandemic strikes
uPfac_L = 0.891         # Factor on marginal utility when the pandemic hits (lockdown)
Dspell_real = 3.0       # Actual average duration of deep unemployment spell, in quarters
Dspell_pcvd = 3.0       # Perceived expected duration of deep unemloyment spell, in quarters
Lspell_real = 2.0       # Actual average duration of the lockdown; if L_shared is True, this must be an integer
Lspell_pcvd = 2.0       # Perceived expected duration of the lockdown
L_shared = False        # Indicator for whether the decrease in marginal utility is shared (True) or idiosyncratic (False)
UnempD = -1.15          # Constant in unemployment shock logit for dropouts
UnempH = -1.30          # Constant in unemployment shock logit for high school grads
UnempC = -1.65          # Constant in unemployment shock logit for college grads
UnempP = -0.1           # Coefficient on log permanent income in unemployment shock logit
UnempA1 = -0.01         # Coefficient on age in unemployment shock logit
UnempA2 = 0.            # Coefficient on age squared in unemployment shock logit
DeepD = -1.50           # Constant in deep unemp shock logit for dropouts
DeepH = -1.75           # Constant in deep unemp shock logit for high school grads
DeepC = -2.20           # Constant in deep unemp shock logit for college grads
DeepP = -0.2            # Coefficient on log permanent income in deep unemp shock logit
DeepA1 = -0.01          # Coefficient on age in deep unemp shock logit
DeepA2 = 0.             # Coefficient on age squared in deep unemp shock logit
DeepPanAdj1 = -0.3      # Coefficient to be added to Unemp parameters for a deep unemployment pandemic
DeepPanAdj2 = 1.2       # Coefficient to be added to Deep parameters for a deep unemployment pandemic
DeepPanAdj3 = -0.1      # Coefficient to be added to UnempP parameter for a deep unemployment pandemic

# Parameters concerning the fiscal stimulus policy in response to the pandemic
T_ahead = 1             # Lag between pandemic and arrival of stimulus checks (quarters)
StimMax = 1.2           # Maximum "normal" stimulus check an individual can receive, in $1000
StimCut0 = 18.75        # Quarterly income threshold where stimulus begins to phase out (can be none)
StimCut1 = 24.75        # Quarterly income threshold where stimulus is fully phased out (can be none)
BonusUnemp = 5.2        # Additional payment to unemployed households when the pandemic hits
BonusDeep = 7.8         # Additional payment to deeply unemp households when the pandemic hits
UpdatePrb = 0.25        # Probability that a household anticipates stimulus check in each quarter before it arrives

# Basic model parameters: CRRA, growth factors, unemployment parameters (for normal times)
CRRA = 1.0              # Coefficient of relative risk aversion
PopGroFac = 1.01**0.25  # Population growth factor
PermGroFacAgg = 1.01**0.25 # Technological growth rate or aggregate productivity growth factor
Urate_normal = 0.05     # Unemployment rate in "normal", non-pandemic times
Uspell = 1.5            # Average duration of normal unemployment spell, in quarters
IncUnemp = 0.3          # Unemployment benefits replacement rate (proportion of permanent income)
IncUnempRet = 0.0       # "Unemployment benefit" when retired
UnempPrbRet = 0.0001    # Probability of "unemployment" when retired-- missing a SS check for some reason

# Parameters concerning the initial distribution of permanent income by education
pLvlInitMeanD = np.log(5.0) # Average quarterly permanent income of "newborn" HS dropout ($1000)
pLvlInitMeanH = np.log(7.5) # Average quarterly permanent income of "newborn" HS graduate ($1000)
pLvlInitMeanC = np.log(12.0)# Average quarterly permanent income of "newborn" HS  ($1000)
pLvlInitStd = 0.4           # Standard deviation of initial log permanent income (within education)
EducShares = [0.11, 0.55, 0.34] # Proportion of dropouts, HS grads, college types

# Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution

# Aggregation factor - what do we mulitply the mean individual by to get aggregate numbers for the US economy
# Resulting number is in billions of USD
AggregationFactor = 253.0
StimMax = 0.3 
