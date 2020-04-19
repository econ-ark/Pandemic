'''
This file has an extension of MarkovConsumerType that is used for the GiveItAwayNow project.
'''
import warnings
import numpy as np
from HARK.simulation import drawUniform, drawBernoulli
from HARK.distribution import DiscreteDistribution
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import MargValueFunc, ConsumerSolution
from HARK.interpolation import LinearInterp, LowerEnvelope
from HARK.core import distanceMetric
from Parameters import makeMrkvArray, T_sim
import matplotlib.pyplot as plt

# Define a modified MarkovConsumerType
class GiveItAwayNowType(MarkovConsumerType):
    time_inv_ = MarkovConsumerType.time_inv_ + ['uPfac']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        MarkovConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.solveOnePeriod = solveConsMarkovALT
        
        
    def preSolve(self):
        self.MrkvArray = self.MrkvArray_pcvd
        MarkovConsumerType.preSolve(self)
        self.updateSolutionTerminal()
        
        
    def initializeSim(self):
        MarkovConsumerType.initializeSim(self)
        if hasattr(self,'T_advance'):
            self.restoreState()
            self.MrkvArray = self.MrkvArray_sim
        elif not hasattr(self,'mortality_off'):
            self.calcAgeDistribution()
            self.initializeAges()
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow[:] = self.Mrkv_univ
        
        
    def getMortality(self):
        '''
        A modified version of getMortality that reads mortality history if the
        attribute read_mortality exists.  This is a workaround to make sure the
        history of death events is identical across simulations.
        '''
        if (self.read_shocks or hasattr(self,'read_mortality')):
            who_dies = self.who_dies_backup[self.t_sim,:]
        else:
            who_dies = self.simDeath()
        self.simBirth(who_dies)
        self.who_dies = who_dies
        return None
    
    
    def simDeath(self):
        if hasattr(self,'mortality_off'):
            return np.zeros(self.AgentCount, dtype=bool)
        else:
            return MarkovConsumerType.simDeath(self)
        
        
    def getStates(self):
        MarkovConsumerType.getStates(self)
        if hasattr(self,'T_advance'): # This means we're in the policy experiment
            self.noticeStimulus()
            self.makeWeights()
        if hasattr(self,'ContUnempBenefits'):
            if self.ContUnempBenefits==True:
                self.continueUnemploymentBenefits()
        
        # Store indicators of whether this agent is a worker and unemployed
        w = self.t_cycle <= self.T_retire
        u = np.logical_and(np.mod(self.MrkvNow,3) > 0, w)
        lLvl = self.pLvlNow*self.TranShkNow
        lLvl[u] = 0.
        lLvl[self.t_cycle > self.T_retire] = 0.
        self.lLvlNow = lLvl
        self.uNow = u
        self.wNow = w
        
        
    def updateMrkvArray(self):
        '''
        Constructs an updated MrkvArray_pcvd attribute to be used in solution (perceived),
        as well as MrkvArray_sim attribute to be used in simulation (actual).
        Uses the primitive attributes Uspell, Urate, Dspell_pcvd, Lspell_pcvd,
        Dspell_real, Lspell_real.
        '''
        self.MrkvArray_pcvd = makeMrkvArray(self.Urate, self.Uspell, self.Dspell_pcvd, self.Lspell_pcvd)
        self.MrkvArray_sim = makeMrkvArray(self.Urate, self.Uspell, self.Dspell_real, self.Lspell_real)

        
    def calcAgeDistribution(self):
        '''
        Calculates the long run distribution of t_cycle in the population.
        '''
        AgeMarkov = np.zeros((self.T_cycle+1,self.T_cycle+1))
        for t in range(self.T_cycle):
            p = self.LivPrb[t][0]
            AgeMarkov[t,t+1] = p
            AgeMarkov[t,0] = 1. - p
        AgeMarkov[-1,0] = 1.
        
        AgeMarkovT = np.transpose(AgeMarkov)
        vals, vecs = np.linalg.eig(AgeMarkovT)
        dist = np.abs(np.abs(vals) - 1.)
        idx = np.argmin(dist)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore warning about casting complex eigenvector to float
            LRagePrbs = vecs[:,idx].astype(float)
        LRagePrbs /= np.sum(LRagePrbs)
        age_vec = np.arange(self.T_cycle+1).astype(int)
        self.LRageDstn = DiscreteDistribution(LRagePrbs, age_vec)
        
        
    def initializeAges(self):
        '''
        Assign initial values of t_cycle to simulated agents, using the attribute
        LRageDstn as the distribution of discrete ages.
        '''
        age = self.LRageDstn.drawDiscrete(self.AgentCount,
                           seed=self.RNG.randint(0,2**31-1))
        age = age.astype(int)
        self.t_cycle = age
        self.t_age = age
        
    
    def switchToCounterfactualMode(self):
        '''
        Very small method that swaps in the "big" six-Markov-state versions of some
        solution attributes, replacing the "small" two-state versions that are used
        only to generate the pre-pandemic initial distbution of state variables.
        It then prepares this type to create alternate shock histories so it can
        run counterfactual experiments.
        '''
        del self.solution
        self.delFromTimeVary('solution')
        
        # Swap in "big" versions of the Markov-state-varying attributes
        self.LivPrb = self.LivPrb_big
        self.PermGroFac = self.PermGroFac_big
        self.MrkvArray = self.MrkvArray_big
        self.Rfree = self.Rfree_big
        self.uPfac = self.uPfac_big
        self.IncomeDstn = self.IncomeDstn_big
        
        # Adjust simulation parameters for the counterfactual experiments
        self.T_sim = T_sim
        self.track_vars = ['cNrmNow','pLvlNow','Weight','lLvlNow','uNow','wNow','TranShkNow','t_cycle']
        self.T_advance = None
        self.MrkvArray_pcvd = self.MrkvArray
        #print('Finished type ' + str(self.seed) + '!')
        
        
    def makeAlternateShockHistories(self):
        '''
        Make a history of Markov states and income shocks starting from each Markov state.
        '''
        self.MrkvArray = self.MrkvArray_sim
        J = self.MrkvArray[0].shape[0]
        DeathHist = np.zeros((J,self.T_sim,self.AgentCount), dtype=bool)
        MrkvHist = np.zeros((J,self.T_sim,self.AgentCount), dtype=int)
        TranShkHist = np.zeros((J,self.T_sim,self.AgentCount))
        PermShkHist = np.zeros((J,self.T_sim,self.AgentCount))
        for j in range(6):
            self.Mrkv_univ = j
            self.read_shocks = False
            self.makeShockHistory()
            DeathHist[j,:,:] = self.who_dies_hist
            MrkvHist[j,:,:] = self.MrkvNow_hist
            PermShkHist[j,:,:] = self.PermShkNow_hist
            TranShkHist[j,:,:] = self.TranShkNow_hist
            self.read_mortality = True # Make sure that every death history is the same
            self.who_dies_backup = self.who_dies_hist.copy()
        self.DeathHistAll = DeathHist
        self.MrkvHistAll = MrkvHist
        self.PermShkHistAll = PermShkHist
        self.TranShkHistAll = TranShkHist
        self.Mrkv_univ = None
        self.MrkvArray_sim_prev = self.MrkvArray_sim
        self.L_shared_prev = self.L_shared
        del(self.read_mortality)
        
        
    def solveIfChanged(self):
        '''
        Re-solve the lifecycle model only if the attributes MrkvArray_pcvd and uPfac
        do not match those in MrkvArray_pcvd_prev and uPfac_prev.
        '''
        # Check whether MrkvArray_pcvd and uPfac have changed (and whether they exist at all!)
        try:
            same_MrkvArray = distanceMetric(self.MrkvArray_pcvd, self.MrkvArray_pcvd_prev) == 0.
            same_uPfac = distanceMetric(self.uPfac, self.uPfac_prev) == 0.
            if (same_MrkvArray and same_uPfac):
                return
        except:
            pass
        
        # Re-solve the model, then note the values in MrkvArray_pcvd and uPfac
        self.solve()
        self.MrkvArray_pcvd_prev = self.MrkvArray_pcvd
        self.uPfac_prev = self.uPfac
        
        
    def makeShocksIfChanged(self):
        '''
        Re-draw the histories of Markov states and income shocks only if the attributes
        MrkvArray_sim and L_shared do not match those in MrkvArray_sim_prev and L_shared_prev.
        '''
        # Check whether MrkvArray_sim and L_shared have changed (and whether they exist at all!)
        try:
            same_MrkvArray = distanceMetric(self.MrkvArray_sim, self.MrkvArray_sim_prev) == 0.
            same_shared = self.L_shared == self.L_shared_prev
            if (same_MrkvArray and same_shared):
                return
        except:
            pass
        
        # Re-draw the shock histories, then note the values in MrkvArray_sim and L_shared
        self.makeAlternateShockHistories()


    def makeWeights(self):
        '''
        Create the attribute Weight.
        '''
        self.Weight = self.PopGroFac**((self.t_sim+self.t_sim_base)-self.t_cycle)
    
    
    def saveState(self):
        '''
        Record the current state of simulation variables for later use.
        '''
        self.aNrm_base = self.aNrmNow.copy()
        self.pLvl_base = self.pLvlNow.copy()
        self.Mrkv_base = self.MrkvNow.copy()
        self.age_base  = self.t_cycle.copy()
        self.t_sim_base = self.t_sim
        self.PlvlAgg_base = self.PlvlAggNow


    def restoreState(self):
        '''
        Restore the state of the simulation to some baseline values.
        '''
        self.aNrmNow = self.aNrm_base.copy()
        self.pLvlNow = self.pLvl_base.copy()
        self.MrkvNow = self.Mrkv_base.copy()
        self.t_cycle = self.age_base.copy()
        self.t_age   = self.age_base.copy()
        self.PlvlAggNow = self.PlvlAgg_base
        
        
    def hitWithPandemicShock(self):
        '''
        Alter the Markov state of each simulated agent, jumping some people into
        an otherwise inaccessible "deep unemployment" state, and others into
        normal unemployment.
        '''
        # Calculate (cumulative) probabilities of each agent being shocked into each state
        age = (self.t_age/4) + 24
        DeepX = self.DeepParam0 + self.DeepParam1*np.log(self.pLvlNow) + self.DeepParam2*age + self.DeepParam3*age**2
        UnempX = self.UnempParam0 + self.UnempParam1*np.log(self.pLvlNow) + self.UnempParam2*age + self.UnempParam3*age**2
        expDeepX = np.exp(DeepX)
        expUnempX = np.exp(UnempX)
        denom = 1. + expDeepX + expUnempX
        EmpPrb = 1./denom
        UnempPrb = expUnempX/denom
        DeepPrb = expDeepX/denom
        PrbArray = np.vstack([EmpPrb,UnempPrb,DeepPrb])
        CumPrbArray = np.cumsum(PrbArray, axis=0)
        
        # Draw new Markov states for each agent
        draws = drawUniform(self.AgentCount, seed=self.RNG.randint(0,2**31-1))
        draws = self.RNG.permutation(draws)
        MrkvNew = np.zeros(self.AgentCount, dtype=int)
        MrkvNew[draws > CumPrbArray[0]] = 1
        MrkvNew[draws > CumPrbArray[1]] = 2
        if (self.PanShock and not self.L_shared): # If the pandemic actually occurs,
            MrkvNew += 3 # then put everyone into the low marginal utility world/
            # This is (momentarily) skipped over if the lockdown state is shared
            # rather than idiosyncratic.  See a few lines below.
        
        # Move agents to those Markov states 
        self.MrkvNow = MrkvNew
        
        # Take the appropriate shock history for each agent, depending on their state
        J = self.MrkvArray[0].shape[0]
        for j in range(J):
            these = self.MrkvNow == j
            self.who_dies_hist[:,these] = self.DeathHistAll[j,:,:][:,these]
            self.MrkvNow_hist[:,these] = self.MrkvHistAll[j,:,:][:,these]
            self.PermShkNow_hist[:,these] = self.PermShkHistAll[j,:,:][:,these]
            self.TranShkNow_hist[:,these] = self.TranShkHistAll[j,:,:][:,these]
        
        # If the lockdown is a common/shared event, rather than idiosyncratic, bump
        # everyone into the lockdown state for *exactly* T_lockdown periods
        if (self.PanShock and self.L_shared):
            T = self.T_lockdown
            self.MrkvNow_hist[0:T,:] += 3
            
        # Edit the first period of the shock history to give all unemployed
        # people a bonus payment in just that quarter
        one_off_benefits = True   # If agents get continued unemployment benefits, the first period benefits are counted later
        if hasattr(self,'ContUnempBenefits'):
            if self.ContUnempBenefits==True:
                one_off_benefits = False
        if one_off_benefits:
            young = self.age_base < self.T_retire
            unemp = np.logical_and(np.mod(self.MrkvNow,3) == 1, young)
            deep  = np.logical_and(np.mod(self.MrkvNow,3) == 2, young)
            self.TranShkNow_hist[0,unemp] += self.BonusUnemp/(self.pLvlNow[unemp]*self.PermShkNow_hist[0,unemp])
            self.TranShkNow_hist[0,deep]  += self.BonusDeep/(self.pLvlNow[deep]*self.PermShkNow_hist[0,deep])
            
        
    def announceStimulus(self):
        '''
        Announce a stimulus payment T periods in advance of when it will actually occur.
        '''
        self.T_til_check = self.T_advance
        self.Stim_unnoticed = np.ones(self.AgentCount, dtype=bool)
        
        # Determine stimulus check size for each simulated agent
        StimLvl = np.ones(self.AgentCount)*self.StimMax
        if self.StimCut1 is not None:
            these = self.pLvl_base > self.StimCut1
            StimLvl[these] = 0. # Eliminate stimulus check for those above top threshold
        if self.StimCut0 is not None:
            these = np.logical_and(self.pLvl_base > self.StimCut0, self.pLvl_base <= self.StimCut1)
            alpha = (self.pLvl_base[these] - self.StimCut0) / (self.StimCut1 - self.StimCut0)
            StimLvl[these] *= 1.-alpha # Phase out stimulus check for those above bottom threshold
        self.StimLvl = StimLvl
        
    
    def noticeStimulus(self):
        '''
        Give each agent the opportunity to notice the future stimulus payment and
        mentally account for it in their market resources.
        '''
        if self.T_til_check > 0:
            self.T_til_check -= 1
        
        updaters = drawBernoulli(self.AgentCount, p=self.UpdatePrb, seed=self.RNG.randint(0,2**31-1))
        if self.T_til_check == 0:
            updaters = np.ones(self.AgentCount, dtype=bool)
        
        self.mNrmNow[updaters] += self.Stim_unnoticed[updaters]*self.StimLvl[updaters]/self.pLvlNow[updaters]*self.Rfree[0]**(-self.T_til_check)
        self.Stim_unnoticed[updaters] = False
        
    def continueUnemploymentBenefits(self):
        '''
        Continue to give unemployment benefits if utility of consumption remains depressed
        '''
        young = self.t_cycle < self.T_retire
        unemp = np.logical_and(self.MrkvNow == 4, young)
        deep  = np.logical_and(self.MrkvNow == 5, young)
        self.mNrmNow[unemp] += self.BonusUnemp/(self.pLvlNow[unemp]*self.PermShkNow_hist[0,unemp])
        self.mNrmNow[deep] += self.BonusDeep/(self.pLvlNow[deep]*self.PermShkNow_hist[0,deep])
        
        
def solveConsMarkovALT(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,uPfac,
                                 MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncomeDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : DiscreteDistribution
        A representation of permanent and transitory income shocks that might
        arrive at the beginning of next period.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroFac : np.array
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    uPfac : np.array
        Scaling factor for (marginal) utility in each current Markov state.
    MrkvArray : np.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Not used.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.  Not used.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin.  All of these attributes are lists or arrays, 
        with elements corresponding to the current Markov state.  E.g.
        solution.cFunc[0] is the consumption function when in the i=0 Markov
        state this period.
    '''
    # Get sizes of grids
    aCount = aXtraGrid.size
    StateCount = MrkvArray.shape[0]

    # Loop through next period's states, assuming we reach each one at a time.
    # Construct EndOfPrdvP_cond functions for each state.
    BoroCnstNat_cond = []
    EndOfPrdvPfunc_cond = []
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j]
        mNrmMinNext = solution_next.mNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncomeDstn[j].pmf
        PermShkValsNext = IncomeDstn[j].X[0]
        TranShkValsNext = IncomeDstn[j].X[1]
        ShkCount = ShkPrbsNext.size
        aXtra_tiled = np.tile(np.reshape(aXtraGrid, (aCount, 1)), (1, ShkCount))

        # Make tiled versions of the income shocks
        # Dimension order: aNow, Shk
        ShkPrbsNext_tiled = np.tile(np.reshape(ShkPrbsNext, (1, ShkCount)), (aCount, 1))
        PermShkValsNext_tiled = np.tile(np.reshape(PermShkValsNext, (1, ShkCount)), (aCount, 1))
        TranShkValsNext_tiled = np.tile(np.reshape(TranShkValsNext, (1, ShkCount)), (aCount, 1))

        # Find the natural borrowing constraint
        aNrmMin_candidates = PermGroFac[j]*PermShkValsNext_tiled/Rfree[j]*(mNrmMinNext - TranShkValsNext_tiled[0, :])
        aNrmMin = np.max(aNrmMin_candidates)
        BoroCnstNat_cond.append(aNrmMin)

        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        aNrmNow_tiled = aNrmMin + aXtra_tiled
        mNrmNext_array = Rfree[j]*aNrmNow_tiled/PermShkValsNext_tiled + TranShkValsNext_tiled

        # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
        vPnext_array = Rfree[j]*PermShkValsNext_tiled**(-CRRA)*vPfuncNext(mNrmNext_array)

        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*np.sum(vPnext_array*ShkPrbsNext_tiled, axis=1)

        # Make the conditional end-of-period marginal value function
        EndOfPrdvPnvrs = EndOfPrdvP**(-1./CRRA)
        EndOfPrdvPnvrsFunc = LinearInterp(np.insert(aNrmMin + aXtraGrid, 0, aNrmMin), np.insert(EndOfPrdvPnvrs, 0, 0.0))
        EndOfPrdvPfunc_cond.append(MargValueFunc(EndOfPrdvPnvrsFunc, CRRA))

    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.
    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        # Find natural borrowing constraint for this state
        aNrmMin_candidates = np.zeros(StateCount) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:  # Irrelevant if transition is impossible
                aNrmMin_candidates[j] = BoroCnstNat_cond[j]
        aNrmMin = np.nanmax(aNrmMin_candidates)
        
        # Find the minimum allowable market resources
        if BoroCnstArt is not None:
            mNrmMin = np.maximum(BoroCnstArt, aNrmMin)
        else:
            mNrmMin = aNrmMin
        mNrmMinNow.append(mNrmMin)

        # Make tiled grid of aNrm
        aNrmNow = aNrmMin + aXtraGrid
        
        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP = np.zeros(aCount)
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:
                temp = MrkvArray[i, j]*EndOfPrdvPfunc_cond[j](aNrmNow)
                EndOfPrdvP += temp
        EndOfPrdvP *= LivPrb[i] # Account for survival out of the current state

        # Calculate consumption and the endogenous mNrm gridpoints for this state
        cNrmNow = (EndOfPrdvP/uPfac[i])**(-1./CRRA)
        mNrmNow = aNrmNow + cNrmNow

        # Make a piecewise linear consumption function
        c_temp = np.insert(cNrmNow, 0, 0.0)  # Add point at bottom
        m_temp = np.insert(mNrmNow, 0, aNrmMin)
        cFuncUnc = LinearInterp(m_temp, c_temp)
        cFuncCnst = LinearInterp(np.array([mNrmMin, mNrmMin+1.0]), np.array([0.0, 1.0]))
        cFuncNow.append(LowerEnvelope(cFuncUnc,cFuncCnst))

        # Construct the marginal value function using the envelope condition
        m_temp = aXtraGrid + mNrmMin
        c_temp = cFuncNow[i](m_temp)
        uP = uPfac[i]*c_temp**(-CRRA)
        vPnvrs = uP**(-1./CRRA)
        vPnvrsFunc = LinearInterp(np.insert(m_temp, 0, mNrmMin), np.insert(vPnvrs, 0, 0.0))
        vPfuncNow.append(MargValueFunc(vPnvrsFunc, CRRA))
        
    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)
    return solution_now
