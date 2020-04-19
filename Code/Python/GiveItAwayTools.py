'''
This file has major functions that are used by GiveItAwayMAIN.py
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from HARK.utilities import getPercentiles, getLorenzShares
from HARK import multiThreadCommands, multiThreadCommandsFake
from Parameters import EducShares, figs_dir

mystr = lambda x : '{:.2f}'.format(x)
mystr2 = lambda x : '{:.3f}'.format(x)

def runExperiment(Agents,PanShock,
                  StimMax,StimCut0,StimCut1,BonusUnemp,BonusDeep,T_ahead,
                  UnempD,UnempH,UnempC,UnempP,UnempA1,UnempA2,
                  DeepD,DeepH,DeepC,DeepP,DeepA1,DeepA2,
                  Dspell_pcvd,Dspell_real,Lspell_pcvd,Lspell_real,L_shared):
    '''
    Conduct a fiscal policy experiment by announcing a fiscal stimulus T periods
    ahead of when checks will actually arrive.  The stimulus is in response to a
    global health crisis that shocks many consumers into unemployment, and others
    into "deep" unemployment that they think will last several quarters.
    
    Parameters
    ----------
    Agents : [AgentType]
        List of agent types in the economy.
    PanShock : bool
        Indicator for whether the pandemic actually hits.
    StimMax : float
        Maximum stimulus check a household can receive, denominated in $1000.
    StimCut0 : float or None
        Permanent income threshold where stimulus check begins to phase out.
        Can only be None if StimCut1 is also None.
    StimCut1 : float or None
        Permanent income threshold where stimulus check is completely phased out.
        None means that the same stimulus check is given to everyone.
    BonusUnemp : float
        One time "bonus benefit" given to regular unemployed people at t=0.
    BonusDeep : float
        One time "bonus benefit" given to deeply unemployed people at t=0.
    T_ahead : int
        Number of quarters after announcement that the stimulus checks will arrive.
    UnempD : float
        Constant for highschool dropouts in the Markov-shock logit for unemployment.
    UnempH : float
        Constant for highschool grads in the Markov-shock logit for unemployment.
    UnempC : float
        Constant for college goers in the Markov-shock logit for unemployment.
    UnempP : float
        Coefficient on log permanent income in the Markov-shock logit for unemploment.
    UnempA1 : float
        Coefficient on age in the Markov-shock logit for unemployment.
    UnempA2 : float
        Coefficient on age squared in the Markov-shock logit for unemployment.
    DeepD : float
        Constant for highschool dropouts in the Markov-shock logit for deep unemployment.
    DeepH : float
        Constant for highschool grads in the Markov-shock logit for deep unemployment.
    DeepC : float
        Constant for college goers in the Markov-shock logit for deep unemployment.
    DeepP : float
        Coefficient on log permanent income in the Markov-shock logit for deep unemploment.
    DeepA1 : float
        Coefficient on age in the Markov-shock logit for deep unemployment.
    DeepA2 : float
        Coefficient on age squared in the Markov-shock logit for deep unemployment.
    Dspell_pcvd : float
        Perceived average duration of a "deep unemployment" spell.
    Dspell_real : float
        Actual average duration of a "deep unemployment" spell.
    Lspell_pcvd : float
        Perceived average duration of the marginal utility-reducing lockdown.
    Lspell_real : float
        Actual average duration of the marginal utility-reducing lockdown.  If
        L_shared is True, it represents that *exact* duration of the lockdown,
        which should be an integer.
    L_shared : bool
        Indicator for whether the "lockdown" being lifted is a common/shared event
        across all agents (True) versus whether it's an idiosyncratic shock (False).
        
    Returns
    -------
    TBD
    '''
    PlvlAgg_adjuster = Agents[0].PlvlAgg_base
    T = Agents[0].T_sim
    
    # Adjust fiscal stimulus parameters by the level of aggregate productivity,
    # which is 96 years more advanced than you would expect because reasons.
    # Multiply unemployment benefits by 0.8 to reflect fact that labor force participation rate is 0.8.
    StimMax *= PlvlAgg_adjuster
    BonusUnemp *= PlvlAgg_adjuster * 0.8
    BonusDeep *= PlvlAgg_adjuster * 0.8
    if StimCut0 is not None:
        StimCut0 *= PlvlAgg_adjuster
    if StimCut1 is not None:
        StimCut1 *= PlvlAgg_adjuster
    
    # Make dictionaries of parameters to give to the agents
    experiment_dict_D = {
            'PanShock' : PanShock,
            'T_advance': T_ahead+1,
            'StimMax'  : StimMax,
            'StimCut0' : StimCut0,
            'StimCut1' : StimCut1,
            'BonusUnemp'  : BonusUnemp,
            'BonusDeep'   : BonusDeep,
            'UnempParam0' : UnempD,
            'UnempParam1' : UnempP,
            'UnempParam2' : UnempA1,
            'UnempParam3' : UnempA2,
            'DeepParam0'  : DeepD,
            'DeepParam1'  : DeepP,
            'DeepParam2'  : DeepA1,
            'DeepParam3'  : DeepA2,
            'Dspell_pcvd' : Dspell_pcvd,
            'Dspell_real' : Dspell_real,
            'Lspell_pcvd' : Lspell_pcvd,
            'Lspell_real' : Lspell_real,
            'L_shared'    : L_shared
    }
    experiment_dict_H = experiment_dict_D.copy()
    experiment_dict_H['UnempParam0'] = UnempH
    experiment_dict_H['DeepParam0'] = DeepH
    experiment_dict_C = experiment_dict_D.copy()
    experiment_dict_C['UnempParam0'] = UnempC
    experiment_dict_C['DeepParam0'] = DeepC
    experiment_dicts = [experiment_dict_D, experiment_dict_H, experiment_dict_C]
    
    # Begin the experiment by resetting each type's state to the baseline values
    PopCount = 0
    for ThisType in Agents:
        ThisType.read_shocks = True
        e = ThisType.EducType
        ThisType(**experiment_dicts[e])
        PopCount += ThisType.AgentCount
        
    # Update the perceived and actual Markov arrays, solve and re-draw shocks if
    # warranted, then impose the pandemic shock and the stimulus, and finally
    # simulate the model for three years.
    experiment_commands = ['updateMrkvArray()', 'solveIfChanged()',
                           'makeShocksIfChanged()', 'initializeSim()',
                           'hitWithPandemicShock()', 'announceStimulus()',
                           'simulate()']
    multiThreadCommandsFake(Agents, experiment_commands)
    
    # Extract simulated consumption, labor income, and weight data
    cNrm_all = np.concatenate([ThisType.cNrmNow_hist for ThisType in Agents], axis=1)
    lLvl_all = np.concatenate([ThisType.lLvlNow_hist for ThisType in Agents], axis=1)
    
    Mrkv_hist = np.concatenate([ThisType.MrkvNow_hist for ThisType in Agents], axis=1)
    t_cycle_hist = np.concatenate([ThisType.t_cycle_hist for ThisType in Agents], axis=1)
    u_all = np.concatenate([ThisType.uNow_hist for ThisType in Agents], axis=1)
    w_all = np.concatenate([ThisType.wNow_hist for ThisType in Agents], axis=1)
    pLvl_all = np.concatenate([ThisType.pLvlNow_hist for ThisType in Agents], axis=1)
    Weight_all = np.concatenate([ThisType.Weight_hist for ThisType in Agents], axis=1)
    pLvl_all /= PlvlAgg_adjuster
    lLvl_all /= PlvlAgg_adjuster
    cLvl_all = cNrm_all*pLvl_all
    
    # Get initial Markov states
    Mrkv_init = np.concatenate([ThisType.MrkvNow_hist[0,:] for ThisType in Agents])
    Age_init = np.concatenate([ThisType.age_base for ThisType in Agents])
    WorkingAge = Age_init <= 163
    Employed = np.logical_and(np.logical_or(Mrkv_init == 0, Mrkv_init == 3), WorkingAge)
    Unemployed = np.logical_and(np.logical_or(Mrkv_init == 1, Mrkv_init == 4), WorkingAge)
    DeepUnemp = np.logical_and(np.logical_or(Mrkv_init == 2, Mrkv_init == 5), WorkingAge)
    MrkvTags = (np.vstack([Employed, Unemployed, DeepUnemp, WorkingAge]))
    
    # Calculate an alternate version of labor and transfer income that removes all
    # transitory shocks except unemployment.
    yAlt_all = pLvl_all.copy()
    Checks = np.zeros_like(yAlt_all)
    Checks[T_ahead,:] = np.concatenate([ThisType.StimLvl for ThisType in Agents], axis=0)
    yAlt_all[u_all.astype(bool)] *= Agents[0].IncUnemp
    if (hasattr(Agents[0],'ContUnempBenefits') and Agents[0].ContUnempBenefits):
        yAlt_all[np.logical_and(Mrkv_hist==4,t_cycle_hist <= 163)] += BonusUnemp/PlvlAgg_adjuster
        yAlt_all[np.logical_and(Mrkv_hist==5,t_cycle_hist <= 163)] += BonusDeep/PlvlAgg_adjuster
    else:
        yAlt_all[0,Unemployed] += BonusUnemp/PlvlAgg_adjuster
        yAlt_all[0,DeepUnemp] += BonusDeep/PlvlAgg_adjuster
    yAlt_all += Checks/PlvlAgg_adjuster
    laborandtransferLvl_all = yAlt_all
    
    # Partition the working age agents by the initial permanent income
    pLvl_init = np.concatenate([ThisType.pLvlNow_hist[0,:] for ThisType in Agents])
    Weight_init = np.concatenate([ThisType.Weight_hist[0,:] for ThisType in Agents])*WorkingAge
    quintile_cuts = getPercentiles(pLvl_init, Weight_init, [0.2,0.4,0.6,0.8])
    inc_quint = np.zeros(PopCount)
    for q in range(4):
        inc_quint += pLvl_init >= quintile_cuts[q]
    which_inc_quint = np.zeros((5,PopCount), dtype=bool)
    for q in range(5):
        which_inc_quint[q,:] = inc_quint == q
    which_inc_quint[:,np.logical_not(WorkingAge)] = False
        
    # Calculate the time series of mean consumption in each quarter
    C = np.sum(cLvl_all*Weight_all, axis=1) / np.sum(Weight_all, axis=1)
    
    # Calculate unemployment rate each quarter
    U = np.sum(u_all*Weight_all, axis=1) / np.sum(w_all*Weight_all, axis=1)
    
    # Calculate mean consumption *among the working age* by initial Markov state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore divide by zero warning when no one is deeply unemployed
        C_by_mrkv = np.zeros((4,T))
        C_by_mrkv[0,:] = np.sum(cLvl_all*Weight_all*Employed, axis=1) / np.sum(Weight_all*Employed, axis=1)
        C_by_mrkv[1,:] = np.sum(cLvl_all*Weight_all*Unemployed, axis=1) / np.sum(Weight_all*Unemployed, axis=1)
        C_by_mrkv[2,:] = np.sum(cLvl_all*Weight_all*DeepUnemp, axis=1) / np.sum(Weight_all*DeepUnemp, axis=1)
        C_by_mrkv[3,:] = np.sum(cLvl_all*Weight_all*WorkingAge, axis=1) / np.sum(Weight_all*WorkingAge, axis=1)
    
    # Calculate mean consumption *among the working age* by income quintile
    C_by_inc = np.zeros((5,T))
    LT_by_inc = np.zeros((5,T))
    for q in range(5):
        C_by_inc[q,:] = np.sum(cLvl_all*Weight_all*which_inc_quint[q,:], axis=1) / np.sum(Weight_all*which_inc_quint[q,:], axis=1)
        LT_by_inc[q,:] = np.sum(laborandtransferLvl_all*Weight_all*which_inc_quint[q,:], axis=1) / np.sum(Weight_all*which_inc_quint[q,:], axis=1)
    return C, C_by_mrkv, C_by_inc, cLvl_all, Weight_all, MrkvTags, U, laborandtransferLvl_all, LT_by_inc


def makePandemicShockProbsFigure(Agents,spec_name,PanShock,
                                 UnempD,UnempH,UnempC,UnempP,UnempA1,UnempA2,
                                 DeepD,DeepH,DeepC,DeepP,DeepA1,DeepA2,
                                 show_fig=True):
    '''
    Make figures showing the probability of becoming unemployed and deeply
    unemployed when the pandemic hits, by age, income, and education.
    
    Parameters
    ----------
    Agents : [AgentType]
        List of types of agents in the economy.  Only the first three types
        will actually be used by this function, and not changed at all--
        they are deepcopied and manipulated.
    spec_name : str
        Filename suffix for figure to be saved.
    PanShock : bool
        This isn't used; it's here for technical reasons.
    UnempD : float
        Constant for highschool dropouts in the Markov-shock logit for unemployment.
    UnempH : float
        Constant for highschool grads in the Markov-shock logit for unemployment.
    UnempC : float
        Constant for college goers in the Markov-shock logit for unemployment.
    UnempP : float
        Coefficient on log permanent income in the Markov-shock logit for unemploment.
    UnempA1 : float
        Coefficient on age in the Markov-shock logit for unemployment.
    UnempA2 : float
        Coefficient on age squared in the Markov-shock logit for unemployment.
    DeepD : float
        Constant for highschool dropouts in the Markov-shock logit for deep unemployment.
    DeepH : float
        Constant for highschool grads in the Markov-shock logit for deep unemployment.
    DeepC : float
        Constant for college goers in the Markov-shock logit for deep unemployment.
    DeepP : float
        Coefficient on log permanent income in the Markov-shock logit for deep unemploment.
    DeepA1 : float
        Coefficient on age in the Markov-shock logit for deep unemployment.
    DeepA2 : float
        Coefficient on age squared in the Markov-shock logit for deep unemployment.
    show_fig : bool
        Indicator for whether the figure should be displayed to screen; default True.
        
    Returns
    -------
    data: Dict
        A dictionary with data to plot pandemic shock unemployment probablities.
    '''
    BigPop = 100000
    T = Agents[0].T_retire+1
    PlvlAgg_adjuster = Agents[0].PermGroFacAgg**(-np.arange(T))
    Unemp0 = [UnempD, UnempH, UnempC]
    Deep0 = [DeepD, DeepH, DeepC]
    
    # Initialize an array to hold permanent income percentile data
    pctiles = [0.05,0.25,0.5,0.75,0.95]
    pLvlPercentiles = np.zeros((3,5,T)) + np.nan
    
    # Get distribution of permanent income at each age for each education level,
    # as well as the probability of unemployment and deep unemployment for all
    TempTypes = deepcopy(Agents)
    for n in range(3):
        ThisType = TempTypes[n]
        e = ThisType.EducType
        ThisType.AgentCount = int(EducShares[e]*BigPop)
        ThisType.mortality_off = True
        ThisType.T_sim = T
        ThisType.initializeSim()
        pLvlInit = ThisType.pLvlNow.copy()
        ThisType.makeShockHistory()
        ThisType.pLvlNow_hist = np.cumprod(ThisType.PermShkNow_hist, axis=0)
        ThisType.pLvlNow_hist *= np.tile(np.reshape(pLvlInit,(1,ThisType.AgentCount)),(T,1))
        ThisType.pLvlNow_hist *= np.tile(np.reshape(PlvlAgg_adjuster,(T,1)),(1,ThisType.AgentCount))
        for t in range(T):
            pLvlPercentiles[n,:,t] = getPercentiles(ThisType.pLvlNow_hist[t,:],percentiles=pctiles)
        AgeArray = np.tile(np.reshape(np.arange(T)/4 + 24,(T,1)),(1,ThisType.AgentCount))
        AgeSqArray = np.tile(np.reshape(np.arange(T)/4 + 24,(T,1)),(1,ThisType.AgentCount))
        UnempX = np.exp(Unemp0[e] + UnempP*np.log(ThisType.pLvlNow_hist) + UnempA1*AgeArray + UnempA2*AgeSqArray)
        DeepX  = np.exp(Deep0[e]  + DeepP*np.log(ThisType.pLvlNow_hist)  + DeepA1*AgeArray  + DeepA2*AgeSqArray)
        denom = (1. + UnempX + DeepX)
        UnempPrb = UnempX/denom
        DeepPrb = DeepX/denom
        ThisType.UnempPrb_hist = UnempPrb
        ThisType.DeepPrb_hist = DeepPrb
        
    UnempPrbAll = np.concatenate([ThisType.UnempPrb_hist for ThisType in TempTypes], axis=1)
    DeepPrbAll = np.concatenate([ThisType.DeepPrb_hist for ThisType in TempTypes], axis=1)
    
    # Get overall unemployment and deep unemployment probabilities at each age
    UnempPrbMean = np.mean(UnempPrbAll,axis=1)
    DeepPrbMean  = np.mean(DeepPrbAll,axis=1)
    
    data = dict()
    # Plot overall unemployment probabilities in top left
    plt.subplot(2,2,1)
    AgeVec = np.arange(T)/4 + 24
    plt.plot(AgeVec,UnempPrbMean,'-b')
    plt.plot(AgeVec,DeepPrbMean,'-r')
    data['overall'] = [AgeVec, UnempPrbMean, DeepPrbMean]
    plt.legend(['Unemployed', 'Deep unemp'], loc=1)
    plt.ylim(0,0.20)
    plt.xticks([])
    plt.ylabel('Probability')
    plt.title('All education (mean)')

    
    # Plot dropout unemployment probabilities by permanent income
    e = 0
    p = 0
    data['dropout'] = dict()
    plt.subplot(2,2,2)
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['dropout'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 4
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['dropout'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 2
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'-b')
    plt.plot(AgeVec,DeepPrb,'-r')
    plt.yticks([])
    plt.xticks([])
    plt.ylim(0,0.20)
    data['dropout'][p] = [AgeVec, UnempPrb, DeepPrb]
    plt.title('Dropout')
    
    # Plot highschool unemployment probabilities by permanent income
    e = 1
    p = 0
    data['highschool'] = dict()
    plt.subplot(2,2,3)
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['highschool'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 4
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['highschool'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 2
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'-b')
    plt.plot(AgeVec,DeepPrb,'-r')
    data['highschool'][p] = [AgeVec, UnempPrb, DeepPrb]
    plt.ylim(0,0.20)
    plt.xlabel('Age')
    plt.ylabel('Probability')
    plt.title('High school')
    
    # Plot college unemployment probabilities by permanent income
    e = 2
    p = 0
    data['college'] = dict()

    plt.subplot(2,2,4)
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['college'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 4
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'--b')
    plt.plot(AgeVec,DeepPrb,'--r')
    data['college'][p] = [AgeVec, UnempPrb, DeepPrb]

    p = 2
    UnempX = np.exp(Unemp0[e] + UnempP*np.log(pLvlPercentiles[e,p,:]) + UnempA1*AgeVec + UnempA2*AgeVec**2)
    DeepX  = np.exp(Deep0[e]  + DeepP*np.log(pLvlPercentiles[e,p,:])  + DeepA1*AgeVec  + DeepA2*AgeVec**2)
    denom = (1. + UnempX + DeepX)
    UnempPrb = UnempX/denom
    DeepPrb = DeepX/denom
    plt.plot(AgeVec,UnempPrb,'-b')
    plt.plot(AgeVec,DeepPrb,'-r')
    data['college'][p] = [AgeVec, UnempPrb, DeepPrb]
    plt.ylim(0,0.20)
    plt.xlabel('Age')
    plt.yticks([])
    plt.title('College')
    
    # Save the figure and display it to screen
    plt.suptitle('Unemployment probability after pandemic shock')
    plt.savefig(figs_dir + 'UnempProbByDemog' + spec_name + '.pdf', bbox_inches='tight')
    plt.savefig(figs_dir + 'UnempProbByDemog' + spec_name + '.png', bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.clf()
    
    return data
    
    
def calcCSTWmpcStats(Agents):
    '''
    Calculate and print to screen overall and education-specific aggregate
    wealth to income ratios, as well as the 20th, 40th, 60th, and 80th percentile
    points of the Lorenz curve for (liquid) wealth.
    
    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes in the economy.
        
    Returns
    -------
    None
    '''
    yLvlAll = np.concatenate([ThisType.lLvlNow for ThisType in Agents])
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    AgeAll  = np.concatenate([ThisType.t_age for ThisType in Agents])
    EducAll = np.concatenate([ThisType.EducType*np.ones(ThisType.AgentCount) for ThisType in Agents])
    WeightAll = 1.01**(-0.25*AgeAll)
    yAgg = np.dot(yLvlAll, WeightAll)
    aAgg = np.dot(aLvlAll, WeightAll)
    yAggD = np.dot(yLvlAll, WeightAll*(EducAll==0))
    yAggH = np.dot(yLvlAll, WeightAll*(EducAll==1))
    yAggC = np.dot(yLvlAll, WeightAll*(EducAll==2))
    aAggD = np.dot(aLvlAll, WeightAll*(EducAll==0))
    aAggH = np.dot(aLvlAll, WeightAll*(EducAll==1))
    aAggC = np.dot(aLvlAll, WeightAll*(EducAll==2))
    LorenzPts = getLorenzShares(aLvlAll, weights=WeightAll, percentiles = [0.2, 0.4, 0.6, 0.8])
    
    print('Overall aggregate wealth to income ratio is ' + mystr(aAgg/yAgg) + ' (target 6.60).')
    print('Aggregate wealth to income ratio for dropouts is ' + mystr(aAggD/yAggD) + ' (target 1.60).')
    print('Aggregate wealth to income ratio for high school grads is ' + mystr(aAggH/yAggH) + ' (target 3.78).')
    print('Aggregate wealth to income ratio for college grads is ' + mystr(aAggC/yAggC) + ' (target 8.84).')
    print('Share of liquid wealth of the bottom 20% is ' + mystr(100*LorenzPts[0]) + '% (target 0.0%).')
    print('Share of liquid wealth of the bottom 40% is ' + mystr(100*LorenzPts[1]) + '% (target 0.4%).')
    print('Share of liquid wealth of the bottom 60% is ' + mystr(100*LorenzPts[2]) + '% (target 2.5%).')
    print('Share of liquid wealth of the bottom 80% is ' + mystr(100*LorenzPts[3]) + '% (target 11.7%).')
    
    
def makeConfigFile(param_name,param_min,param_max,N,int_bool=False):
    '''
    Makes a yaml file for the configurator to read, perturbing one parameter over
    a specified range of values.  Saves the file to ./Code/Python/Config/
    
    Parameters
    ----------
    param_name : str
        Name of parameter to perturb in the experiment.
    param_min : float
        Lower bound of values for the paramter to take on.
    param_max : float
        Upper bound of values for the parameter to take on.
    N : int
        Number of evenly spaced parameter values to use.
    int_bool : bool
        Indicator for whether parameter values should be integers.
    '''
    # Make a vector of parameter values
    param_vec = np.linspace(param_min,param_max,N)
    if int_bool:
        param_vec = param_vec.astype(int)
    
    # Initialize the string to be written to the config file
    out = '# This config file perturbs the parameter ' + param_name + ' over ' + str(N) + ' values between ' + str(param_min) + ' and ' + str(param_max) + '.\n'
    
    # Loop over parameter values, writing a new spec for each
    for n in range(N):
        if not int_bool:
            val_str = mystr2(np.abs(param_vec[n]))
        else:
            val_str = str(param_vec[n])
        if param_vec[n] < 0.:
            val_str = 'n' + val_str
        out += '\n'
        out += param_name + '_' + val_str + ':\n'
        if not int_bool:
            out += '    ' + param_name + ': ' + mystr2(param_vec[n]) + '\n'
        else:
            out += '    ' + param_name + ': ' + str(param_vec[n]) + '\n'
        
    # Write the output string to a file
    file_name = './Config/' + param_name + '_vary.yaml'
    with open(file_name, 'w') as f:
        f.write(out)
        f.close()
    print('Wrote config file to ' + file_name)
