'''
This is the main script for the paper, "Modeling the Consumption Response to the
CARES Act" by Carroll, Crawley, Slacalek, and White.  It produces all figures.
'''
from Parameters import T_sim, init_dropout, init_highschool, init_college, EducShares, DiscFacDstns,\
     AgentCountTotal, base_dict, stimulus_changes, pandemic_changes, deep_pandemic_changes, figs_dir, AggregationFactor
from GiveItAwayModel import GiveItAwayNowType
from GiveItAwayTools import runExperiment, makePandemicShockProbsFigure, calcCSTWmpcStats
from HARK import multiThreadCommands, multiThreadCommandsFake
from HARK.distribution import DiscreteDistribution
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


if __name__ == '__main__':
    
    mystr = lambda x : '{:.2f}'.format(x)
    t_start = time()

    # Make baseline types
    DropoutType = GiveItAwayNowType(**init_dropout)
    HighschoolType = GiveItAwayNowType(**init_highschool)
    CollegeType = GiveItAwayNowType(**init_college)
    BaseTypeList = [DropoutType, HighschoolType, CollegeType]
    
    # Fill in the Markov income distribution for each base type
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([DropoutType.IncUnemp])])
    IncomeDstn_big = []
    for ThisType in BaseTypeList:
        for t in range(ThisType.T_cycle):
            if t < ThisType.T_retire:
                IncomeDstn_big.append([ThisType.IncomeDstn[t], IncomeDstn_unemp, IncomeDstn_unemp, ThisType.IncomeDstn[t], IncomeDstn_unemp, IncomeDstn_unemp])
                ThisType.IncomeDstn[t] = [ThisType.IncomeDstn[t], IncomeDstn_unemp]
            else:
                IncomeDstn_big.append(6*[ThisType.IncomeDstn[t]])
                ThisType.IncomeDstn[t] = 2*[ThisType.IncomeDstn[t]]
        ThisType.IncomeDstn_big = IncomeDstn_big
            
    # Make the overall list of types
    TypeList = []
    n = 0
    for b in range(DiscFacDstns[0].X.size):
        for e in range(3):
            DiscFac = DiscFacDstns[e].X[b]
            AgentCount = int(np.floor(AgentCountTotal*EducShares[e]*DiscFacDstns[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = DiscFac
            ThisType.seed = n
            TypeList.append(ThisType)
            n += 1
    base_dict['Agents'] = TypeList
    
    # Make a figure to show unemployment probabilities by demographics
    if True:
        t0 = time()    
        makePandemicShockProbsFigure(BaseTypeList,'Deep',**deep_pandemic_changes)
        t1 = time()
        print('Making unemployment probability by demographics (long pandemic) figure took ' + mystr(t1-t0) + ' seconds.')

        t0 = time()    
        makePandemicShockProbsFigure(BaseTypeList,'Basic',**pandemic_changes)
        t1 = time()
        print('Making unemployment probability by demographics figure took ' + mystr(t1-t0) + ' seconds.')
    # Solve and simulate each type to get to the initial distribution of states
    # and then prepare for new counterfactual simulations
    t0 = time()
    baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()',
                         'switchToCounterfactualMode()', 'makeAlternateShockHistories()']
    multiThreadCommands(TypeList, baseline_commands)
    t1 = time()
    print('Making the baseline distribution of states and preparing to run counterfactual simulations took ' + mystr(t1-t0) + ' seconds.')
    calcCSTWmpcStats(TypeList)
    
    # Define dictionaries to be used in counterfactual scenarios
    stim_dict = base_dict.copy()
    stim_dict.update(**stimulus_changes)
    pan_dict = base_dict.copy()
    pan_dict.update(**pandemic_changes)
    both_dict = pan_dict.copy()
    both_dict.update(**stimulus_changes)
    checks_pan_dict = both_dict.copy()
    checks_pan_dict["BonusUnemp"] = 0.0
    checks_pan_dict["BonusDeep"] = 0.0
    unemp_pan_dict = both_dict.copy()
    unemp_pan_dict["StimMax"] = 0.0
    
    deep_pan_dict = pan_dict.copy()
    deep_pan_dict.update(**deep_pandemic_changes)
    stim_deep_pan_dict = deep_pan_dict.copy()
    stim_deep_pan_dict.update(**stimulus_changes)
    
    # Run the baseline consumption level
    t0 = time()
    C_base, X_base, Z_base, cAll_base, Weight_base, Mrkv_base, U_base, ltAll_base, LT_by_inc_base = runExperiment(**base_dict)
    t1 = time()
    print('Calculating baseline consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Get consumption when there's a stimulus (no pandemic)
    t0 = time()
    C_stim, X_stim, Z_stim, cAll_stim, Weight_stim, Mrkv_stim, U_stim, ltAll_stim, LT_by_inc_stim = runExperiment(**stim_dict)
    t1 = time()
    print('Calculating consumption with stimulus took ' + mystr(t1-t0) + ' seconds.')
    
    # Get consumption when the pandemic hits (no stim)
    t0 = time()
    C_pan, X_pan, Z_pan, cAll_pan, Weight_pan, Mrkv_pan, U_pan, ltAll_pan, LT_by_inc_pan = runExperiment(**pan_dict)
    t1 = time()
    print('Calculating consumption with pandemic took ' + mystr(t1-t0) + ' seconds.')
    
    # Get consumption when the pandemic hits and there's a stimulus
    t0 = time()
    C_both, X_both, Z_both, cAll_both, Weight_both, Mrkv_both, U_both, ltAll_both, LT_by_inc_both = runExperiment(**both_dict)
    t1 = time()
    print('Calculating consumption with pandemic and stimulus took ' + mystr(t1-t0) + ' seconds.')
    
    # Get consumption when the pandemic hits and there's a stimulus check, but no unemployment benefits
    t0 = time()
    C_checks_pan, X_checks_pan, Z_checks_pan, cAll_checks_pan, Weight_checks_pan, Mrkv_checks_pan, U_checks_pan, ltAll_checks_pan, LT_by_inc_checks_pan = runExperiment(**checks_pan_dict)
    t1 = time()
    print('Calculating consumption with pandemic and checks took ' + mystr(t1-t0) + ' seconds.')
    
    # Get consumption when the pandemic hits and there's unemployement benefits, but no check
    t0 = time()
    C_unemp_pan, X_unemp_pan, Z_unemp_pan, cAll_unemp_pan, Weight_unemp_pan, Mrkv_unemp_pan, U_unemp_pan, ltAll_unemp_pan, LT_by_inc_unemp_pan = runExperiment(**unemp_pan_dict)
    t1 = time()
    print('Calculating consumption with pandemic and unemployment benefits took ' + mystr(t1-t0) + ' seconds.')
    
    uniform_pan_dict = both_dict.copy()
    uniform_pan_dict["BonusUnemp"] = 0.0
    uniform_pan_dict["BonusDeep"] = 0.0
    uniform_pan_dict["StimCut0"] = None
    uniform_pan_dict["StimCut1"] = None
    
    mean_benefits = np.sum((ltAll_both-ltAll_pan)*Weight_pan, axis=1) / np.sum(Weight_pan, axis=1)
    mean_check = mean_benefits[0]
    mean_stim = mean_benefits[1]
    uniform_pan_dict["StimMax"] = mean_check + mean_stim # Amount if we gave everyone the same, keeping the total the same
       
    # Get consumption when the pandemic hits and there's unemployement benefits, but no check
    t0 = time()
    C_uniform_pan, X_uniform_pan, Z_uniform_pan, cAll_uniform_pan, Weight_uniform_pan, Mrkv_uniform_pan, U_uniform_pan, ltAll_uniform_pan, LT_by_inc_unif_pan = runExperiment(**uniform_pan_dict)
    t1 = time()
    print('Calculating consumption with pandemic and uniform checks took ' + mystr(t1-t0) + ' seconds.')
    
    #Run a much scaled down version
    scale_down_fac = 100.0
    scaled_down_dict = both_dict.copy()
    scaled_down_dict["BonusUnemp"] = both_dict["BonusUnemp"]/scale_down_fac
    scaled_down_dict["BonusDeep"] = both_dict["BonusDeep"]/scale_down_fac
    scaled_down_dict["StimMax"] = both_dict["StimMax"]/scale_down_fac
    scaled_down_uniform_dict = uniform_pan_dict.copy()
    scaled_down_uniform_dict["StimMax"] = uniform_pan_dict["StimMax"]/scale_down_fac
    
    # Get consumption when the pandemic hits and there's scaled down response
    t0 = time()
    C_scaled_down, X_scaled_down, Z_scaled_down, cAll_scaled_down, Weight_scaled_down, Mrkv_scaled_down, U_scaled_down, ltAll_scaled_down, LT_by_inc_scaled_down = runExperiment(**scaled_down_dict)
    C_scaled_down_uniform, X_scaled_down_uniform, Z_scaled_down_uniform, cAll_scaled_down_uniform, Weight_scaled_down_uniform, Mrkv_scaled_down_uniform, U_scaled_down_uniform, ltAll_scaled_down_uniform, LT_by_inc_scaled_down_uniform = runExperiment(**scaled_down_uniform_dict)
    t1 = time()
    print('Calculating consumption with scaled down response took ' + mystr(t1-t0) + ' seconds.')
    
    long_pandemic_dict = deep_pan_dict.copy()
    long_pandemic_dict['Lspell_pcvd'] = 4.0
    long_pandemic_dict['Lspell_real'] = 4.0
    long_pandemic_stim_dict = long_pandemic_dict.copy()
    long_pandemic_stim_dict.update(**stimulus_changes)
    # Get consumption when the pandemic hits unemployment, but there is no loss to utility
    t0 = time()
    C_long_pandemic, X_long_pandemic, Z_long_pandemic, cAll_long_pandemic, Weight_long_pandemic, Mrkv_long_pandemic, U_long_pandemic, ltAll_long_pandemic, LT_by_inc_long_pandemic = runExperiment(**long_pandemic_dict)
    C_long_pandemic_stim, X_long_pandemic_stim, Z_long_pandemic_stim, cAll_long_pandemic_stim, Weight_long_pandemic_stim, Mrkv_long_pandemic_stim, U_long_pandemic_stim, ltAll_long_pandemic_stim, LT_by_inc_long_pandemic_stim = runExperiment(**long_pandemic_stim_dict)
    t1 = time()
    print('Calculating consumption with long pandemic took ' + mystr(t1-t0) + ' seconds.')
    for a in long_pandemic_dict['Agents']:
        a.ContUnempBenefits=True
    # Get consumption when the pandemic hits unemployment, but there is no loss to utility
    t0 = time()
    C_long_pandemic_cont, X_long_pandemic_cont, Z_long_pandemic_cont, cAll_long_pandemic_cont, Weight_long_pandemic_cont, Mrkv_long_pandemic_cont, U_long_pandemic_cont, ltAll_long_pandemic_cont, LT_by_inc_long_pandemic_cont = runExperiment(**long_pandemic_stim_dict)
    t1 = time()
    print('Calculating consumption with long pandemic and continued benefits took ' + mystr(t1-t0) + ' seconds.')
    for a in long_pandemic_dict['Agents']:
        a.ContUnempBenefits = False

    do_no_utility_loss = True
    if (do_no_utility_loss):
        TypeList_no_utility_loss = []
        n = 0
        for b in range(DiscFacDstns[0].X.size):
            for e in range(3):
                DiscFac = DiscFacDstns[e].X[b]
                AgentCount = int(np.floor(AgentCountTotal*EducShares[e]*DiscFacDstns[e].pmf[b]))
                ThisType = deepcopy(BaseTypeList[e])
                ThisType.AgentCount = AgentCount
                ThisType.DiscFac = DiscFac
                ThisType.uPfac_big = np.array(3*[1.0] + 3*[1.0])
                ThisType.seed = n
                TypeList_no_utility_loss.append(ThisType)
                n += 1
        no_utility_loss_dict = pan_dict.copy()
        no_utility_loss_dict['Agents'] = TypeList_no_utility_loss
        # Solve and simulate each type 
        t0 = time()
        multiThreadCommands(TypeList_no_utility_loss, baseline_commands)
        t1 = time()
        print('Making the no utility loss distribution of states and preparing to run counterfactual simulations took ' + mystr(t1-t0) + ' seconds.')
       
        # Get consumption when the pandemic hits unemployment, but there is no loss to utility
        t0 = time()
        C_no_utility_loss, X_no_utility_loss, Z_no_utility_loss, cAll_no_utility_loss, Weight_no_utility_loss, Mrkv_no_utility_loss, U_no_utility_loss, ltAll_no_utility_loss, LT_by_inc_no_utility_loss = runExperiment(**no_utility_loss_dict)
        t1 = time()
        print('Calculating consumption with pandemic but no utility loss took ' + mystr(t1-t0) + ' seconds.')
     
    # Calculate baseline consumption for those who *would* be in each Markov state in the pandemic
    X_alt = np.zeros((4,T_sim))
    X_alt[0,:] = np.sum(cAll_base*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    X_alt[1,:] = np.sum(cAll_base*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    X_alt[2,:] = np.sum(cAll_base*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    X_alt[3,:] = np.sum(cAll_base*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    
    # Calculate [baseline] labor and transfer income for those who *would* be in each Markov state in the pandemic
    LT_base = np.zeros((4,T_sim))
    LT_base[0,:] = np.sum(ltAll_base*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    LT_base[1,:] = np.sum(ltAll_base*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    LT_base[2,:] = np.sum(ltAll_base*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    LT_base[3,:] = np.sum(ltAll_base*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    LT_base_all = np.sum(ltAll_base*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # Calculate [pandemic] labor and transfer income for those who *would* be in each Markov state in the pandemic
    LT_pan = np.zeros((4,T_sim))
    LT_pan[0,:] = np.sum(ltAll_pan*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    LT_pan[1,:] = np.sum(ltAll_pan*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    LT_pan[2,:] = np.sum(ltAll_pan*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    LT_pan[3,:] = np.sum(ltAll_pan*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    LT_pan_all = np.sum(ltAll_pan*Weight_base, axis=1) / np.sum(Weight_base, axis=1)   
    
    # Calculate [pandemic with stimulus] labor and transfer income for those who *would* be in each Markov state in the pandemic
    LT_both = np.zeros((4,T_sim))
    LT_both[0,:] = np.sum(ltAll_both*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    LT_both[1,:] = np.sum(ltAll_both*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    LT_both[2,:] = np.sum(ltAll_both*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    LT_both[3,:] = np.sum(ltAll_both*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    LT_both_all = np.sum(ltAll_both*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # Calculate [pandemic with only stimulus checks] labor and transfer income for those who *would* be in each Markov state in the pandemic
    LT_checks_pan = np.zeros((4,T_sim))
    LT_checks_pan[0,:] = np.sum(ltAll_checks_pan*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    LT_checks_pan[1,:] = np.sum(ltAll_checks_pan*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    LT_checks_pan[2,:] = np.sum(ltAll_checks_pan*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    LT_checks_pan[3,:] = np.sum(ltAll_checks_pan*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    LT_checks_pan_all = np.sum(ltAll_checks_pan*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # Calculate [pandemic with only bonus unemployment] labor and transfer income for those who *would* be in each Markov state in the pandemic
    LT_unemp_pan = np.zeros((4,T_sim))
    LT_unemp_pan[0,:] = np.sum(ltAll_unemp_pan*Weight_base*Mrkv_pan[0,:], axis=1) / np.sum(Weight_base*Mrkv_pan[0,:], axis=1)
    LT_unemp_pan[1,:] = np.sum(ltAll_unemp_pan*Weight_base*Mrkv_pan[1,:], axis=1) / np.sum(Weight_base*Mrkv_pan[1,:], axis=1)
    LT_unemp_pan[2,:] = np.sum(ltAll_unemp_pan*Weight_base*Mrkv_pan[2,:], axis=1) / np.sum(Weight_base*Mrkv_pan[2,:], axis=1)
    LT_unemp_pan[3,:] = np.sum(ltAll_unemp_pan*Weight_base*Mrkv_pan[3,:], axis=1) / np.sum(Weight_base*Mrkv_pan[3,:], axis=1)
    LT_unemp_pan_all = np.sum(ltAll_unemp_pan*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # long pandemic, no stimulus
    LT_long_pandemic = np.zeros((4,T_sim))
    LT_long_pandemic[0,:] = np.sum(ltAll_long_pandemic*Weight_base*Mrkv_long_pandemic[0,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[0,:], axis=1)
    LT_long_pandemic[1,:] = np.sum(ltAll_long_pandemic*Weight_base*Mrkv_long_pandemic[1,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[1,:], axis=1)
    LT_long_pandemic[2,:] = np.sum(ltAll_long_pandemic*Weight_base*Mrkv_long_pandemic[2,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[2,:], axis=1)
    LT_long_pandemic[3,:] = np.sum(ltAll_long_pandemic*Weight_base*Mrkv_long_pandemic[3,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[3,:], axis=1)
    LT_long_pandemic_all = np.sum(ltAll_long_pandemic*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # long pandemic, CARES act
    LT_long_pandemic_stim = np.zeros((4,T_sim))
    LT_long_pandemic_stim[0,:] = np.sum(ltAll_long_pandemic_stim*Weight_base*Mrkv_long_pandemic[0,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[0,:], axis=1)
    LT_long_pandemic_stim[1,:] = np.sum(ltAll_long_pandemic_stim*Weight_base*Mrkv_long_pandemic[1,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[1,:], axis=1)
    LT_long_pandemic_stim[2,:] = np.sum(ltAll_long_pandemic_stim*Weight_base*Mrkv_long_pandemic[2,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[2,:], axis=1)
    LT_long_pandemic_stim[3,:] = np.sum(ltAll_long_pandemic_stim*Weight_base*Mrkv_long_pandemic[3,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[3,:], axis=1)
    LT_long_pandemic_stim_all = np.sum(ltAll_long_pandemic_stim*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    # long pandemic, CARES act + continued unemployment insurance
    LT_long_pandemic_cont = np.zeros((4,T_sim))
    LT_long_pandemic_cont[0,:] = np.sum(ltAll_long_pandemic_cont*Weight_base*Mrkv_long_pandemic[0,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[0,:], axis=1)
    LT_long_pandemic_cont[1,:] = np.sum(ltAll_long_pandemic_cont*Weight_base*Mrkv_long_pandemic[1,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[1,:], axis=1)
    LT_long_pandemic_cont[2,:] = np.sum(ltAll_long_pandemic_cont*Weight_base*Mrkv_long_pandemic[2,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[2,:], axis=1)
    LT_long_pandemic_cont[3,:] = np.sum(ltAll_long_pandemic_cont*Weight_base*Mrkv_long_pandemic[3,:], axis=1) / np.sum(Weight_base*Mrkv_long_pandemic[3,:], axis=1)
    LT_long_pandemic_cont_all = np.sum(ltAll_long_pandemic_cont*Weight_base, axis=1) / np.sum(Weight_base, axis=1)
    
    show_fig = False
    quarter_labels=["Q2\n2020","Q3","Q4","Q1\n2021","Q2","Q3","Q4","Q1\n2022","Q2","Q3","Q4","Q1\n2023","Q2"]
    linestyle1 = "-"
    linestyle2 = "--"
    linestyle3 = ":"
    dashes1 = [1,0]
    dashes2 = [10,2]
    dashes3 = [7,1]
    dashes4 = [1,1]
    dashes5 = [2,1,1,1]
    
    plt.plot(U_base*100)
    plt.plot(U_pan*100)
    plt.plot(U_long_pandemic*100)
    plt.xlabel('Quarter')
    plt.ylabel('Unemployment rate (%)')
    plt.title('Unemployment rate with and without pandemic')
    plt.legend(['Baseline','Pandemic','Long Pandemic'])
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'Urate.pdf')
    plt.savefig(figs_dir + 'Urate.png')
    plt.savefig(figs_dir + 'Urate.svg')
    if show_fig:
        plt.show()
    plt.close()
    
    plt.plot(C_base*1000)
    plt.plot(C_pan*1000)
    plt.plot(C_both*1000)
    plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"])
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Average consumption under alternate scenarios')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConResp_examples.pdf')
    plt.savefig(figs_dir + 'ConResp_examples.png')
    plt.savefig(figs_dir + 'ConResp_examples.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(C_base*AggregationFactor,dashes=dashes1)
    plt.plot(C_pan*AggregationFactor,dashes=dashes2)
    plt.plot(C_both*AggregationFactor,dashes=dashes3)
    plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"])
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate quarterly consumption (billion $)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.title('Aggregate consumption under alternate scenarios')
    plt.tight_layout()
    plt.savefig(figs_dir + 'AggConResp_examples.pdf')
    plt.savefig(figs_dir + 'AggConResp_examples.png')
    plt.savefig(figs_dir + 'AggConResp_examples.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_base_all*AggregationFactor,dashes=dashes1)
    plt.plot(LT_pan_all*AggregationFactor,dashes=dashes2)
    plt.plot(LT_both_all*AggregationFactor,dashes=dashes3)
    plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"])
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate labor and transfer income (billion $)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.title('Aggregate household income under alternate scenarios')
    plt.tight_layout()
    plt.savefig(figs_dir + 'AggLT.pdf')
    plt.savefig(figs_dir + 'AggLT.png')
    plt.savefig(figs_dir + 'AggLT.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot((C_stim-C_base)*1000)
    plt.plot((C_both-C_pan)*1000)
    plt.legend(["Stimulus effect, normal times","Stimulus effect, pandemic"])
    plt.xlabel('Quarter')
    plt.ylabel('Average consumption response ($)')
    plt.title('Change in mean consumption from stimulus')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConResp_pandemic_vs_normal.pdf')
    plt.savefig(figs_dir + 'ConResp_pandemic_vs_normal.png')
    plt.savefig(figs_dir + 'ConResp_pandemic_vs_normal.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(X_pan[0,:]*1000,'b',dashes=dashes1)
    plt.plot(X_pan[1,:]*1000,'g',dashes=dashes2)
    plt.plot(X_pan[2,:]*1000,'r',dashes=dashes3)
    plt.plot(X_alt[0,:]*1000,':b')
    plt.plot(X_alt[1,:]*1000,':g')
    plt.plot(X_alt[2,:]*1000,':r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Consumption among working age population (no CARES Act)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    bottom, top = plt.ylim() # want graph graphs with and without stimulus to have same ylim
    plt.ylim(bottom, top+500)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim.pdf')
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim.png')
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(X_both[0,:]*1000,'b',dashes=dashes1)
    plt.plot(X_both[1,:]*1000,'g',dashes=dashes2)
    plt.plot(X_both[2,:]*1000,'r',dashes=dashes3)
    plt.plot(X_alt[0,:]*1000,':b')
    plt.plot(X_alt[1,:]*1000,':g')
    plt.plot(X_alt[2,:]*1000,':r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Consumption among working age population (CARES Act)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.ylim(bottom, top+500)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim.pdf')
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim.png')
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    colors = ['b','r','g','c','m']
    for q in range(5):
        plt.plot((Z_pan[q,:] - Z_base[q,:])/Z_base[q,:]*100, colors[q]+'-')
    plt.legend(['Bottom','Second','Third','Fourth','Top'])
    plt.xlabel('Quarter')
    plt.ylabel('Percentage change in consumption from baseline')
    plt.title('Consumption by income quintile after pandemic')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConPctChangeByIncomeNoStim.pdf')
    plt.savefig(figs_dir + 'ConPctChangeByIncomeNoStim.png')
    plt.savefig(figs_dir + 'ConPctChangeByIncomeNoStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    for q in range(5):
        plt.plot((Z_both[q,:] - Z_base[q,:])/Z_base[q,:]*100, colors[q]+'-')
    plt.legend(['Bottom','Second','Third','Fourth','Top'])
    plt.xlabel('Quarter')
    plt.ylabel('Percentage change in consumption from baseline')
    plt.title('Consumption by income quintile, pandemic plus stimulus')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConPctChangeByIncome.pdf')
    plt.savefig(figs_dir + 'ConPctChangeByIncome.png')
    plt.savefig(figs_dir + 'ConPctChangeByIncome.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot((C_both-C_pan)*AggregationFactor,dashes=dashes1)
    plt.plot((C_checks_pan-C_pan)*AggregationFactor,dashes=dashes2)
    plt.plot((C_unemp_pan-C_pan)*AggregationFactor,dashes=dashes3)
    plt.legend(["Checks and unemployment benefits","Stimulus checks only","Unemployment benefits only"])
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate consumption response (billion $)')
    plt.title('Decomposition of CARES Act effect on aggregate consumption')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'Checks_vs_Unemp.pdf')
    plt.savefig(figs_dir + 'Checks_vs_Unemp.png')
    plt.savefig(figs_dir + 'Checks_vs_Unemp.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_pan[0,:]*1000,'-b')
    plt.plot(LT_pan[1,:]*1000,'-g')
    plt.plot(LT_pan[2,:]*1000,'-r')
    plt.plot(LT_base[0,:]*1000,'--b')
    plt.plot(LT_base[1,:]*1000,'--g')
    plt.plot(LT_base[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Income among working age population (no policy)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomeByEmpStateNoStim.pdf')
    plt.savefig(figs_dir + 'IncomeByEmpStateNoStim.png')
    plt.savefig(figs_dir + 'IncomeByEmpStateNoStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_both[0,:]*1000,'-b')
    plt.plot(LT_both[1,:]*1000,'-g')
    plt.plot(LT_both[2,:]*1000,'-r')
    plt.plot(LT_base[0,:]*1000,'--b')
    plt.plot(LT_base[1,:]*1000,'--g')
    plt.plot(LT_base[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Income among working age population (CARES Act)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim.pdf')
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim.png')
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_checks_pan[0,:]*1000,'-b')
    plt.plot(LT_checks_pan[1,:]*1000,'-g')
    plt.plot(LT_checks_pan[2,:]*1000,'-r')
    plt.plot(LT_base[0,:]*1000,'--b')
    plt.plot(LT_base[1,:]*1000,'--g')
    plt.plot(LT_base[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Income among working age population (with checks only)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomeByEmpStateWChecks.pdf')
    plt.savefig(figs_dir + 'IncomeByEmpStateWChecks.png')
    plt.savefig(figs_dir + 'IncomeByEmpStateWChecks.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_unemp_pan[0,:]*1000,'-b')
    plt.plot(LT_unemp_pan[1,:]*1000,'-g')
    plt.plot(LT_unemp_pan[2,:]*1000,'-r')
    plt.plot(LT_base[0,:]*1000,'--b')
    plt.plot(LT_base[1,:]*1000,'--g')
    plt.plot(LT_base[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Income among working age population (with unemployment benefits only)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomeByEmpStateWUnemp.pdf')
    plt.savefig(figs_dir + 'IncomeByEmpStateWUnemp.png')
    plt.savefig(figs_dir + 'IncomeByEmpStateWUnemp.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot((X_both[0,:]-X_pan[0,:])*1000,'b',dashes=dashes1)
    plt.plot((X_both[1,:]-X_pan[1,:])*1000,'g',dashes=dashes2)
    plt.plot((X_checks_pan[1,:]-X_pan[1,:])*1000,'g',dashes=[3,1])
    plt.plot((X_both[2,:]-X_pan[2,:])*1000,'r',dashes=dashes3)
    plt.plot((X_checks_pan[2,:]-X_pan[2,:])*1000,':r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","(Stimulus checks only)","Deeply unemp in Q2 2020","(Stimulus checks only)"])
    plt.xlabel('Quarter')
    plt.ylabel('Consumption response from CARES Act ($)')
    plt.title('Consumption effect of CARES Act among working age population')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConDeltaByEmpState.pdf')
    plt.savefig(figs_dir + 'ConDeltaByEmpState.png')
    plt.savefig(figs_dir + 'ConDeltaByEmpState.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot((C_both-C_pan)*AggregationFactor,dashes=dashes1)
    plt.plot((C_uniform_pan-C_pan)*AggregationFactor,dashes=dashes2)
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate consumption response (billion $)')
    plt.title('Effect of targeted stimulus on aggregate consumption')
    plt.legend(["Means tested checks and unemployment benefits","Equal checks to all households"])
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'EffectTargeting.pdf')
    plt.savefig(figs_dir + 'EffectTargeting.png')
    plt.savefig(figs_dir + 'EffectTargeting.svg')
    if show_fig:
        plt.show()
    plt.close()

    colors = ['b','r','g','c','m']
    for q in range(5):
        plt.plot((LT_by_inc_pan[q,:] - LT_by_inc_base[q,:])/LT_by_inc_base[q,:]*100, colors[q]+'-')
    plt.legend(['Bottom','Second','Third','Fourth','Top'])
    plt.xlabel('Quarter')
    plt.ylabel('Percentage change in income from baseline')
    plt.title('Income and transfers by permanent income quintile after pandemic')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomePctChangeByIncomeNoStim.pdf')
    plt.savefig(figs_dir + 'IncomePctChangeByIncomeNoStim.png')
    plt.savefig(figs_dir + 'IncomePctChangeByIncomeNoStim.svg')
    if show_fig:
        plt.show()
    plt.close()

    for q in range(5):
        plt.plot((LT_by_inc_both[q,:] - LT_by_inc_base[q,:])/LT_by_inc_base[q,:]*100, colors[q]+'-')
    plt.legend(['Bottom','Second','Third','Fourth','Top'])
    plt.xlabel('Quarter')
    plt.ylabel('Percentage change in income from baseline')
    plt.title('Income and transfers by permanent income quintile, pandemic plus stimulus')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncPctChangeByIncome.pdf')
    plt.savefig(figs_dir + 'IncPctChangeByIncome.png')
    plt.savefig(figs_dir + 'IncPctChangeByIncome.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot((C_both-C_pan)*1000, '-b')
    plt.plot((C_scaled_down-C_pan)*1000*100,'--b')
    plt.plot((C_uniform_pan-C_pan)*1000,'-r')
    plt.plot((C_scaled_down_uniform-C_pan)*1000*100, '--r')
    plt.xlabel('Quarter')
    plt.ylabel('Average consumption response ($1000)')
    plt.title('Effect of targeting')
    plt.legend(["Means tested checks and unemployment benefits","(marginal)","Equal checks","(marginal)"])
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'EffectTargeting_marginal.pdf')
    plt.savefig(figs_dir + 'EffectTargeting_marginal.png')
    plt.savefig(figs_dir + 'EffectTargeting_marginal.svg')
    if show_fig:
        plt.show()
    plt.close()

    if do_no_utility_loss:
        plt.plot(C_base*AggregationFactor)
        plt.plot(C_pan*AggregationFactor)
        plt.plot(C_no_utility_loss*AggregationFactor)
        plt.legend(["Baseline","Pandemic, no fiscal response","Pandemic, no loss to consumption utility"])
        plt.xlabel('Quarter')
        plt.ylabel('Aggregate consumption (billion $)')
        plt.xticks(ticks=range(T_sim), labels=quarter_labels)
        plt.title('Consumption')
        plt.tight_layout()
        plt.savefig(figs_dir + 'AggConResp_no_distancing.pdf')
        plt.savefig(figs_dir + 'AggConResp_no_distancing.png')
        plt.savefig(figs_dir + 'AggConResp_no_distancing.svg')
        if show_fig:
            plt.show()
        plt.close()

        plt.plot((C_pan-C_base)*AggregationFactor,'-k')
        plt.bar(range(T_sim),(C_no_utility_loss-C_base)*AggregationFactor)
        plt.bar(range(T_sim),(C_pan-C_no_utility_loss)*AggregationFactor,bottom=(C_no_utility_loss-C_base)*AggregationFactor,width=0.35)
        plt.legend(['Total effect','Income component','Marginal utility drop component'])
        plt.title('Decomposition of change in consumption from baseline')
        plt.xlabel('Quarter')
        plt.ylabel('Change in aggregate consumption (billion $)')
        plt.xticks(ticks=range(T_sim), labels=quarter_labels)
        plt.hlines(0,-1.0,T_sim,linewidth=0.5)
        plt.xlim(-1,T_sim)
        plt.tight_layout()
        plt.savefig(figs_dir + 'Decomposition.pdf')
        plt.savefig(figs_dir + 'Decomposition.png')
        plt.savefig(figs_dir + 'Decomposition.svg')
        if show_fig:
            plt.show()
        plt.close()

    plt.plot((C_base)*AggregationFactor,dashes=dashes1)
    plt.plot((C_long_pandemic)*AggregationFactor,dashes=dashes2)
    plt.plot((C_long_pandemic_stim)*AggregationFactor,dashes=dashes3)
    plt.plot((C_long_pandemic_cont)*AggregationFactor,dashes=[1,1])
    plt.legend(["Baseline","Long pandemic","Long pandemic with CARES act","Long pandemic, CARES act\nand continued unemployment payments"])
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate quarterly consumption (billion $)')
    plt.title('Long Pandemic Aggregate Consumption')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'DeepPandemic.pdf')
    plt.savefig(figs_dir + 'DeepPandemic.png')
    plt.savefig(figs_dir + 'DeepPandemic.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_base_all*AggregationFactor,dashes=dashes1)
    plt.plot(LT_long_pandemic_all*AggregationFactor,dashes=dashes2)
    plt.plot(LT_long_pandemic_stim_all*AggregationFactor,dashes=dashes3)
    plt.plot(LT_long_pandemic_cont_all*AggregationFactor,dashes=[1,1])
    plt.legend(["Baseline","Long pandemic","Long pandemic with CARES act","Long pandemic, CARES act\nand continued unemployment payments"])
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate labor and transfer income (billion $)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.title('Aggregate household income, long pandemic')
    plt.tight_layout()
    plt.savefig(figs_dir + 'AggLT_long_pandemic.pdf')
    plt.savefig(figs_dir + 'AggLT_long_pandemic.png')
    plt.savefig(figs_dir + 'AggLT_long_pandemic.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(X_long_pandemic[0,:]*1000,'-b')
    plt.plot(X_long_pandemic[1,:]*1000,'-g')
    plt.plot(X_long_pandemic[2,:]*1000,'-r')
    plt.plot(X_alt[0,:]*1000,'--b')
    plt.plot(X_alt[1,:]*1000,'--g')
    plt.plot(X_alt[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Long pandemic, consumption (no policy)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    bottom, top = plt.ylim() # want graph graphs with and without stimulus to have same ylim
    plt.ylim(bottom, top+500)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim_long.pdf')
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim_long.png')
    plt.savefig(figs_dir + 'ConRespByEmpStateNoStim_long.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(X_long_pandemic_stim[0,:]*1000,'-b')
    plt.plot(X_long_pandemic_stim[1,:]*1000,'-g')
    plt.plot(X_long_pandemic_stim[2,:]*1000,'-r')
    plt.plot(X_alt[0,:]*1000,'--b')
    plt.plot(X_alt[1,:]*1000,'--g')
    plt.plot(X_alt[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Long pandemic, consumption (CARES Act)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.ylim(bottom, top+500)
    plt.tight_layout()
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim_long.pdf')
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim_long.png')
    plt.savefig(figs_dir + 'ConRespByEmpStateWStim_long.svg')
    if show_fig:
        plt.show()
    plt.close()

    plt.plot(LT_long_pandemic_stim[0,:]*1000,'-b')
    plt.plot(LT_long_pandemic_stim[1,:]*1000,'-g')
    plt.plot(LT_long_pandemic_stim[2,:]*1000,'-r')    
    plt.plot(LT_long_pandemic[0,:]*1000,'--b')
    plt.plot(LT_long_pandemic[1,:]*1000,'--g')
    plt.plot(LT_long_pandemic[2,:]*1000,'--r')
    plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"])
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Income among working age population (CARES Act)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.tight_layout()
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim_long.pdf')
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim_long.png')
    plt.savefig(figs_dir + 'IncomeByEmpStateWStim_long.svg')
    if show_fig:
        plt.show()
    plt.close()

    # graph in appendix to show quality adjusted cost of consumption
    marg_fac = 0.65
    dollars_spent = np.linspace(0.0,3.0,200)
    luxury_cost = dollars_spent**marg_fac
    real_cost = np.minimum(dollars_spent,luxury_cost)
    
    plt.plot(dollars_spent,dollars_spent,dashes=dashes2)
    plt.plot(dollars_spent,luxury_cost,dashes=dashes3)
    plt.plot(dollars_spent,real_cost,color='black',linewidth=2,dashes=dashes1)
    plt.xlabel("Cost")
    plt.ylabel("Number of consumption units")
    plt.title("Quality Adjusted Cost of Consumption Units")
    plt.legend(["Normal Times",r"$C^{\alpha}$","Lockdown"],loc="lower right")
    plt.tight_layout()
    plt.savefig(figs_dir + 'QualityCost.pdf')
    plt.savefig(figs_dir + 'QualityCost.png')
    plt.savefig(figs_dir + 'QualityCost.svg')
    if show_fig:
        plt.show()
    plt.close()

    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    
