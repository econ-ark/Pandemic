'''
This script runs a stripped down version of the main code for the paper "Modeling
the Consumption Response to the CARES Act" by Carroll, Crawley, Slacalek, and White.
The script produces only a small subset of the figures in the paper.  To use it,
edit the parameter values in parameter_config.py, including the string spec_name,
then run this script.  Figures will be displayed to screen and saved to
./Figures/spec_name/ in PDF format.
'''
from Parameters import T_sim, init_dropout, init_highschool, init_college, EducShares, DiscFacDstns,\
     AgentCountTotal, base_dict, stimulus_changes, pandemic_changes, AggregationFactor, spec_name
from GiveItAwayModel import GiveItAwayNowType
from GiveItAwayTools import runExperiment, makePandemicShockProbsFigure
from HARK.distribution import DiscreteDistribution
from HARK import multiThreadCommands
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os
sns.set()


if __name__ == '__main__':
    t0 = time()
    
    if spec_name == 'ChangeMe':
        print("Hi! It looks like you've left spec_name in parameter_config.py at its default")
        print("value of 'ChangeMe'.  This script will save some figures to ./Figures/ChangeMe/,")
        print("which is fine, but probably not what you intended.  You should change spec_name")
        print("so that your figures are saved in a different subdirectory of ./Figures/.")
    
    mystr = lambda x : '{:.2f}'.format(x)
    figs_dir = '../../Figures/' + spec_name + '/'
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)

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
    for b in range(DiscFacDstns[0].pmf.size):
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
    pandemic_changes['for_mini'] = True
    makePandemicShockProbsFigure(BaseTypeList,spec_name,**pandemic_changes)
    del pandemic_changes['for_mini']
    
    # Solve and simulate each type to get to the initial distribution of states
    # and then prepare for new counterfactual simulations
    baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()',
                         'switchToCounterfactualMode()', 'makeAlternateShockHistories()']
    multiThreadCommands(TypeList, baseline_commands)
    
    # Define dictionaries to be used in counterfactual scenarios
    stim_dict = base_dict.copy()
    stim_dict.update(**stimulus_changes)
    pan_dict = base_dict.copy()
    pan_dict.update(**pandemic_changes)
    both_dict = pan_dict.copy()
    both_dict.update(**stimulus_changes)
    
    # Run the baseline consumption level
    C_base, X_base, Z_base, cAll_base, Weight_base, Mrkv_base, U_base, ltAll_base, LT_by_inc_base = runExperiment(**base_dict)
    
    # Get consumption when the pandemic hits (no stim)
    C_pan, X_pan, Z_pan, cAll_pan, Weight_pan, Mrkv_pan, U_pan, ltAll_pan, LT_by_inc_pan = runExperiment(**pan_dict)
    
    # Get consumption when the pandemic hits and there's a stimulus
    C_both, X_both, Z_both, cAll_both, Weight_both, Mrkv_both, U_both, ltAll_both, LT_by_inc_both = runExperiment(**both_dict)
    
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
    
    quarter_labels=["Q2\n2020","Q3","Q4","Q1\n2021","Q2","Q3","Q4","Q1\n2022","Q2","Q3","Q4","Q1\n2023","Q2","Q3"]
    
    # Plot the unemployment rate over time in baseline and pandemic
    plt.plot(U_base*100)
    plt.plot(U_pan*100)
    plt.xlabel('Quarter')
    plt.ylabel('Unemployment rate (%)')
    plt.title('Unemployment rate', fontsize=14)
    plt.legend(['Baseline','Pandemic'], loc=1)
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.ylim(0.,22.)
    plt.tight_layout()
    plt.savefig(figs_dir + 'Urate.pdf')
    plt.show()
    
    # Plot aggregate consumption under baseline, pandemic, and pandemic + CARES Act
    plt.plot(C_base*AggregationFactor)
    plt.plot(C_pan*AggregationFactor)
    plt.plot(C_both*AggregationFactor)
    plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"], loc=4)
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate quarterly consumption (billion $)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.title('Aggregate consumption', fontsize=14)
    plt.ylim(2200.,3000.)
    plt.tight_layout()
    plt.savefig(figs_dir + 'AggC.pdf')
    plt.show()
    
    # Plot aggregate income under baseline, pandemic, and pandemic + CARES Act
    plt.plot(LT_base_all*AggregationFactor)
    plt.plot(LT_pan_all*AggregationFactor)
    plt.plot(LT_both_all*AggregationFactor)
    plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"], loc=4)
    plt.xlabel('Quarter')
    plt.ylabel('Aggregate labor and transfer income (billion $)')
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.title('Aggregate income', fontsize=14)
    plt.ylim(2200.,3000.)
    plt.tight_layout()
    plt.savefig(figs_dir + 'AggLT.pdf')
    plt.show()
    
    # Plot average consumption by initial employment state after pandemic
    plt.plot(X_both[0,:]*1000,'-b')
    plt.plot(X_both[1,:]*1000,'-g')
    plt.plot(X_both[2,:]*1000,'-r')
    plt.plot(X_pan[0,:]*1000,'-.b')
    plt.plot(X_pan[1,:]*1000,'-.g')
    plt.plot(X_pan[2,:]*1000,'-.r')
    plt.plot(X_alt[0,:]*1000,'--b')
    plt.plot(X_alt[1,:]*1000,'--g')
    plt.plot(X_alt[2,:]*1000,'--r')
    plt.legend(["Employed after pandemic","Unemployed after pandemic","Deeply unemp after pandemic"], loc=4)
    plt.xlabel('Quarter')
    plt.ylabel('Average quarterly consumption ($)')
    plt.title('Average consumption among working age population', fontsize=14)
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.ylim(6000,15000)
    plt.tight_layout()
    plt.savefig(figs_dir + 'CbyEmpState.pdf')
    plt.show()
    
    # Plot average labor income plus transfers by initial employment state after pandemic
    plt.plot(LT_both[0,:]*1000,'-b')
    plt.plot(LT_both[1,:]*1000,'-g')
    plt.plot(LT_both[2,:]*1000,'-r')
    plt.plot(LT_pan[0,:]*1000,'-.b')
    plt.plot(LT_pan[1,:]*1000,'-.g')
    plt.plot(LT_pan[2,:]*1000,'-.r')
    plt.plot(LT_base[0,:]*1000,'--b')
    plt.plot(LT_base[1,:]*1000,'--g')
    plt.plot(LT_base[2,:]*1000,'--r')
    plt.legend(["Employed after pandemic","Unemployed after pandemic","Deeply unemp after pandemic"], loc=4)
    plt.xlabel('Quarter')
    plt.ylabel('Average labor and transfer income ($)')
    plt.title('Average income among working age population', fontsize=14)
    plt.xticks(ticks=range(T_sim), labels=quarter_labels)
    plt.ylim(2000,15000)
    plt.tight_layout()
    plt.savefig(figs_dir + 'LTbyEmpState.pdf')
    plt.show()
    
    t1 = time()
    print('Running the specification called ' + spec_name + ' took ' + mystr(t1-t0) + ' seconds.')