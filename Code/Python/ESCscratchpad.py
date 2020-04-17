# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:59:07 2020

@author: edmun
"""
    
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
print('Calculating consumption with long pandemic  took ' + mystr(t1-t0) + ' seconds.')


#    four_quarters_dict = deep_pan_dict.copy()
#    four_quarters_dict['Lspell_pcvd'] = 4.0
#    four_quarters_dict['Lspell_real'] = 4.0
#    four_quarters_dict['L_shared'] = True
#    four_quarters_stim_dict = four_quarters_dict.copy()
#    four_quarters_stim_dict.update(**stimulus_changes)
#    for a in four_quarters_dict['Agents']:
#        a.T_lockdown = 4
#    # Get consumption when the pandemic hits unemployment, but there is no loss to utility
#    t0 = time()
#    C_four_quarters, X_four_quarters, Z_four_quarters, cAll_four_quarters, Weight_four_quarters, Mrkv_four_quarters, U_four_quarters, ltAll_four_quarters, LT_by_inc_four_quarters = runExperiment(**four_quarters_dict)
#    C_four_quarters_stim, X_four_quarters_stim, Z_four_quarters_stim, cAll_four_quarters_stim, Weight_four_quarters_stim, Mrkv_four_quarters_stim, U_four_quarters_stim, ltAll_four_quarters_stim, LT_by_inc_four_quarters_stim = runExperiment(**four_quarters_stim_dict)
#    t1 = time()
#    print('Calculating consumption with long pandemic  took ' + mystr(t1-t0) + ' seconds.')
#    for a in four_quarters_dict['Agents']:
#        a.T_lockdown = None
    
for a in long_pandemic_dict['Agents']:
    a.ContUnempBenefits=True
# Get consumption when the pandemic hits unemployment, but there is no loss to utility
t0 = time()
C_long_pandemic_cont, X_long_pandemic_cont, Z_long_pandemic_cont, cAll_long_pandemic_cont, Weight_long_pandemic_cont, Mrkv_long_pandemic_cont, U_long_pandemic_cont, ltAll_long_pandemic_cont, LT_by_inc_long_pandemic_cont = runExperiment(**long_pandemic_stim_dict)
t1 = time()
print('Calculating consumption with long pandemic  took ' + mystr(t1-t0) + ' seconds.')
for a in long_pandemic_dict['Agents']:
    a.ContUnempBenefits=False

    
plt.plot((C_base)*1000)
plt.plot((C_long_pandemic)*1000)
plt.plot((C_long_pandemic_stim)*1000)
plt.plot((C_long_pandemic_cont)*1000)
plt.legend(["Baseline","Long pandemic","Long pandemic with stimulus","Long pandemic with CONTINUED unemployment"])
plt.xlabel('Quarter')
plt.ylabel('Consumption ($)')
plt.title('Deep Pandemic Consumption')
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
plt.tight_layout()
plt.show()


LT_long_pandemic_cont = np.sum(ltAll_long_pandemic_cont*Weight_base, axis=1) / np.sum(Weight_base, axis=1)   
LT_long_pandemic_stim = np.sum(ltAll_long_pandemic_stim*Weight_base, axis=1) / np.sum(Weight_base, axis=1)  
LT_long_pandemic = np.sum(ltAll_long_pandemic*Weight_base, axis=1) / np.sum(Weight_base, axis=1)  
LT_base = np.sum(ltAll_base*Weight_base, axis=1) / np.sum(Weight_base, axis=1)


plt.plot((LT_base)*1000)
plt.plot((LT_long_pandemic)*1000)
plt.plot((LT_long_pandemic_stim)*1000)
plt.plot((LT_long_pandemic_cont)*1000)
plt.legend(["Baseline","Long pandemic","Long pandemic with stimulus","Long pandemic with CONTINUED unemployment"])
plt.xlabel('Quarter')
plt.ylabel('Consumption ($)')
plt.title('Deep Pandemic Consumption')
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
plt.tight_layout()
plt.show()
    
import types
for a in long_pandemic_dict['Agents']:
    a.updateMrkvArray=types.MethodType(updateMrkvArray,a)
    a.hitWithPandemicShock=types.MethodType(hitWithPandemicShock,a)
    a.continueUnemploymentBenefits=types.MethodType(continueUnemploymentBenefits,a)
    a.switchToCounterfactualMode=types.MethodType(switchToCounterfactualMode,a)


    

 