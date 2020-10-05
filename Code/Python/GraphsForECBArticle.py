# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:47:24 2020

@author: edmun
"""
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
font =  font_manager.FontProperties(family='Ariel',
                                   weight='normal',
                                   style='normal', size=9)

big_dpi = 1000

plt.figure(figsize=cm2inch(12.8, 9.0))
plt.plot(LT_base_all*AggregationFactor,dashes=dashes1)
plt.plot(LT_pan_all*AggregationFactor,dashes=dashes2)
plt.plot(LT_both_all*AggregationFactor,dashes=dashes3)
plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"],prop=font)
plt.xlabel('Quarter',fontname="Ariel",fontsize=9)
plt.ylabel('Aggregate labour and transfer income (billion $)',fontname="Ariel",fontsize=9)
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
#plt.title('Aggregate household income under alternate scenarios',fontname="Ariel",fontsize=9)
plt.tight_layout()
plt.savefig(figs_dir + 'AggLT_high_res.png', dpi=big_dpi)
plt.show()

plt.figure(figsize=cm2inch(12.8, 9.0))
plt.plot(C_base*AggregationFactor,dashes=dashes1)
plt.plot(C_pan*AggregationFactor,dashes=dashes2)
plt.plot(C_both*AggregationFactor,dashes=dashes3)
plt.legend(["Baseline","Pandemic, no policy","Pandemic, CARES Act"],prop=font)
plt.xlabel('Quarter',fontname="Ariel",fontsize=9)
plt.ylabel('Aggregate quarterly consumption (billion $)',fontname="Ariel",fontsize=9)
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
#plt.title('Aggregate consumption under alternate scenarios',fontname="Ariel",fontsize=9)
plt.tight_layout()
plt.savefig(figs_dir + 'AggConResp_examples_high_res.png', dpi=big_dpi)
plt.show()

plt.figure(figsize=cm2inch(12.8, 9.0))
plt.plot(X_pan[0,:]*1000,'b',dashes=dashes1)
plt.plot(X_pan[1,:]*1000,'g',dashes=dashes2)
plt.plot(X_pan[2,:]*1000,'r',dashes=dashes3)
plt.plot(X_alt[0,:]*1000,':b')
plt.plot(X_alt[1,:]*1000,':g')
plt.plot(X_alt[2,:]*1000,':r')
plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"],prop=font)
plt.xlabel('Quarter',fontname="Ariel",fontsize=9)
plt.ylabel('Average quarterly consumption ($)',fontname="Ariel",fontsize=9)
plt.title('Consumption among working age population (no CARES Act)',fontname="Ariel",fontsize=9)
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
bottom, top = plt.ylim() # want graph graphs with and without stimulus to have same ylim
plt.ylim(bottom, top+500)
plt.tight_layout()
plt.savefig(figs_dir + 'ConRespByEmpStateNoStim_high_res.png', dpi=big_dpi)
plt.show()

plt.figure(figsize=cm2inch(12.8, 9.0))
plt.plot(X_both[0,:]*1000,'b',dashes=dashes1)
plt.plot(X_both[1,:]*1000,'g',dashes=dashes2)
plt.plot(X_both[2,:]*1000,'r',dashes=dashes3)
plt.plot(X_alt[0,:]*1000,':b')
plt.plot(X_alt[1,:]*1000,':g')
plt.plot(X_alt[2,:]*1000,':r')
plt.legend(["Employed in Q2 2020","Unemployed in Q2 2020","Deeply unemp in Q2 2020"],prop=font)
plt.xlabel('Quarter',fontname="Ariel",fontsize=9)
plt.ylabel('Average quarterly consumption ($)',fontname="Ariel",fontsize=9)
plt.title('Consumption among working age population (CARES Act)',fontname="Ariel",fontsize=9)
plt.xticks(ticks=range(T_sim), labels=quarter_labels)
plt.ylim(bottom, top+500)
plt.tight_layout()
plt.savefig(figs_dir + 'ConRespByEmpStateWStim_high_res.png', dpi=big_dpi)
plt.show()

#Suggested plot for Oslo presentation
#plt.figure(figsize=cm2inch(12.8, 9.0))
#plt.plot(X_both[0,:]*1000-X_pan[0,:]*1000,'b',dashes=dashes1)
#plt.plot(X_both[1,:]*1000-X_pan[1,:]*1000,'g',dashes=dashes2)
#plt.plot(X_both[2,:]*1000-X_pan[2,:]*1000,'r',dashes=dashes3)
#plt.legend(["Employed after pandemic","Unemployed after pandemic","Deeply unemp after pandemic"],prop=font)
#plt.xlabel('Quarter',fontname="Ariel",fontsize=9)
#plt.ylabel('Average quarterly consumption difference ($)',fontname="Ariel",fontsize=9)
#plt.title('Consumption among working age population (CARES Act impact)',fontname="Ariel",fontsize=9)
#plt.xticks(ticks=range(T_sim), labels=quarter_labels)
##plt.ylim(bottom, top+500)
#plt.tight_layout()
#plt.savefig(figs_dir + 'ConRespByEmpStateDiff_high_res.png', dpi=big_dpi)
#plt.show()
