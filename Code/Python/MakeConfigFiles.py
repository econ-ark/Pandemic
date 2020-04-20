'''
This script makes a bunch of yaml files for the configurator to use to produce
data for the interactive web dashboard.
'''
from GiveItAwayTools import makeConfigFile

makeConfigFile('uPfac_L', 0.8, 1.0, 21)
makeConfigFile('Lspell_real', 1.0, 5.0, 21)
makeConfigFile('Lspell_pcvd', 1.0, 5.0, 21)
makeConfigFile('Dspell_real', 1.0, 5.0, 21)
makeConfigFile('Dspell_pcvd', 1.0, 5.0, 21)
makeConfigFile('Unemp0', -1.0, 1.0, 21)
makeConfigFile('UnempD', -2.15, -0.15, 21)
makeConfigFile('UnempH', -2.30, -0.30, 21)
makeConfigFile('UnempC', -2.65, -0.65, 21)
makeConfigFile('UnempP', -0.2, 0.0, 21)
makeConfigFile('UnempA1', -0.03, 0.01, 21)
makeConfigFile('Deep0', -1.0, 1.0, 21)
makeConfigFile('DeepD', -2.50, -0.50, 21)
makeConfigFile('DeepH', -2.75, -0.75, 21)
makeConfigFile('DeepC', -3.20, -1.20, 21)
makeConfigFile('DeepP', -0.4, 0.0, 21)
makeConfigFile('DeepA1', -0.03, 0.01, 21)
makeConfigFile('T_ahead', 0, 3, 4, True)
makeConfigFile('UpdatePrb', 0.0, 1.0, 21)
makeConfigFile('StimMax', 0.2, 2.2, 21)
makeConfigFile('StimCut0', 4.0, 24.0, 21)
makeConfigFile('StimCut1', 19.0, 39.0, 21)
makeConfigFile('BonusUnemp', 3.0, 7.0, 21)
makeConfigFile('BonusDeep', 5.0, 10.0, 21)
