$ENV{'BIBINPUTS'}='./LaTeX//:' . $ENV{'BIBINPUTS'};
$bibtex_use=2;
@default_files = {'ConsumptionResponse.tex'};
latexmk
