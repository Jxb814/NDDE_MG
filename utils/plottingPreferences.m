function plottingPreferences()
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
    N = 12;
    set(0,'DefaultLineLineWidth',1)
    
    set(0,'defaultAxesFontSize',N)
    set(0, 'defaultLegendFontSize', N)
    set(0, 'defaultColorbarFontSize', N);
    set(0, 'DefaultFigureRenderer', 'painters');
    set(0,'defaulttextinterpreter','latex')
end