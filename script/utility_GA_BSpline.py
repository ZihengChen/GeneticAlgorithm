from pylab import *
from scipy.interpolate import BSpline

def TranslateDNA(DNA):
    para_upper = 5*np.ones(12)
    para_lower = -5*np.ones(12)
    para = DNA*(para_upper-para_lower)+para_lower
    return para

def PlotGeneration(fitness,best_DNA,generation):
    fig, axes = plt.subplots(2, 1, sharex=False, 
                             gridspec_kw={'height_ratios':[2,1]},
                             figsize=(5,5))
    ax = axes[0]
    
    w     = np.linspace(0,1,100)
    truth = 1/2*np.sin(5*w)+np.cos(10*w)
    coeff = TranslateDNA(best_DNA)
    spl   = BSpline(np.linspace(-0.2,1.2,15), coeff, 2, extrapolate=False)
    guess = spl(w)
    ax.plot(w, truth, lw=3, color='b',linestyle='--',label="Ground Truth")
    ax.plot(w, guess,lw=2, color='r',label="Best Guess")
    
    ax.set_xlim(0.05,0.95)
    ax.set_ylim(-3,4)
    ax.legend()
    ax.grid()
    ax.set_title('Generation {}'.format(generation))
    
    ax = axes[1]
    ax.hist(fitness,bins=np.arange(0,200,4),histtype="stepfilled",lw=0,facecolor="gray",label='Fittness')
    #ax.legend()
    ax.grid()
    ax.set_xlabel("Generation Fitness")