import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import genpareto as gprnd


np.random.seed(seed=2)

# random.seed(50)
# print(random.random()) 

### noise parameters ####

# alpha-stable parameter
k = 0.62

sigma_laminar = 2
sigma_turbulent = 0.1
obs_laminar = 0.2
laminar = False
laminar_durations = []
turbulent_durations = []
bias_correction = -1.3076202432497247e-07

theta = sigma_laminar/k
mean_xi = theta + sigma_laminar/(1-k)
dt = 0.01
time = np.arange(-200,400, dt)
noise = np.zeros_like(time)

for i in range(len(time)):
    if noise[i] == 0: 
        if laminar:  # compute the noise for the next laminar period
            n_laminar = np.ceil(gprnd.rvs(k,
                                          loc=theta,
                                          scale=sigma_laminar))
            # n_laminar = np.ceil(gprnd.rvs(k,
            #                               scale=sigma_laminar / k))
            laminar = False
            laminar_durations.append(n_laminar)
            if i + n_laminar > len(time):
                n_laminar = len(time) - i

            for jl in range(int(n_laminar)):
                noise[i+jl] = dt*obs_laminar

        else:  # compute the noise for the next turbulent period
            n_Brownian = np.ceil(mean_xi
                                 + mean_xi*(np.random.random()-0.5))
            laminar = True
            turbulent_durations.append(n_Brownian)
            if i + n_Brownian > len(time):
                n_Brownian = len(time) - i

            for jl in range(int(n_Brownian)):
                noise[i+jl] = (sigma_turbulent *
                               np.random.normal()-obs_laminar) * dt


noise -= bias_correction
#mask = noise> 0.002
#noise[mask] = 0.002

lam_stats = np.unique(laminar_durations, return_counts = True)
tur_stats = np.unique(turbulent_durations, return_counts = True)


fig, ax = plt.subplots(2,1, figsize = (80 / 25.4, 80/25.4))
# ax[0].scatter(*lam_stats[:20], facecolor = 'C0', edgecolor = 'k',
#               marker = 'o', s = 30, label = 'turbulent')
# rax = ax[0].twinx()
# ax[0].patch.set_visible(False)
# rax.set_zorder(-2)
# rax.bar(tur_stats[0], height=tur_stats[1],color='C1', lw=1, edgecolor='k',
#         label = 'laminar')
# rax.scatter([],[], facecolor = 'C0', edgecolor = 'k',
#               marker = 'o', s = 30, label = 'turbulent')
# ax[0].set_xlim((1,30))
# ax[0].yaxis.set_ticklabels(ax[0].get_yticks()/1000.0)
# rax.yaxis.set_ticklabels(rax.get_yticks()/ 1000)
# ax[0].set_xlabel('duration in simulation time steps')
# ax[0].set_ylabel('occurences in thousands')
# ax[0].annotate('highest laminar duration = %i' %(np.max(laminar_durations)),
#                (0, 1.1),
#                xycoords='axes fraction',
#                fontsize = 6)
# rax.legend(frameon = False, fontsize = 6)

ax[0].plot(time, noise, lw = 0.3)
#ax[0].set_xlabel('time')
ax[0].set_ylabel(r'$\xi_t$')
ax[0].xaxis.set_ticks_position('top')
ax[0].xaxis.set_label_position('top')
ax[0].ticklabel_format(axis='y', style='sci', scilimits = (-1,1))
t = ax[0].yaxis.get_offset_text()
t.set_x(-0.3)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_xlim(0,400)
ax[0].annotate('(a)', (0.9,0.1), xycoords = 'axes fraction')

#ax[0].yaxis.set_label_coords(0, 1.1)
#ax[0].yaxis.label.set_rotation()
ax[1].yaxis.set_ticks_position('right')
ax[1].yaxis.set_label_position('right')
ax[1].plot(time, np.cumsum(noise) * dt)
ax[1].set_xlabel('time [y]')
ax[1].set_ylabel(r'$\int_{0}^{t}\xi_s ds$')
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].set_xlim(0,400)
ax[1].annotate('(b)', (0.02,0.9), xycoords = 'axes fraction')
#ax[1].yaxis.set_label_coords(0, 1.05)
#ax[1].yaxis.label.set_rotation(0)
fig.subplots_adjust(wspace = 0.2,
                    top = 0.9,
                    left = 0.2,
                    bottom = 0.15,
                    right = 0.72)
fig.savefig('fig03.png', dpi = 300)
fig.savefig('fig03.pdf')
