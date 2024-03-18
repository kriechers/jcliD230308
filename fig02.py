import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rc
import tol_colors as tc

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


#################################################################
# set matplotlib parameters                                     #
#################################################################

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif',
                     'text.usetex': True,
                     'font.serif': ['Computer Modern'],
                     'font.size': 12,
                     'axes.labelsize': 10,
                     'axes.titlesize': 12,
                     'figure.titlesize': 14})
#plt.rcParams.update({'text.usetex': True})

plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}' +
                               r'\usepackage{xcolor}' +
                               r'\usepackage{textcomp}'))

colors = {'theta': tc.tol_cset('muted')[5],
          'PIP' : tc.tol_cset('muted')[7],
          'PaTh' : tc.tol_cset('muted')[3],
          'benthic': tc.tol_cset('muted')[4],
          'd18o': tc.tol_cset('muted')[0]}

width = 165 / 25.4
height = 100 / 25.4
fig, ax = plt.subplots(1,2,figsize=(width, height))
fig.subplots_adjust(top = 0.94,
                    bottom = 0.1,
                    left = 0.01,
                    right = 0.92,
                    wspace = 0.1)

#################################################################
# INTERSTADIAL CONFIGURATION                                    #
#################################################################
ax[0].set_title('A) Interstadial Configuration')

#################################################################
# ATLANTIC OCEAN                                                #
#################################################################

ax[0].imshow([[0.2, 0.8]], cmap=plt.cm.Blues,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  0, 4],
          alpha=1,
          vmin=0,
          vmax=1)

ax[0].annotate(r'$T_{\mathrm{e}}, S_{\mathrm{e}}$', (0.5, 2),
            va='center', fontsize = 16)
ax[0].annotate(r'$T_{\mathrm{p}}, S_{\mathrm{p}}$', (9.5, 2),
            va='center', ha='right', color='whitesmoke',
            fontsize = 16)

ax[0].set_xlim(0,10)
ax[0].set_ylim(0,10)
ax[0].set_aspect('equal')
ax[0].axvline(5, color = 'k', lw = 0.8, ls = '--', ymax = 0.8)
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)

ax[0].spines['top'].set_position(('data', 8))
ax[0].spines['left'].set_bounds(low = 0, high = 8)
ax[0].spines['right'].set_bounds(low = 0, high = 8)


#################################################################
# GENERAL ATMOSPHERE                                            #
#################################################################

ax[0].imshow([[0.1, 0.3]], cmap=plt.cm.RdBu,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  4, 8],
          alpha=0.5,
          vmin=0,
          vmax=1)
# ax[0].annotate(r'$\theta_{0}$', (5, 7), ha='center', fontsize=16,
#             alpha=0.6)


#################################################################
# ARCTIC ATMOSPHERE                                             #
#################################################################

AAy = 6
AAx = 8

# ax.scatter([AAx], [AAy], s=3000, color=plt.cm.Blues(0.2), alpha=1,
#            edgecolor='C0')
ax[0].annotate(r'$\boldsymbol{\theta_{\mathrm{p}}}$', (AAx, AAy),
            ha='center',
            va='center',
            fontsize = 16,
            color = colors['theta'])


#################################################################
# EQUATORIAL ATMOSPHERE                                         #
#################################################################

EAy = 6
EAx = 2

#ax.scatter([EAx], [EAy], s=3000, color=plt.cm.RdBu(0.35), edgecolor='C3')
ax[0].annotate(r'$\boldsymbol{\theta_{\mathrm{e}}}$', (EAx, EAy),
            ha='center', va='center',
            fontsize = 16,
            color = colors['theta'])
# ax[0].annotate(r'$\eta$', (4.5, 5.5), ha='center', color=tc.tol_cset('muted')[0],
#             fontsize=16)


#################################################################
# annotations to on the right of the plto                       #
#################################################################

ax[0].annotate('Box 1', (0,8.15), ha = 'left',
            annotation_clip = False)
ax[0].annotate('Equator', (0,-0.1), ha = 'left', va = 'top', 
            annotation_clip = False)
ax[0].annotate('Box 2', (10, 8.15), ha = 'right',
            annotation_clip = False)
ax[0].annotate('High \n Latitudes', (10,-0.1), ha = 'right', va = 'top', 
            annotation_clip = False)

# ax[0].annotate('North \n Atlantic', (10.2, 2), rotation = -90,
#             ha = 'left', va = 'center', multialignment = 'center', 
#             annotation_clip = False, xycoords = 'data',
#             fontsize =10)

# ax[0].annotate('Northern Hemisphere \n Atmosphere', (10.2, 6),
#             rotation = -90,
#             ha = 'left', va = 'center', multialignment = 'center',
#             annotation_clip = False, xycoords = 'data',
#             fontsize = 10)




#################################################################
# SKETCH SOLAR DIFFERENTIAL HEATING                             #
#################################################################

def rotation(x, y, theta = 0):
    rot_x = np.cos(theta) * x - np.sin(theta) * y
    rot_y = np.sin(theta) * x + np.cos(theta) * y
    return rot_x, rot_y

xx = np.arange(0,1, 0.001)
yy = 0.2* np.sin(xx * 20)

xx, yy = rotation(xx, yy, theta = -np.pi / 2)


ax[0].plot(xx + 2, yy + 8.5, color='gold', lw=3, zorder = 11)
ax[0].plot(xx + 2, yy + 8.5, color='k', lw=3.5, zorder = 10)


ax[0].arrow(xx[-1] +2 , yy[-1] + 8.5, 0., -0.55,
         width=0.05,
         length_includes_head=True,
         head_width=0.4,
         head_length=0.45,
         color='gold',
         zorder = 11)
ax[0].arrow(xx[-1] +2 , yy[-1] + 8.5, 0., -0.58,
         width=0.07,
         length_includes_head=True,
         head_width=0.41,
         head_length=0.47,
         color='k', zorder = 10)


xx = np.arange(0,-1, -0.001)
yy = 0.2* np.sin(xx * 20)
xx, yy = rotation(xx, yy, theta = np.pi / 2)

ax[0].plot(xx + 8, yy + 8.5, color='gold', lw=1.5, zorder = 11)
ax[0].plot(xx + 8, yy + 8.5, color='k', lw=1.9, zorder = 10)
ax[0].arrow(xx[-1] + 8, yy[-1] + 8.5, 0, -0.3,
         width=0.03,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.2,
         color='gold',
         zorder = 11)
ax[0].arrow(xx[-1] + 8, yy[-1] + 8.5, 0, -0.33,
         width=0.04,
         length_includes_head=True,
         head_width=0.21,
         head_length=0.22,
         color='k', zorder = 10)

yy = np.arange(0, 1, 0.001)
xx = 0.1 * np.sin(10 * yy)

ax[0].plot(xx + 2.6, yy + 7.5, color = tc.tol_cset('muted')[0], lw = 3, zorder = 100)
ax[0].arrow(xx[-1]+2.6, yy[-1]+7.5,
         0.,0.4,
         width = 0.02,
         length_includes_head = True,
         head_width = 0.35,
         head_length = 0.35,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

yy = np.arange(0, 0.7, 0.001)
xx = 0.1 * np.sin(10 * yy)

ax[0].plot(xx + 7.4, yy + 7.5, color = tc.tol_cset('muted')[0], lw = 1.5, zorder = 100)
ax[0].arrow(xx[-1]+7.4, yy[-1]+7.5,
         0.,0.35,
         width = 0.02,
         length_includes_head = True,
         head_width = 0.2,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

ax[0].annotate('$Q_{\mathrm{e}}$', (2.3,9.2), ha='center', color='k',
            fontsize=14)
ax[0].annotate('$Q_{\mathrm{p}}$', (7.7,9.2), ha='center', color='k',
            fontsize=14)


#################################################################
# SKETCH AMTOSPHERIC HEAT DIFFUSION                             #
#################################################################

xx = np.arange(0, 2, 0.001)
yy = 0.05 * np.sin(15 * xx)

head_width = 0.15
width = 0.04

ax[0].plot(xx + 4, yy + EAy, color = tc.tol_cset('muted')[0], lw = 2, zorder = 100)
ax[0].arrow(xx[-1]+4, yy[-1]+EAy,
         0.5,0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)
ax[0].arrow(xx[0]+4, yy[0]+EAy,
         -0.5, 0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

ax[0].annotate(r'$\chi_{\theta}$', (5.6,5.5), ha='center', color='k',
            fontsize=14)



#################################################################
# SKETCH THE EQUATORIAL HEAT EXCHANGE                           #
#################################################################

yy = np.arange(3.7, 4.3, 0.001)
xx = 0.1 * np.sin(14*yy)
ax[0].plot(xx+1, yy, color='C3', lw=1.5)
#ax[0].plot(xx+2, yy, color='C3', lw=3)
ax[0].arrow(xx[0]+1, yy[0], 0, -0.3,
         width=0.02,
         length_includes_head=True,
         head_width=0.15,
         head_length=0.2,
         color='C3')
ax[0].arrow(xx[-1]+1, yy[-1], 0, 0.3,
         width=0.02,
         length_includes_head=True,
         head_width=0.15,
         head_length=0.2,
         color='C3')


ax[0].annotate(r'$\phi_{\mathrm{e}}$', (0.5,3.3), ha='center', color='C3',
            fontsize=14)



#################################################################
# SKETCH THE POLAR HEAT EXCHANGE                                #
#################################################################

yy = np.arange(3.5, 4.5, 0.001)
xx = 0.1 * np.sin(14*yy)

ax[0].plot(xx+9, yy, color='C3', lw=2, zorder = 30)
#ax[0].plot(xx+2, yy, color='C3', lw=3)
ax[0].arrow(xx[0]+9, yy[0], 0, -0.3,
         width=0.025,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.2,
         color='C3')
ax[0].arrow(xx[-1]+9, yy[-1], 0, 0.3,
         width=0.025,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.2,
         color='C3')

ax[0].annotate(r'$\phi_{\mathrm{p}}$', (9.5,4.5), ha='center', color='C3',
            fontsize=14, zorder = 30)



#################################################################
# SKETCH THE EQUATORIAL EVAPORATION                             #
#################################################################

yy = np.arange(3.8, 5, 0.001)
xx = 0.1 * np.sin(8*yy)
ax[0].plot(xx+3, yy, color='slategray', lw=1.)
ax[0].plot(xx+3.5, yy, color='slategray', lw=1.)
ax[0].plot(xx+4, yy, color='slategray', lw=1.)


ax[0].scatter(xx[-1]+3 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)
ax[0].scatter(xx[-1]+3.5 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)
ax[0].scatter(xx[-1]+4 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)


#################################################################
# DRAW CLOUDS                                                   #
#################################################################

ell1 = plt.Circle((4, 0.5+6), 0.2, color='slategray')
ell2 = plt.Circle((4.2,0.5+ 6.2), 0.15, color='slategray')
ell3 = plt.Circle((4.3,0.5+ 6.), 0.15, color='slategray')
ell4 = plt.Circle((4.4,0.5+ 6), 0.12, color='slategray')
ell5 = plt.Circle((4.4,0.5+ 6.3), 0.2, color='slategray')
ell6 = plt.Circle((4.6,0.5+ 6.), 0.2, color='slategray')

ax[0].add_artist(ell1)
ax[0].add_artist(ell2)
ax[0].add_artist(ell3)
ax[0].add_artist(ell4)
ax[0].add_artist(ell5)
ax[0].add_artist(ell6)


ell1 = plt.Circle((6, 7), 0.25, color='slategray')
ell2 = plt.Circle((6.2, 7.2), 0.20, color='slategray')
ell3 = plt.Circle((6.3, 7.), 0.20, color='slategray')
ell4 = plt.Circle((6.4, 7), 0.22, color='slategray')
ell5 = plt.Circle((6.4, 7.3), 0.25, color='slategray')
ell6 = plt.Circle((6.6, 7.), 0.25, color='slategray')
ell7 = plt.Circle((6.6, 7.3), 0.2, color='slategray')

ax[0].add_artist(ell1)
ax[0].add_artist(ell2)
ax[0].add_artist(ell3)
ax[0].add_artist(ell4)
ax[0].add_artist(ell5)
ax[0].add_artist(ell6)
ax[0].add_artist(ell7)

ell1 = plt.Circle((7.4, 6.8), 0.18, color='slategray')
ell2 = plt.Circle((7.2, 6.6), 0.20, color='slategray')
ell3 = plt.Circle((7.4, 6.5), 0.20, color='slategray')
ell4 = plt.Circle((7.2, 6.8), 0.22, color='slategray')
ell5 = plt.Circle((7.5, 6.6), 0.18, color='slategray')
ell6 = plt.Circle((7.4, 6.7), 0.25, color='slategray')
ell7 = plt.Circle((7.5, 6.7), 0.2, color='slategray')

ax[0].add_artist(ell1)
ax[0].add_artist(ell2)
ax[0].add_artist(ell3)
ax[0].add_artist(ell4)
ax[0].add_artist(ell5)
ax[0].add_artist(ell6)
ax[0].add_artist(ell7)

#################################################################
# DRAW RAIN                                                     #
#################################################################

yy = np.linspace(4,6.8,100)
xx = np.ones_like(yy)

ax[0].plot(6 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[0].plot(6.5 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[0].plot(6.25 * xx, yy, color = 'slategray',dashes=[3,4], lw = 0.5)
ax[0].plot(6.75 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.5)
ax[0].plot(7 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[0].plot(7.25 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.5)
ax[0].plot(7.5 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)

ax[0].annotate(r'$\sigma_{e}$', (2.8,5.6), fontsize = 12, color = 'slategray')
ax[0].annotate(r'$\sigma_{p}$', (7.6,7), fontsize = 12, color = 'slategray')

# ax[0].annotate('Sea Ice', (10.2, 2),
#             ha = 'left', va = 'center', 
#             annotation_clip = False, xycoords = 'data',
#             fontsize = 8)



#################################################################
# OCEANIC DIFFUSION                                             #
#################################################################

ax[0].plot([4, 6], [2,2], color = 'whitesmoke', ls = ':', lw = 1.5)
ax[0].arrow(4, 2, -0.4, 0.,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='w', zorder = 4)
ax[0].arrow(6, 2, 0.4, 0.,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='w', zorder = 4)
ax[0].annotate(r'$\chi_{T,S}$', (6,2.5), ha='center', color='whitesmoke',
            fontsize=14)

#################################################################
# AMOC                                                          #
#################################################################

xq = np.arange(2, 8, 0.001)
yq = 3.5 - 1/2000 * (xq-5) ** 6

ax[0].plot(xq, yq, color=colors['PaTh'], lw=4, zorder = 10)
ax[0].plot(xq, yq, color='k', lw=5, zorder = 9)
ax[0].arrow(xq[-1], yq[-1], 0.25, -0.2,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color=colors['PaTh'], zorder = 12)
ax[0].arrow(xq[-1], yq[-1], 0.28, -0.23,
         width=0.22,
         length_includes_head=True,
         head_width=0.33,
         head_length=0.43,
         color='k', zorder = 11)



xq = np.arange(2, 8, 0.001)
yq = .5 + 1/2000 * (xq-5) ** 6


ax[0].plot(xq, yq, color=colors['PaTh'], lw=4, zorder = 10)
ax[0].plot(xq, yq, color='k', lw=5, zorder = 9)
ax[0].arrow(xq[0], yq[0], -0.25, 0.2,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color=colors['PaTh'], zorder = 12)
ax[0].arrow(xq[0], yq[0], -0.28, 0.23,
         width=0.22,
         length_includes_head=True,
         head_width=0.33,
         head_length=0.44,
         color='k', zorder = 11)

ax[0].annotate('$\psi$', (4, 1), va='center', ha='center', fontsize=16,
               color='k')


#################################################################
# STADIAL STATE                                            #
#################################################################

ax[1].set_title('B) Stadial Configuration')

#################################################################
# ATLANTIC OCEAN                                                #
#################################################################

ax[1].imshow([[0.2, 0.5]], cmap=plt.cm.Blues,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  0, 4],
          alpha=1,
          vmin=0,
          vmax=1)

ax[1].annotate(r'$T_{\mathrm{e}}, S_{\mathrm{e}}$', (0.5, 2),
            va='center', fontsize = 16)
ax[1].annotate(r'$T_{\mathrm{p}}, S_{\mathrm{p}}$', (9.5, 2),
            va='center', ha='right', color='whitesmoke',
            fontsize = 16)

ax[1].set_xlim(0,10)
ax[1].set_ylim(0,10)
ax[1].set_aspect('equal')
ax[1].axvline(5, color = 'k', lw = 0.8, ls = '--', ymax = 0.8)
ax[1].xaxis.set_visible(False)
ax[1].yaxis.set_visible(False)

ax[1].spines['top'].set_position(('data', 8))
ax[1].spines['left'].set_bounds(low = 0, high = 8)
ax[1].spines['right'].set_bounds(low = 0, high = 8)


#################################################################
# GENERAL ATMOSPHERE                                            #
#################################################################

ax[1].imshow([[0.1, 0.5]], cmap=plt.cm.RdBu,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  4, 8],
          alpha=0.5,
          vmin=0,
          vmax=1)
# ax[1].annotate(r'$\theta_{0}$', (5, 7), ha='center', fontsize=16,
#             alpha=0.6)


#################################################################
# ARCTIC ATMOSPHERE                                             #
#################################################################

AAy = 6
AAx = 8

# ax.scatter([AAx], [AAy], s=3000, color=plt.cm.Blues(0.2), alpha=1,
#            edgecolor='C0')
ax[1].annotate(r'$\boldsymbol{\theta_{\mathrm{p}}}$', (AAx, AAy),
            ha='center',
            va='center',
            fontsize = 16,
            color = colors['theta'])


#################################################################
# EQUATORIAL ATMOSPHERE                                         #
#################################################################

EAy = 6
EAx = 2

#ax.scatter([EAx], [EAy], s=3000, color=plt.cm.RdBu(0.35), edgecolor='C3')
ax[1].annotate(r'$\boldsymbol{\theta_{\mathrm{e}}}$', (EAx, EAy),
            ha='center', va='center',
            fontsize = 16,
            color = colors['theta'])
# ax[1].annotate(r'$\eta$', (4.5, 5.5), ha='center', color=tc.tol_cset('muted')[0],
#             fontsize=16)


#################################################################
# annotations to on the right of the plto                       #
#################################################################

ax[1].annotate('Box 1', (0,8.15), ha = 'left',
            annotation_clip = False)
ax[1].annotate('Equator', (0,-0.1), ha = 'left', va = 'top', 
            annotation_clip = False)
ax[1].annotate('Box 2', (10, 8.15), ha = 'right',
            annotation_clip = False)
ax[1].annotate('High \n Latitudes', (10,-0.1), ha = 'right', va = 'top', 
            annotation_clip = False)

ax[1].annotate('North \n Atlantic', (10.2, 2), rotation = -90,
            ha = 'left', va = 'center', multialignment = 'center', 
            annotation_clip = False, xycoords = 'data',
            fontsize =10)

ax[1].annotate('Northern Hemisphere \n Atmosphere', (10.2, 6),
            rotation = -90,
            ha = 'left', va = 'center', multialignment = 'center',
            annotation_clip = False, xycoords = 'data',
            fontsize = 10)




#################################################################
# SKETCH SOLAR DIFFERENTIAL HEATING                             #
#################################################################

def rotation(x, y, theta = 0):
    rot_x = np.cos(theta) * x - np.sin(theta) * y
    rot_y = np.sin(theta) * x + np.cos(theta) * y
    return rot_x, rot_y

xx = np.arange(0,1, 0.001)
yy = 0.2* np.sin(xx * 20)

xx, yy = rotation(xx, yy, theta = -np.pi / 2)


ax[1].plot(xx + 2, yy + 8.5, color='gold', lw=3, zorder = 11)
ax[1].plot(xx + 2, yy + 8.5, color='k', lw=3.5, zorder = 10)


ax[1].arrow(xx[-1] +2 , yy[-1] + 8.5, 0., -0.55,
         width=0.05,
         length_includes_head=True,
         head_width=0.4,
         head_length=0.45,
         color='gold',
         zorder = 11)
ax[1].arrow(xx[-1] +2 , yy[-1] + 8.5, 0., -0.58,
         width=0.07,
         length_includes_head=True,
         head_width=0.41,
         head_length=0.47,
         color='k', zorder = 10)


xx = np.arange(0,-1, -0.001)
yy = 0.2* np.sin(xx * 20)
xx, yy = rotation(xx, yy, theta = np.pi / 2)

ax[1].plot(xx + 8, yy + 8.5, color='gold', lw=1.5, zorder = 11)
ax[1].plot(xx + 8, yy + 8.5, color='k', lw=1.9, zorder = 10)
ax[1].arrow(xx[-1] + 8, yy[-1] + 8.5, 0, -0.3,
         width=0.03,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.2,
         color='gold',
         zorder = 11)
ax[1].arrow(xx[-1] + 8, yy[-1] + 8.5, 0, -0.33,
         width=0.04,
         length_includes_head=True,
         head_width=0.21,
         head_length=0.22,
         color='k', zorder = 10)

yy = np.arange(0, 1, 0.001)
xx = 0.1 * np.sin(10 * yy)

ax[1].plot(xx + 2.6, yy + 7.5, color = tc.tol_cset('muted')[0], lw = 3, zorder = 100)
ax[1].arrow(xx[-1]+2.6, yy[-1]+7.5,
         0.,0.4,
         width = 0.02,
         length_includes_head = True,
         head_width = 0.35,
         head_length = 0.35,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

yy = np.arange(0, 0.7, 0.001)
xx = 0.1 * np.sin(10 * yy)

ax[1].plot(xx + 7.4, yy + 7.5, color = tc.tol_cset('muted')[0], lw = 1.5, zorder = 100)
ax[1].arrow(xx[-1]+7.4, yy[-1]+7.5,
         0.,0.35,
         width = 0.02,
         length_includes_head = True,
         head_width = 0.2,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

ax[1].annotate('$Q_{\mathrm{e}}$', (2.3,9.2), ha='center', color='k',
            fontsize=14)
ax[1].annotate('$Q_{\mathrm{p}}$', (7.7,9.2), ha='center', color='k',
            fontsize=14)


#################################################################
# SKETCH AMTOSPHERIC HEAT DIFFUSION                             #
#################################################################

xx = np.arange(0, 2, 0.001)
yy = 0.05 * np.sin(15 * xx)

head_width = 0.15
width = 0.04

ax[1].plot(xx + 4, yy + EAy, color = tc.tol_cset('muted')[0], lw = 2, zorder = 100)
ax[1].arrow(xx[-1]+4, yy[-1]+EAy,
         0.5,0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)
ax[1].arrow(xx[0]+4, yy[0]+EAy,
         -0.5, 0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = tc.tol_cset('muted')[0],
         zorder = 100)

ax[1].annotate(r'$\chi_{\theta}$', (5.6,5.5), ha='center', color='k',
            fontsize=14)



#################################################################
# SKETCH THE EQUATORIAL HEAT EXCHANGE                           #
#################################################################

yy = np.arange(3.7, 4.3, 0.001)
xx = 0.1 * np.sin(14*yy)
ax[1].plot(xx+1, yy, color='C3', lw=1.5)
#ax[1].plot(xx+2, yy, color='C3', lw=3)
ax[1].arrow(xx[0]+1, yy[0], 0, -0.3,
         width=0.02,
         length_includes_head=True,
         head_width=0.15,
         head_length=0.2,
         color='C3')
ax[1].arrow(xx[-1]+1, yy[-1], 0, 0.3,
         width=0.02,
         length_includes_head=True,
         head_width=0.15,
         head_length=0.2,
         color='C3')


ax[1].annotate(r'$\phi_{\mathrm{e}}$', (0.5,3.3), ha='center', color='C3',
            fontsize=14)



#################################################################
# SKETCH THE POLAR HEAT EXCHANGE                                #
#################################################################

yy = np.arange(3.7, 4.3, 0.001)
xx = 0.1 * np.sin(14*yy)

ax[1].plot(xx+9, yy, color='C3', lw=1, zorder = 30)
#ax[1].plot(xx+2, yy, color='C3', lw=3)
ax[1].arrow(xx[0]+9, yy[0], 0, -0.12,
         width=0.02,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.08,
         color='C3')
ax[1].arrow(xx[-1]+9, yy[-1], 0, 0.12,
         width=0.02,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.08,
         color='C3')

ax[1].annotate(r'$\phi_{\mathrm{p}}$', (9.5,4.5), ha='center', color='C3',
            fontsize=14, zorder = 30)



#################################################################
# SKETCH THE EQUATORIAL EVAPORATION                             #
#################################################################

yy = np.arange(3.8, 5, 0.001)
xx = 0.1 * np.sin(8*yy)
ax[1].plot(xx+3, yy, color='slategray', lw=1.)
ax[1].plot(xx+3.5, yy, color='slategray', lw=1.)
ax[1].plot(xx+4, yy, color='slategray', lw=1.)


ax[1].scatter(xx[-1]+3 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)
ax[1].scatter(xx[-1]+3.5 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)
ax[1].scatter(xx[-1]+4 , yy[-1] + 0.02, color = 'slategray', marker = (3,0,0),
           s = 10)


#################################################################
# DRAW CLOUDS                                                   #
#################################################################

ell1 = plt.Circle((4, 0.5+6), 0.2, color='slategray')
ell2 = plt.Circle((4.2,0.5+ 6.2), 0.15, color='slategray')
ell3 = plt.Circle((4.3,0.5+ 6.), 0.15, color='slategray')
ell4 = plt.Circle((4.4,0.5+ 6), 0.12, color='slategray')
ell5 = plt.Circle((4.4,0.5+ 6.3), 0.2, color='slategray')
ell6 = plt.Circle((4.6,0.5+ 6.), 0.2, color='slategray')

ax[1].add_artist(ell1)
ax[1].add_artist(ell2)
ax[1].add_artist(ell3)
ax[1].add_artist(ell4)
ax[1].add_artist(ell5)
ax[1].add_artist(ell6)


ell1 = plt.Circle((6, 7), 0.25, color='slategray')
ell2 = plt.Circle((6.2, 7.2), 0.20, color='slategray')
ell3 = plt.Circle((6.3, 7.), 0.20, color='slategray')
ell4 = plt.Circle((6.4, 7), 0.22, color='slategray')
ell5 = plt.Circle((6.4, 7.3), 0.25, color='slategray')
ell6 = plt.Circle((6.6, 7.), 0.25, color='slategray')
ell7 = plt.Circle((6.6, 7.3), 0.2, color='slategray')

ax[1].add_artist(ell1)
ax[1].add_artist(ell2)
ax[1].add_artist(ell3)
ax[1].add_artist(ell4)
ax[1].add_artist(ell5)
ax[1].add_artist(ell6)
ax[1].add_artist(ell7)

ell1 = plt.Circle((7.4, 6.8), 0.18, color='slategray')
ell2 = plt.Circle((7.2, 6.6), 0.20, color='slategray')
ell3 = plt.Circle((7.4, 6.5), 0.20, color='slategray')
ell4 = plt.Circle((7.2, 6.8), 0.22, color='slategray')
ell5 = plt.Circle((7.5, 6.6), 0.18, color='slategray')
ell6 = plt.Circle((7.4, 6.7), 0.25, color='slategray')
ell7 = plt.Circle((7.5, 6.7), 0.2, color='slategray')

ax[1].add_artist(ell1)
ax[1].add_artist(ell2)
ax[1].add_artist(ell3)
ax[1].add_artist(ell4)
ax[1].add_artist(ell5)
ax[1].add_artist(ell6)
ax[1].add_artist(ell7)

#################################################################
# DRAW RAIN                                                     #
#################################################################

yy = np.linspace(4,6.8,100)
xx = np.ones_like(yy)

ax[1].plot(6 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[1].plot(6.5 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[1].plot(6.25 * xx, yy, color = 'slategray',dashes=[3,4], lw = 0.5)
ax[1].plot(6.75 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.5)
ax[1].plot(7 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)
ax[1].plot(7.25 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.5)
ax[1].plot(7.5 * xx, yy, color = 'slategray', dashes=[3,4], lw = 0.8)

ax[1].annotate(r'$\sigma_{e}$', (2.8,5.6), fontsize = 12, color = 'slategray')
ax[1].annotate(r'$\sigma_{p}$', (7.6,7), fontsize = 12, color = 'slategray')

# ax[1].annotate('Sea Ice', (10.2, 2),
#             ha = 'left', va = 'center', 
#             annotation_clip = False, xycoords = 'data',
#             fontsize = 8)



#################################################################
# OCEANIC DIFFUSION                                             #
#################################################################

ax[1].plot([4, 6], [2,2], color = 'whitesmoke', ls = ':', lw = 1.5)
ax[1].arrow(4, 2, -0.4, 0.,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='w', zorder = 4)
ax[1].arrow(6, 2, 0.4, 0.,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='w', zorder = 4)
ax[1].annotate(r'$\chi_{T,S}$', (6,2.5), ha='center', color='whitesmoke',
            fontsize=14)

#################################################################
# AMOC                                                          #
#################################################################

xq = np.arange(2, 8, 0.001)
yq = 3.5 - 1/2000 * (xq-5) ** 6

ax[1].plot(xq, yq, color=colors['PaTh'], lw=2, zorder = 10)
ax[1].plot(xq, yq, color='k', lw=2.5, zorder = 9)
ax[1].arrow(xq[0], yq[0], -0.25, -0.2,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color=colors['PaTh'], zorder = 12)
ax[1].arrow(xq[0], yq[0], -0.28, -0.23,
         width=0.22,
         length_includes_head=True,
         head_width=0.33,
         head_length=0.43,
         color='k', zorder = 11)



xq = np.arange(2, 8, 0.001)
yq = .5 + 1/2000 * (xq-5) ** 6


ax[1].plot(xq, yq, color=colors['PaTh'], lw=2, zorder = 10)
ax[1].plot(xq, yq, color='k', lw=2.5, zorder = 9)
ax[1].arrow(xq[-1], yq[-1], 0.25, 0.2,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color=colors['PaTh'], zorder = 12)
ax[1].arrow(xq[-1], yq[-1], 0.28, 0.23,
         width=0.22,
         length_includes_head=True,
         head_width=0.33,
         head_length=0.43,
         color='k', zorder = 11)

ax[1].annotate('$\psi$', (4, 1), va='center', ha='center', fontsize=16,
               color='k', zorder = 11)





#################################################################
# SEA ICE                                                       #
#################################################################

ax[1].fill_between(np.array([7.5, 10]),
                np.array([3.8, 3.8]),
                np.array([4.1, 4.1]),
                color=colors['PIP'], zorder = 20)
ax[1].plot(np.array([7.5, 10]),
             np.array([3.8, 3.8]),
             color = 'k', zorder = 20, lw = 1)
ax[1].plot(np.array([7.5, 10]),
             np.array([4.1, 4.1]),
             color = 'k',zorder = 20, lw = 1)
ax[1].plot(np.array([7.5, 7.5]),
             np.array([3.8, 4.1]),
             color = 'k', zorder = 20, lw = 1)

ax[1].annotate('$I$', (8, 4.5), color='k')


fig.savefig('fig02.png', dpi = 300)
fig.savefig('fig02.pdf')

plt.close()
