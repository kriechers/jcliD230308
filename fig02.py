import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rc
from plot_functions import make_patch_spines_invisible


#################################################################
# set matplotlib parameters                                     #
#################################################################

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif',
                     'font.size': 10,
                     'axes.labelsize': 12,
                     'axes.titlesize': 14,
                     'figure.titlesize': 14})

plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}' +
                               r'\usepackage{xcolor}' +
                               r'\usepackage{textcomp}'))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


width = 87 / 25.4
height = 60 / 25.4
fig, ax = plt.subplots(figsize=(width, height))

#################################################################
# ATLANTIC OCEAN                                                #
#################################################################

ax.imshow([[0.2, 1]], cmap=plt.cm.Blues,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  0, 2],
          alpha=1,
          vmin=0,
          vmax=1)

ax.annotate(r'$T_{\mathrm{e}}, S_{\mathrm{e}}$', (0.5, 1),
            va='center')
ax.annotate(r'$T_{\mathrm{p}}, S_{\mathrm{p}}$', (9.5, 1),
            va='center', ha='right', color='whitesmoke')

#################################################################
# AMOC                                                          #
#################################################################

yq = np.arange(0.4, 1.6, 0.001)
xq = 2 + 2 * (yq-1) ** 2 + 6 * (yq-1) ** 4

ax.plot(xq, yq, color='m', lw=3)
ax.arrow(xq[-1], yq[-1], 0.4, 0,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='m')

yq = np.arange(0.4, 1.6, 0.001)
xq = 8 - 2 * (yq-1) ** 2 - 6 * (yq-1) ** 4

ax.plot(xq, yq, color='m', lw=3)
ax.arrow(xq[0], yq[0], -0.4, 0,
         width=0.2,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.4,
         color='m')

ax.annotate('$q$', (5, 1), va='center', ha='center', fontsize=16,
            color='m')

#################################################################
# GENERAL ATMOSPHERE                                            #
#################################################################

ax.imshow([[0.2, 0.6]], cmap=plt.cm.RdBu,
          interpolation='bicubic', aspect='auto',
          extent=[0, 10,
                  2, 8],
          alpha=0.5,
          vmin=0,
          vmax=1)
ax.annotate(r'$\theta_{0}$', (5, 7), ha='center', fontsize=16,
            alpha=0.6)

#################################################################
# SEA ICE                                                       #
#################################################################

ax.fill_between(np.array([7.5, 10]),
                np.array([1.9, 1.9]),
                np.array([2.2, 2.2]),
                color='C2')
ax.annotate('$I$', (9.5, 2.5), color='C2')


#################################################################
# ARCTIC ATMOSPHERE                                             #
#################################################################

AAy = 5
AAx = 8

ax.scatter([AAx], [AAy], s=3000, color=plt.cm.Blues(0.2), alpha=1,
           edgecolor='C0')
ax.annotate(r'$\theta_{\mathrm{p}}$', (AAx, AAy), ha='center', va='center')


#################################################################
# EQUATORIAL ATMOSPHERE                                         #
#################################################################

EAy = 5
EAx = 1.5

ax.scatter([EAx], [EAy], s=3000, color=plt.cm.RdBu(0.35), edgecolor='C3')
ax.annotate(r'$\theta_{\mathrm{e}}$', (EAx, EAy), ha='center', va='center')
ax.annotate(r'$\gamma$', (6., 3.5), ha='center', color='C3',
            fontsize=16)
ax.annotate(r'$\eta$', (4.5, 5.5), ha='center', color='salmon',
            fontsize=16)

#################################################################
# SKETCH THE EQUATORIAL HEAT EXCHANGE                           #
#################################################################

yy = np.arange(2.5, 3.8, 0.01)
xx = 0.3 * np.sin(10 * yy)
ax.plot(xx+1, yy, color='C3', lw=3)
ax.plot(xx+2, yy, color='C3', lw=3)
ax.arrow(xx[-1]+1, yy[-1], 0, 0.8,
         width=0.1,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.5,
         color='C3')

ax.arrow(xx[0]+1, yy[0], 0, -0.8,
         width=0.1,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.5,
         color='C3')

ax.arrow(xx[-1]+2, yy[-1], 0, 0.8,
         width=0.1,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.5,
         color='C3')

ax.arrow(xx[0]+2, yy[0], 0, -0.8,
         width=0.1,
         length_includes_head=True,
         head_width=0.3,
         head_length=0.5,
         color='C3')

#################################################################
# SKETCH THE POLAR HEAT EXCHANGE                                #
#################################################################

yy = np.arange(2.4, 4.2, 0.001)
xx = 0.2 * np.sin(14*yy)
ax.plot(xx+7, yy, color='C3', lw=1.5)
ax.arrow(xx[-1]+7, yy[-1], 0, 0.5,
         width=0.02,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.3,
         color='C3')

ax.arrow(xx[0]+7, yy[0], 0, -0.5,
         width=0.02,
         length_includes_head=True,
         head_width=0.2,
         head_length=0.3,
         color='C3')

yy = np.arange(2.5, 4, 0.001)
xx = 0.05 * np.sin(12 * yy)
ax.plot(xx+7.9, yy, color='C3', lw=0.8)
ax.arrow(xx[-1]+7.9, yy[-1], 0, 0.4,
         width=0.002,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.2,
         color='C3')

ax.arrow(xx[0]+7.9, yy[0], 0, -0.4,
         width=0.002,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.2,
         color='C3')

ax.plot(xx+8.6, yy, color='C3', lw=0.8)
ax.arrow(xx[-1]+8.6, yy[-1], 0, 0.4,
         width=0.002,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.2,
         color='C3')

ax.arrow(xx[0]+8.6, yy[0], 0, -0.4,
         width=0.002,
         length_includes_head=True,
         head_width=0.1,
         head_length=0.2,
         color='C3')

#################################################################
# SKETCH AMTOSPHERIC HEAT DIFFUSION                             #
#################################################################

yy = np.arange(1.5, 2, 0.001)
xx = 0.05 * np.sin(12 * yy)

head_width = 0.15
width = 0.04


def rotate(x, y, deg):
    xrot = np.cos(deg) * x - np.sin(deg) * y
    yrot = np.sin(deg) * x + np.cos(deg) * y
    return xrot, yrot


ax.plot(xx+8, yy+AAy, color='salmon', lw=2)
ax.arrow(xx[-1]+8, yy[-1]+AAy, 0, 0.5,
         width=width,
         length_includes_head=True,
         head_width=head_width,
         head_length=0.3,
         color='salmon')
ax.arrow(xx[0]+8, yy[0]+AAy, 0, -0.5,
         width=width,
         length_includes_head=True,
         head_width=head_width,
         head_length=0.3,
         color='salmon')

xx45, yy45 = rotate(xx, yy, 2*np.pi * 45/360)
ax.plot(xx45+8, yy45+AAy, color = 'salmon', lw = 2)
ax.arrow(xx45[-1]+8, yy45[-1]+AAy,
         -0.5/2/np.sqrt(0.5), 0.5/2/np.sqrt(0.5),
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')
ax.arrow(xx45[0]+8, yy45[0]+AAy,
         0.5 /2/ np.sqrt(0.5), -0.5/2 / np.sqrt(0.5),
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')

xx90, yy90 = rotate(xx, yy, 2*np.pi * 90/360)
ax.plot(xx90+8, yy90+AAy, color = 'salmon', lw = 2)
ax.arrow(xx90[-1]+8, yy90[-1]+AAy,
         -0.5,0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')
ax.arrow(xx90[0]+8, yy90[0]+AAy,
         0.5, 0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')


ax.plot(xx+EAx, yy+EAy, color = 'salmon', lw = 2)
ax.arrow(xx[-1]+EAx, yy[-1]+EAy, 0, 0.5,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')
ax.arrow(xx[0]+EAx, yy[0]+EAy,
         0, -0.5,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')

xx45, yy45 = rotate(xx, yy, -2*np.pi * 45/360)
ax.plot(xx45+EAx, yy45+EAy, color = 'salmon', lw = 2)
ax.arrow(xx45[-1]+EAx, yy45[-1]+EAy,
         0.5/2/np.sqrt(0.5), 0.5/2/np.sqrt(0.5),
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')
ax.arrow(xx45[0]+EAx, yy45[0]+EAy,
         -0.5 /2/ np.sqrt(0.5), -0.5/2 / np.sqrt(0.5),
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')

xx90, yy90 = rotate(xx, yy, -2*np.pi * 90/360)
ax.plot(xx90+EAx, yy90+EAy, color = 'salmon', lw = 2)
ax.arrow(xx90[-1]+EAx, yy90[-1]+EAy,
         0.5,0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')
ax.arrow(xx90[0]+EAx, yy90[0]+EAy,
         -0.5, 0,
         width = width,
         length_includes_head = True,
         head_width = head_width,
         head_length = 0.3,
         color = 'salmon')

ax.annotate('Equator', (0,8.1), ha = 'left',
            annotation_clip = False)
ax.annotate('North Pole', (10, 8.1), ha = 'right',
            annotation_clip = False)

ax.annotate('North \n Atlantic', (10.2, 1), rotation = -90,
            ha = 'left', va = 'center', multialignment = 'center', 
            annotation_clip = False, xycoords = 'data',
            fontsize =8)

ax.annotate('Northern Hemisphere \n Atmosphere', (10.2, 5),
            rotation = -90,
            ha = 'left', va = 'center', multialignment = 'center',
            annotation_clip = False, xycoords = 'data',
            fontsize = 8)

ax.annotate('Sea Ice', (10.2, 2),
            ha = 'left', va = 'center', 
            annotation_clip = False, xycoords = 'data',
            fontsize = 8)

ax.set_xlim(0,10)
ax.set_ylim(0,8)
ax.set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

fig.subplots_adjust(left = 0.05,
                    top = 0.99,
                    bottom = 0.,
                    right = 0.82)

fig.savefig('fig02.png', dpi = 300)
fig.savefig('fig02.pdf')
#plt.close()
