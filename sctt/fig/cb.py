import numpy as np
import matplotlib.pyplot as plt


# Example data

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot((0,10),(0,0), 'black', lw=2)
plt.plot((10,10),(0,4), 'black', lw=2)
plt.plot((0,10),(4,4), 'black', lw=2)

plt.plot((10.5,25.5),(0,0), 'black', lw=2)
plt.plot((10.5,10.5),(0,4), 'black', lw=2)
plt.plot((10.5,25.5),(4,4), 'black', lw=2)

plt.plot((0,0),(-0.5, 4.5), 'k--')
plt.plot((25.5,25.5),(-0.5, 4.5), 'k--')

plt.plot((0,25.5), (2,2), 'k')
plt.plot((0,25.5), (2.4,2.4), 'k')
plt.plot((0,25.5), (2.2,2.2), 'k')
plt.plot((0,25.5), (1.8,1.8), 'k')
plt.plot((0,25.5), (1.6,1.6), 'k')

plt.plot((0,0),(-1, -3.5), 'k')
plt.plot((25.5,25.5),(-1, -3.5), 'k')
plt.plot((10.25,10.25),(-1, -3.5), 'k')

plt.annotate('', xy=(0, -3), xycoords='data',
    xytext=(10.25, -3), textcoords='data',
    arrowprops={'arrowstyle': '<->'})

plt.annotate('', xy=(10.25, -3), xycoords='data',
    xytext=(25.5, -3), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
# 
plt.annotate(r'$L_{\uparrow}$', 
             xy=(4.8, -2.5), xycoords='data', xytext=(5, 0), textcoords='offset points')

plt.annotate(r'$L_{\downarrow}$', 
             xy=(17.25, -2.5), xycoords='data', xytext=(5, 0), textcoords='offset points')

plt.axis('off')

# plt.text(-2, 5, r'$L_{\uparrow}$')





# plt.xlabel(r'\textbf{time} (s)')
# plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
# plt.title(r"\TeX\ is Number "
#           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
#           fontsize=16, color='gray')
# # Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)
# 
# plt.savefig('tex_demo')
plt.show()
