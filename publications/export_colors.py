#!/usr/bin/env python3

from amata.constants import COLORS

with open('./colordef.tex', 'w') as ofile:
    for name, color in COLORS.items():
        colorname = '{' + name.lower() + '}'
        ofile.write('\\definecolor{colorname}'.format(colorname=colorname))
        rgb = map(lambda x: x/255., color)
        rgb = [round(x, 3) for x in rgb]
        rgb = '{' + str(rgb)[1:-1] + '}'
        ofile.write('{method}{rgb}\n'.format(method='{rgb}', rgb=rgb))
    ofile.write('\n')
    ofile.write('\\newcommand{\\airp}[1]{\\textcolor{#1}{\\textsc{#1}}}\n')
