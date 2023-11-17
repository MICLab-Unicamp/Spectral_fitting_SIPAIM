"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import numpy as np

A_Ace  = (0.0, 0.0)
A_Ala  = (0.1, 1.8)
A_Asc  = (0.2, 1.8)
A_Asp  = (1.0, 2.0)
A_Cr   = (4.5, 10.5)
A_GABA = (1.0, 2.0)
A_Glc  = (1.0, 2.0)
A_Gln  = (3.0, 6.0)
A_Glu  = (6.0, 12.5)
A_Gly  = (0.2, 1.0)
A_GPC  = (0.5, 1.8)
A_GSH  = (1.5, 3.0)
A_Ins  = (4.0, 9.0)
A_Lac  = (0.2, 1.0)
A_MM   = (14.0, 17.0)
A_NAA  = (7.5, 17.0)
A_NAAG = (0.5, 2.5)
A_PCho = (0.2, 1.0)
A_PCr  = (3.0, 5.5)
A_PE   = (1.0, 2.0)
A_sIns = (0.2, 0.5)
A_Tau  = (3.0, 6.0)

alpha = (-15, 0)
freq = (-3, 3)
theta = (-np.pi/10, np.pi/10)

A_ranges = [A_Ace, A_Ala, A_Asc, A_Asp, A_Cr, A_GABA, A_Glc, A_Gln, A_Glu, A_Gly, A_GPC, A_GSH, A_Ins, A_Lac, A_MM, A_NAA, A_NAAG, A_PCho, A_PCr, A_PE, A_sIns,A_Tau]

param = {
        "A_Ace": 1.0,
        "A_Ala": 1.0,
        "A_Asc": 1.0,
        "A_Asp": 1.0,
        "A_Cr": 1.0,
        "A_GABA": 1.0,
        "A_Glc": 1.0,
        "A_Gln": 1.0,
        "A_Glu": 1.0,
        "A_Gly": 1.0,
        "A_GPC": 1.0,
        "A_GSH": 1.0,
        "A_Ins": 1.0,
        "A_Lac": 1.0,
        "A_MM": 1.0,
        "A_NAA": 1.0,
        "A_NAAG": 1.0,
        "A_PCho": 1.0,
        "A_PCr": 1.0,
        "A_PE": 1.0,
        "A_sIns": 1.0,
        "A_Tau": 1.0,
        "damping": 0.0,
        "freq_s": 0.0,
        "phase_s": 0.0,
        "SNR": 0.0,
    }

metab_list = ['Ace.txt', 'Ala.txt', 'Asc.txt', 'Asp.txt', 'Cr.txt', 'GABA.txt', 'Glc.txt', 'Gln.txt', 'Glu.txt',
                      'Gly.txt', 'GPC.txt', 'GSH.txt', 'Ins.txt', 'Lac.txt', 'Mac.txt', 'NAA.txt', 'NAAG.txt',
                      'PCho.txt', 'PCr.txt', 'PE.txt', 'sIns.txt', 'Tau.txt']