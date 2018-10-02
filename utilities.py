import numpy as np


def calculate_poling_period(lamr=0, lamg=0, lamb=0, nr=None, ng=None, nb=None, order=1, **kwargs):
    """
    Function to calculate the poling period of a specific process. To ensure energy conservation, specify only 2
    wavelengths (in meter) and leave the free one to 0

    :param lamr:
    :param lamg:
    :param lamb:
    :param nr:
    :param ng:
    :param nb:
    :param order:
    :param kwargs:
    :return:
    """
    propagation_type = kwargs.get("propagation_type", "copropagation")
    if (lamb == 0):
        lamb = 1. / (1. / abs(lamg) + 1. / abs(lamr))
    if (lamg == 0):
        lamg = 1. / (1. / abs(lamb) - 1. / abs(lamr))
    if (lamr == 0):
        lamr = 1. / (1. / abs(lamb) - 1. / abs(lamg))
    if propagation_type.lower() == "copropagation":
        Lambda = order / (nb(abs(lamb) * 1e6) / lamb - ng(abs(lamg) * 1e6) / lamg - nr(abs(lamr) * 1e6) / lamr)
    elif propagation_type.lower() == "counterpropagation":
        Lambda = order / (nb(abs(lamb) * 1e6) / lamb - ng(abs(lamg) * 1e6) / lamg + nr(abs(lamr) * 1e6) / lamr)
    else:
        raise ValueError("Don't know " + propagation_type)
    return lamr, lamg, lamb, Lambda


def deltabeta(lam1, lam2, free_param, nr, ng, nb, poling=np.infty):
    if free_param == "b":
        wlr, wlg = lam1, lam2
        wlb = (wlr ** -1 + wlg ** -1) ** -1
    elif free_param == "g":
        wlr, wlb = lam1, lam2
        wlg = (wlb ** -1 - wlr ** -1) ** -1
    elif free_param == "r":
        wlg, wlb = lam1, lam2
        wlr = (wlb ** -1 - wlg ** -1) ** -1
    else:
        raise ValueError("Wrong label for free parameter")
    return wlr, wlg, wlb, 2 * np.pi * (nb(wlb) / wlb - ng(wlg) / wlg - nr(wlr) / wlr - 1 / poling)
