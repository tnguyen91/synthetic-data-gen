import numpy as np


def fac_and_cf_sample(a1, a2):
    k = np.random.normal(0, 1)
    gpa0 = 2
    lsat0 = 3
    w_k_gpa = 4
    w_k_lsat = 5
    w_k_zfya = 6

    w_a_gpa = 7 * np.ones(10)
    w_a_lsat = 8 * np.ones(10)
    w_a_zfya = 9 * np.ones(10)

    sigma_gpa_2 = 10

    mu_f = gpa0 + (w_k_gpa * k) + np.dot(a1, w_a_gpa)
    mu_cf = gpa0 + (w_k_gpa * k) + np.dot(a2, w_a_gpa)

    gpa_f = np.random.normal(mu_f, np.sqrt(sigma_gpa_2))
    gpa_cf = np.random.normal(mu_cf, np.sqrt(sigma_gpa_2))

    lsat_f = np.random.poisson(np.exp(lsat0 + w_k_lsat * k + np.dot(a1, w_a_lsat)))
    lsat_cf = np.random.poisson(np.exp(lsat0 + w_k_lsat * k + np.dot(a2, w_a_lsat)))

    s_f = np.concatenate((a1[0:8], np.array(gpa_f).reshape(1), np.array(lsat_f).reshape(1), a1[8:]))
    s_cf = np.concatenate((a2[0:8], np.array(gpa_cf).reshape(1), np.array(lsat_cf).reshape(1), a2[8:]))

    zfya_f = np.random.normal(w_k_zfya * k + np.dot(a1, w_a_zfya), 1)
    zfya_cf = np.random.normal(w_k_zfya * k + np.dot(a1, w_a_zfya), 1)

    return s_f, s_cf, zfya_f, zfya_cf


Race_dict = {"Amerindian": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Asian": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Black": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
             "Hispanic": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
             "Mexican": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
             "Other": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
             "Puertorican": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
             "White": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
             }

Num = 500
p = 9819 / 17432  # men vs women portion.
gender = np.random.binomial(1, p, size=Num)
# Figure 1 Black vs White
for i in range(Num):
    g = np.array([1, 0]) if gender[i] == 1 else np.array([0, 1])
    a1 = np.concatenate((Race_dict["White"], g))
    a2 = np.concatenate((Race_dict["Black"], g))
    x_f, x_cf = fac_and_cf_sample(a1, a2)
