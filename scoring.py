import math
import numpy as np




def calculate_framingham_risk(age, sex, bmi, sbp, treated_bp, smoker, diabetic):
  
    if sex == 'F':
        coeffs = {
            'ln_age': 2.32888, 'ln_bmi': 0.20081,
            'ln_sbp_untreated': 2.76157, 'ln_sbp_treated': 2.82263,
            'smoker': 0.52873, 'diabetes': 0.69154
        }
        S0 = 0.95012
        mean_sum = 26.1931
    else:
        coeffs = {
            'ln_age': 3.11296, 'ln_bmi': 0.79277,
            'ln_sbp_untreated': 1.85508, 'ln_sbp_treated': 1.92672,
            'smoker': 0.70953, 'diabetes': 0.53160
        }
        S0 = 0.88936
        mean_sum = 23.9802


    ln_age = np.log(age)
    ln_bmi = np.log(bmi)
    ln_sbp = np.log(sbp)


    score = (
        coeffs['ln_age'] * ln_age +
        coeffs['ln_bmi'] * ln_bmi +
        (coeffs['ln_sbp_treated'] if treated_bp else coeffs['ln_sbp_untreated']) * ln_sbp +
        coeffs['smoker'] * int(smoker) +
        coeffs['diabetes'] * int(diabetic)
    )
    risk = 1 - S0 ** math.exp(score - mean_sum)
    return round(risk * 100, 2)




def calculate_ascvd_risk(age, sex, race, sbp, treated_bp, total_chol, hdl_chol, smoker, diabetic):
   
    age = np.clip(age, 40, 79)
    sex = sex.upper()


    if sex == 'F' and race.lower() == 'aa':
        b = {
            'ln_age': 17.1141, 'ln_total': 0.9396, 'ln_hdl': -18.9196,
            'ln_age_total': 4.4748, 'ln_treated_sbp': 29.2907,
            'ln_age_treated': -6.4321, 'ln_untreated_sbp': 27.8197,
            'ln_age_untreated': -6.0873, 'smoker': 0.6908, 'diabetes': 0.8738
        }
        baseline = -86.6081
        S10 = 0.9533
    elif sex == 'F':
        b = {
            'ln_age': -29.799, 'ln_age_squared': 4.884, 'ln_total': 13.540,
            'ln_age_total': -3.114, 'ln_hdl': -13.578, 'ln_age_hdl': 3.149,
            'ln_treated_sbp': 2.019, 'ln_untreated_sbp': 1.957,
            'smoker': 7.574, 'ln_age_smoker': -1.665, 'diabetes': 0.661
        }
        baseline = -29.1817
        S10 = 0.9665
    elif race.lower() == 'aa':
        b = {
            'ln_age': 2.469, 'ln_total': 0.302, 'ln_hdl': -0.307,
            'ln_treated_sbp': 1.916, 'ln_untreated_sbp': 1.809,
            'smoker': 0.549, 'diabetes': 0.645
        }
        baseline = 19.6181
        S10 = 0.8954
    else:
        b = {
            'ln_age': 12.344, 'ln_total': 11.853, 'ln_age_total': -2.664,
            'ln_hdl': -7.990, 'ln_age_hdl': 1.769, 'ln_treated_sbp': 1.797,
            'ln_untreated_sbp': 1.764, 'smoker': 7.837,
            'ln_age_smoker': -1.795, 'diabetes': 0.658
        }
        baseline = 61.18
        S10 = 0.9144


    ln_age = math.log(age)
    ln_tot = math.log(total_chol)
    ln_hdl = math.log(hdl_chol)
    ln_sbp = math.log(sbp)


    Xbeta = (
        b['ln_age'] * ln_age +
        b.get('ln_age_squared', 0) * ln_age**2 +
        b['ln_total'] * ln_tot +
        b.get('ln_age_total', 0) * ln_age * ln_tot +
        b['ln_hdl'] * ln_hdl +
        b.get('ln_age_hdl', 0) * ln_age * ln_hdl +
        (b['ln_treated_sbp'] if treated_bp else b['ln_untreated_sbp']) * ln_sbp +
        b.get('ln_age_treated' if treated_bp else 'ln_age_untreated', 0) * ln_age * ln_sbp +
        b['smoker'] * int(smoker) +
        b.get('ln_age_smoker', 0) * ln_age * int(smoker) +
        b['diabetes'] * int(diabetic)
    )
    risk = 1 - S10 ** math.exp(Xbeta - baseline)
    return round(risk * 100, 2)




def calculate_qrisk3_placeholder(age, sex, sbp, smoker, diabetic):
    base = 5.0  # assume a flat 5% as placeholder
    modifier = (0.2 * (age - 40)) + (2 if smoker else 0) + (3 if diabetic else 0)
    risk = base + modifier
    return round(min(max(risk, 0), 100), 2)
