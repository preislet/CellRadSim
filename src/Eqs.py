import numpy as np

def p53_mdm2_system_combined_damage(t, y, p):
    """
    Extended p53-Mdm2 model including both IR (DSB) and UV (SSB) induced damage.
    State variables:
     y[0] = p53_tot
     y[1] = p53U
     y[2] = p53UU
     y[3] = Mdm2_cyt
     y[4] = Mdm2P_cyt
     y[5] = Mdm2_nuc
     y[6] = DNAdam_DSB (from IR)
     y[7] = DNAdam_UV (from UV)
    """
    p53_tot, p53U, p53UU, Mdm2_cyt, Mdm2P_cyt, Mdm2_nuc, DNAdam_DSB, DNAdam_UV = y

    p53_free = max(0, p53_tot - (p53U + p53UU))

    # --- Input Signals ---
    IR_signal, UV_signal = 0.0, 0.0
    if p['IR_start'] < t < p['IR_end']:IR_signal = p['ampl_IR'] 
    if p['UV_start'] < t < p['UV_end']:UV_signal = p['ampl_UV']

    # --- kd2 for Mdm2_nuc degradation (influenced by both DSB and UV damage) ---
    # Effect from DSB damage
    kd2_dsb_effect = (DNAdam_DSB / (p['Jdam_DSB'] + DNAdam_DSB + 1e-9)) * p['kd2__DSB']
    # Effect from UV damage
    kd2_uv_effect = (DNAdam_UV / (p['Jdam_UV'] + DNAdam_UV + 1e-9)) * p['kd2__UV']
    
    kd2 = p['kd2_'] + kd2_dsb_effect + kd2_uv_effect

    # --- Derivatives ---
    dp53_tot_dt = p['ks53'] - p['kd53_'] * p53_tot - p['kd53'] * p53UU

    dp53U_dt = p['kf'] * Mdm2_nuc * p53_free + p['kr'] * p53UU - p53U * (p['kr'] + p['kf'] * Mdm2_nuc) - p['kd53_'] * p53U

    dp53UU_dt = p['kf'] * Mdm2_nuc * p53U - p53UU * (p['kr'] + p['kd53_'] + p['kd53'])

    dMdm2_nuc_dt = p['Vratio'] * (p['ki'] * Mdm2P_cyt - p['ko'] * Mdm2_nuc) - kd2 * Mdm2_nuc
    hill_denominator = (p['Js']**p['m'] + p53_tot**p['m'] + 1e-9)
    mdm2_prod_p53_dependent = (p['ks2'] * (p53_tot**p['m'])) / hill_denominator
    
    phos_denominator = (p['J'] + p53_tot + 1e-9)
    mdm2_phos_term = (p['kph'] / phos_denominator) * Mdm2_cyt
    
    dMdm2_cyt_dt = p['ks2_'] + mdm2_prod_p53_dependent - p['kd2_'] * Mdm2_cyt + p['kdeph'] * Mdm2P_cyt - mdm2_phos_term

    dMdm2P_cyt_dt = mdm2_phos_term - p['kdeph'] * Mdm2P_cyt - p['ki'] * Mdm2P_cyt + p['ko'] * Mdm2_nuc - p['kd2_'] * Mdm2P_cyt

    # --- DNA Damage Dynamics ---
    # DSB Damage (from IR)
    repair_dsb_saturation = (p['JDNA_DSB'] + DNAdam_DSB + 1e-9)
    dna_repair_dsb_term = p['kdDNA_DSB'] * p53_tot * (DNAdam_DSB / repair_dsb_saturation)
    dDNAdam_DSB_dt = p['kDNA_DSB'] * IR_signal - dna_repair_dsb_term
    
    # UV Damage (SSB)
    repair_uv_saturation = (p['JDNA_UV'] + DNAdam_UV + 1e-9)
    dna_repair_uv_term = p['kdDNA_UV'] * p53_tot * (DNAdam_UV / repair_uv_saturation)
    dDNAdam_UV_dt = p['kDNA_UV'] * UV_signal - dna_repair_uv_term
    
    return [
        dp53_tot_dt, dp53U_dt, dp53UU_dt,
        dMdm2_cyt_dt, dMdm2P_cyt_dt, dMdm2_nuc_dt,
        dDNAdam_DSB_dt, dDNAdam_UV_dt
    ]

def p53_full_system(t, y, p):
    """
    Extended p53 model with ATM, ATR, p21, Wip1, and a generic Pro-Apoptotic Factor.
    y = [p53_tot, p53U, p53UU, Mdm2_cyt, Mdm2P_cyt, Mdm2_nuc, DNAdam_DSB, DNAdam_UV, ATM_active, ATR_active, p21_protein, Wip1_protein, ApoptFactor]
    """
    (p53_tot, p53U, p53UU, Mdm2_cyt, Mdm2P_cyt, Mdm2_nuc,DNAdam_DSB, DNAdam_UV, ATM_active, ATR_active, p21_protein, Wip1_protein, ApoptFactor) = y # Added ApoptFactor

    p53_free = max(0, p53_tot - (p53U + p53UU))

    # --- Input Signals ---
    IR_signal = 0.0
    if p.get('IR_start', -1) < t < p.get('IR_end', -1): IR_signal = p.get('ampl_IR', 0.0)
    UV_signal = 0.0
    if p.get('UV_start', -1) < t < p.get('UV_end', -1): UV_signal = p.get('ampl_UV', 0.0)

    # --- ATM & ATR Activation/Inactivation ---
    atm_activation_term = p['k_act_atm_dsb'] * DNAdam_DSB * (p['ATM_total'] - ATM_active)
    atm_inactivation_term = p['k_inact_atm_wip1'] * Wip1_protein * ATM_active
    dATM_active_dt = atm_activation_term - atm_inactivation_term

    atr_activation_term = p['k_act_atr_uv'] * DNAdam_UV * (p['ATR_total'] - ATR_active)
    atr_inactivation_term = p['k_inact_atr'] * ATR_active
    dATR_active_dt = atr_activation_term - atr_inactivation_term

    # --- kd2 for Mdm2_nuc degradation ---
    kd2_atm_effect = (ATM_active / (p['J_atm_kd2'] + ATM_active + 1e-9)) * p['kd2__ATM_max']
    kd2_atr_effect = (ATR_active / (p['J_atr_kd2'] + ATR_active + 1e-9)) * p['kd2__ATR_max']
    kd2 = p['kd2_'] + kd2_atm_effect + kd2_atr_effect

    # --- Effective kf ---
    kf_effective = p['kf']

    # --- Core p53-Mdm2 Loop Derivatives ---
    dp53_tot_dt = p['ks53'] - p['kd53_'] * p53_tot - p['kd53'] * p53UU
    dp53U_dt = kf_effective * Mdm2_nuc * p53_free + p['kr'] * p53UU - p53U * (p['kr'] + kf_effective * Mdm2_nuc) - p['kd53_'] * p53U
    dp53UU_dt = kf_effective * Mdm2_nuc * p53U - p53UU * (p['kr'] + p['kd53_'] + p['kd53'])

    hill_denom_mdm2 = (p['Js']**p['m'] + p53_tot**p['m'] + 1e-9)
    mdm2_prod_p53_dep = (p['ks2'] * (p53_tot**p['m'])) / hill_denom_mdm2
    phos_denom_mdm2 = (p['J'] + p53_tot + 1e-9)

    dMdm2_nuc_dt = p['Vratio'] * (p['ki'] * Mdm2P_cyt - p['ko'] * Mdm2_nuc) - kd2 * Mdm2_nuc
    mdm2_phos_term = (p['kph'] / phos_denom_mdm2) * Mdm2_cyt
    dMdm2_cyt_dt = p['ks2_'] + mdm2_prod_p53_dep - p['kd2_'] * Mdm2_cyt + p['kdeph'] * Mdm2P_cyt - mdm2_phos_term
    dMdm2P_cyt_dt = mdm2_phos_term - p['kdeph'] * Mdm2P_cyt - p['ki'] * Mdm2P_cyt + p['ko'] * Mdm2_nuc - p['kd2_'] * Mdm2P_cyt


    # --- DNA Damage Dynamics ---
    repair_dsb_sat = (p['JDNA_DSB'] + DNAdam_DSB + 1e-9)
    dna_repair_dsb = p['kdDNA_DSB'] * p53_tot * (DNAdam_DSB / repair_dsb_sat)
    dDNAdam_DSB_dt = p['kDNA_DSB'] * IR_signal - dna_repair_dsb
    
    repair_uv_sat = (p['JDNA_UV'] + DNAdam_UV + 1e-9)
    dna_repair_uv = p['kdDNA_UV'] * p53_tot * (DNAdam_UV / repair_uv_sat)
    dDNAdam_UV_dt = p['kDNA_UV'] * UV_signal - dna_repair_uv

    # --- p21 Dynamics ---
    p21_synthesis = (p['ks_p21'] * (p53_tot**p['h_p21'])) / (p['J_p21']**p['h_p21'] + p53_tot**p['h_p21'] + 1e-9)
    dp21_protein_dt = p21_synthesis - p['kd_p21'] * p21_protein

    # --- Wip1 Dynamics ---
    wip1_synthesis = (p['ks_wip1'] * (p53_tot**p['h_wip1'])) / (p['J_wip1']**p['h_wip1'] + p53_tot**p['h_wip1'] + 1e-9)
    dWip1_protein_dt = wip1_synthesis - p['kd_wip1'] * Wip1_protein

    # --- Apoptotic Factor Dynamics (NEW) ---

    p53_driven_apopt_synthesis = (p['ks_apopt'] * (p53_tot**p['h_apopt'])) / (p['J_apopt']**p['h_apopt'] + p53_tot**p['h_apopt'] + 1e-9)
    atm_driven_apopt_synthesis = (p['ks_apopt_atm'] * ATM_active) / (p['J_apopt_atm'] + ATM_active + 1e-9) # Michaelis-Menten for kinase effect
    atr_driven_apopt_synthesis = (p['ks_apopt_atr'] * ATR_active) / (p['J_apopt_atr'] + ATR_active + 1e-9) # Michaelis-Menten for kinase effect

    total_apopt_factor_synthesis = p53_driven_apopt_synthesis + atm_driven_apopt_synthesis + atr_driven_apopt_synthesis 
    dApoptFactor_dt = total_apopt_factor_synthesis - p['kd_apopt'] * ApoptFactor
    return np.array([  
        dp53_tot_dt, dp53U_dt, dp53UU_dt, dMdm2_cyt_dt, dMdm2P_cyt_dt, dMdm2_nuc_dt,
        dDNAdam_DSB_dt, dDNAdam_UV_dt, dATM_active_dt, dATR_active_dt,
        dp21_protein_dt, dWip1_protein_dt, dApoptFactor_dt
    ])