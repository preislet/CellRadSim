import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from Eqs import p53_mdm2_system_combined_damage

CELL_CYCLE_STATE_MAP = {
    "Proliferating": 0,
    "G1 Arrest": 1,
    "Stress Arrest (G2/M?)": 2,
    "Apoptosis": 3,
    "Necrosis": 4  # New state
}
CELL_CYCLE_STATE_COLORS = {
    "Proliferating": 'lightgreen',
    "G1 Arrest": 'gold',
    "Stress Arrest (G2/M?)": 'lightcoral',
    "Apoptosis": 'darkred',
    "Necrosis": 'black'  # New color
}

def determine_cell_cycle_state(y_timepoint_all_species, p_params, time_point, apoptosis_info, necrosis_info=None):
    """
    Determines the cell cycle state based on molecular levels.
    Args:
        y_timepoint_all_species: Array of all species values at a single time point.
                                 Order: [p53_tot, ..., ATM_active, ATR_active, p21, Wip1, ApoptFactor]
        p_params: Dictionary of parameters containing thresholds.
        time_point: The current time (used for checking against apoptosis time).
        apoptosis_info: A tuple (is_apoptotic_overall, time_of_death_overall, reason)
                        from a run-wide apoptosis check (e.g., using ApoptFactor).
    Returns:
        A string representing the cell state.
    """
    p53_tot      = y_timepoint_all_species[0]
    ATM_active   = y_timepoint_all_species[8]
    ATR_active   = y_timepoint_all_species[9]
    p21_protein  = y_timepoint_all_species[10]

    if necrosis_info is not None:
        is_necrotic_overall, time_of_necrosis_overall, _ = necrosis_info
        if is_necrotic_overall and time_point >= time_of_necrosis_overall:
            return "Necrosis"
    else:
        is_necrotic_overall = False
    # Use the provided apoptosis_info tuple


    is_apoptotic_overall, time_of_death_overall, _ = apoptosis_info
    

    # 1. Check for Apoptosis (based on overall simulation check)
    if is_apoptotic_overall and time_point >= time_of_death_overall:
        return "Apoptosis"

    # 2. Check for G1 Arrest (p21-driven)
    if p21_protein >= p_params.get('p21_G1_arrest_threshold', 0.4):
        return "G1 Arrest"

    # 3. Check for Stress-Induced Arrest (Potential G2/M or severe S-phase delay)
    #    This is more speculative and depends on combined high p53 and active kinases.
    #    This condition should be met if not already in p21-mediated G1 arrest.
    is_stressed_by_kinases = (ATM_active >= p_params.get('ATM_active_stress_threshold', 0.3) or
                              ATR_active >= p_params.get('ATR_active_stress_threshold', 0.3))
    if is_stressed_by_kinases and p53_tot >= p_params.get('p53_stress_arrest_threshold', 0.8):
        return "Stress Arrest (G2/M?)"

    # 4. Default: Proliferating / Cycling
    return "Proliferating"

def check_for_necrosis(time_array, DNAdam_DSB_array, DNAdam_UV_array, p_params):
    """
    Checks for necrosis based on sustained extreme DNA damage.
    Returns: (is_necrotic, time_of_necrosis_commitment, reason)
    """
    if not len(time_array) or (not len(DNAdam_DSB_array) and not len(DNAdam_UV_array)):
        return False, None, "No data for necrosis check"

    dt_approx = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    duration_necro_thresh_min = p_params.get('DNAdam_necro_duration_min', 60)

    # Necrosis due to extreme DSB damage
    dsb_necro_thresh = p_params.get('DNAdam_DSB_necro_thresh', float('inf'))
    if dsb_necro_thresh != float('inf') and len(DNAdam_DSB_array) > 0:
        start_time_dsb_necrotic = None
        for i in range(len(DNAdam_DSB_array)):
            if DNAdam_DSB_array[i] >= dsb_necro_thresh:
                if start_time_dsb_necrotic is None:
                    start_time_dsb_necrotic = time_array[i]
                current_duration = time_array[i] - start_time_dsb_necrotic + dt_approx # commitment at end of duration
                if current_duration >= duration_necro_thresh_min:
                    return True, time_array[i], "Necrosis_Extreme_DSB"
            else:
                start_time_dsb_necrotic = None

    # Necrosis due to extreme UV damage
    uv_necro_thresh = p_params.get('DNAdam_UV_necro_thresh', float('inf'))
    if uv_necro_thresh != float('inf') and len(DNAdam_UV_array) > 0:
        start_time_uv_necrotic = None
        for i in range(len(DNAdam_UV_array)):
            if DNAdam_UV_array[i] >= uv_necro_thresh:
                if start_time_uv_necrotic is None:
                    start_time_uv_necrotic = time_array[i]
                current_duration = time_array[i] - start_time_uv_necrotic + dt_approx
                if current_duration >= duration_necro_thresh_min:
                    return True, time_array[i], "Necrosis_Extreme_UV"
            else:
                start_time_uv_necrotic = None
    
    return False, None, "Survived_Necrosis_Check"

def check_apoptosis_by_factor(time_array, factor_array, factor_threshold, duration_threshold_min):
    if not len(time_array) or not len(factor_array): return False, None, "No data"
    dt_approx = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    if duration_threshold_min <= 0: # Instantaneous
        above_thresh_idx = np.where(factor_array >= factor_threshold)[0]
        return (True, time_array[above_thresh_idx[0]], "Factor_threshold_met") if len(above_thresh_idx) > 0 else (False, None, "Survived")

    start_time_sustained_high = None
    for i in range(len(factor_array)):
        if factor_array[i] >= factor_threshold:
            if start_time_sustained_high is None: start_time_sustained_high = time_array[i]
            current_duration = time_array[i] - start_time_sustained_high + dt_approx
            if current_duration >= duration_threshold_min:
                return True, time_array[i], "Factor_duration_met"
        else:
            start_time_sustained_high = None
    return False, None, "Survived"



def check_for_apoptosis_comprehensive(
    time_array, p53_total_array,
    DNAdam_DSB_array, DNAdam_UV_array,
    p_params 
    ):
    # ... (p53 threshold logic as above) ...
    # Call the previous function or integrate its logic:
    p53_apoptosis_triggered, p53_death_time, p53_reason = check_for_apoptosis_damage_specific(
        time_array, p53_total_array, DNAdam_DSB_array, DNAdam_UV_array, p_params
    )
    if p53_apoptosis_triggered:
        return True, p53_death_time, p53_reason

    # --- Now check for direct DNA damage thresholds ---
    dt_approx = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    duration_dna_thresh_min = p_params.get('DNAdam_apopt_duration_min', 30)
    
    # Check DSB damage
    dsb_thresh = p_params.get('DNAdam_DSB_apopt_thresh', float('inf')) # Default to infinity if not set
    start_time_dsb_critical = None
    for i in range(len(DNAdam_DSB_array)):
        if DNAdam_DSB_array[i] >= dsb_thresh:
            if start_time_dsb_critical is None:
                start_time_dsb_critical = time_array[i]
            current_duration = time_array[i] - start_time_dsb_critical + dt_approx
            if current_duration >= duration_dna_thresh_min:
                return True, start_time_dsb_critical, "High_DSB_Damage_Duration"
        else:
            start_time_dsb_critical = None

    # Check UV damage
    uv_thresh = p_params.get('DNAdam_UV_apopt_thresh', float('inf')) # Default to infinity if not set
    start_time_uv_critical = None
    for i in range(len(DNAdam_UV_array)):
        if DNAdam_UV_array[i] >= uv_thresh:
            if start_time_uv_critical is None:
                start_time_uv_critical = time_array[i]
            current_duration = time_array[i] - start_time_uv_critical + dt_approx
            if current_duration >= duration_dna_thresh_min:
                return True, start_time_uv_critical, "High_UV_Damage_Duration"
        else:
            start_time_uv_critical = None
            
    return False, None, "Survived" # If p53 check also returned False



def check_for_apoptosis_damage_specific(
    time_array, p53_total_array,
    DNAdam_DSB_array, DNAdam_UV_array,
    p_params # Pass the whole params dictionary
    ):
    """
    Checks for apoptosis based on p53 levels, where the p53 threshold
    can depend on the dominant type of DNA damage.
    """
    if not len(time_array) or not len(p53_total_array):
        return False, None, "No data"

    dt_approx = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
    duration_threshold_min = p_params['p53_apoptosis_duration_min']
    
    current_p53_threshold = p_params.get('p53_apoptosis_threshold', 1.1) # Fallback

    start_time_of_sustained_high_p53 = None
    apoptosis_reason = "Survived"

    for i in range(len(p53_total_array)):
        # Determine dominant damage at this time point to select p53 threshold
        # This logic can be sophisticated. Simple example:
        is_dsb_high = DNAdam_DSB_array[i] > p_params.get('Jdam_DSB', 0.2) * 0.5 # e.g. if DSB damage is above half its saturation for kd2 effect
        is_uv_high = DNAdam_UV_array[i] > p_params.get('Jdam_UV', 0.1) * 0.5   # e.g. if UV damage is above half its saturation for kd2 effect

        if is_dsb_high and is_uv_high:
            current_p53_threshold = p_params.get('p53_thresh_apopt_BOTH', 1.1)
            current_reason_prefix = "Both_Damage_p53"
        elif is_dsb_high:
            current_p53_threshold = p_params.get('p53_thresh_apopt_DSB', 1.2)
            current_reason_prefix = "DSB_Damage_p53"
        elif is_uv_high:
            current_p53_threshold = p_params.get('p53_thresh_apopt_UV', 1.0)
            current_reason_prefix = "UV_Damage_p53"
        else:
            current_p53_threshold = max(p_params.get('p53_thresh_apopt_DSB', 1.2), 
                                        p_params.get('p53_thresh_apopt_UV', 1.0),
                                        p_params.get('p53_thresh_apopt_BOTH', 1.1))
            current_reason_prefix = "Low_Damage_p53"


        if p53_total_array[i] >= current_p53_threshold:
            if start_time_of_sustained_high_p53 is None:
                start_time_of_sustained_high_p53 = time_array[i]
            
            current_duration = time_array[i] - start_time_of_sustained_high_p53 + dt_approx
            
            if current_duration >= duration_threshold_min:
                apoptosis_reason = f"{current_reason_prefix}_duration_met"
                return True, start_time_of_sustained_high_p53, apoptosis_reason
        else:
            start_time_of_sustained_high_p53 = None
            
    return False, None, apoptosis_reason


def count_p53_pulses_and_check_apoptosis(stimulus_duration, stimulus_type, base_params_dict, y0_val, t_span_val, t_eval_val, ode_func, factor=False, apopt_factor_trace=None):
    sim_params_loop = base_params_dict.copy()
    sim_params_loop['IR_start'], sim_params_loop['IR_end'], sim_params_loop['ampl_IR'] = -1, -1, 0.0
    sim_params_loop['UV_start'], sim_params_loop['UV_end'], sim_params_loop['ampl_UV'] = -1, -1, 0.0
    stim_start_time = 10

    if stimulus_type == 'IR' or stimulus_type == 'BOTH':
        sim_params_loop['IR_start'] = stim_start_time
        if stimulus_duration <= 0: sim_params_loop['IR_end'] = stim_start_time; sim_params_loop['ampl_IR'] = 0.0
        else: sim_params_loop['IR_end'] = stim_start_time + stimulus_duration; sim_params_loop['ampl_IR'] = base_params_dict.get('ampl_IR', 1.0)
    if stimulus_type == 'UV' or stimulus_type == 'BOTH':
        sim_params_loop['UV_start'] = stim_start_time
        if stimulus_duration <= 0: sim_params_loop['UV_end'] = stim_start_time; sim_params_loop['ampl_UV'] = 0.0
        else: sim_params_loop['UV_end'] = stim_start_time + stimulus_duration; sim_params_loop['ampl_UV'] = base_params_dict.get('ampl_UV', 1.0)
    if stimulus_type not in ['IR', 'UV', 'BOTH']: raise ValueError("stimulus_type must be 'IR', 'UV', or 'BOTH'")

    sol_loop_run = solve_ivp(ode_func, t_span_val, y0_val, t_eval=t_eval_val, args=(sim_params_loop,), method='LSODA')
    
    num_pulses = 0
    apoptosis_triggered_loop = False
    time_of_death_loop = None

    if sol_loop_run.success:
        p53_tot_trace = sol_loop_run.y[0]
        
        # Apoptosis Check
        if not factor:
            apoptosis_triggered_loop, time_of_death_loop, reason = check_for_apoptosis_comprehensive(
                                            sol_loop_run.t, p53_tot_trace,
                                            sol_loop_run.y[6], # DNAdam_DSB_trace
                                            sol_loop_run.y[7], # DNAdam_UV_trace
                                            sim_params_loop
                                            )
            
        else:
            apopt_factor_trace = sol_loop_run.y[12]
            apoptosis_triggered_loop, time_of_death_loop, reason = check_apoptosis_by_factor(
                sol_loop_run.t, apopt_factor_trace,
                sim_params_loop['ApoptFactor_threshold'],
                sim_params_loop['ApoptFactor_duration_min'])

        # Pulse Counting (only count pulses before apoptosis if it occurs)
        peak_height_threshold = 0.4; peak_prominence_threshold = 0.3
        points_per_min = 1.0 / (t_eval_val[1] - t_eval_val[0]) if len(t_eval_val) > 1 else 1.0
        peak_distance_samples = int(300 * points_per_min)
        
        time_mask_for_peaks = sol_loop_run.t >= stim_start_time
        if apoptosis_triggered_loop and time_of_death_loop is not None:
            # Only consider time points before or at the point apoptosis is committed
            time_mask_for_peaks &= (sol_loop_run.t <= time_of_death_loop)

        p53_for_counting = p53_tot_trace[time_mask_for_peaks]
        
        if len(p53_for_counting) >= peak_distance_samples : # Check length of the potentially truncated array
            peaks, _ = find_peaks(p53_for_counting, height=peak_height_threshold, prominence=peak_prominence_threshold, distance=peak_distance_samples)
            num_pulses = len(peaks)
        else:
            num_pulses = 0 # Not enough data to find peaks (e.g., apoptosis too early)
    else:
        print(f"Solver failed for type {stimulus_type}, duration {stimulus_duration}")

    return num_pulses, apoptosis_triggered_loop, time_of_death_loop, reason


def plot_p53_mdm2_simulation(sol_main,params_combined, y0_combined, t_span_main_sim, t_eval_main_sim):

    t_values_main = sol_main.t
    p53_tot_main = sol_main.y[0]
    p53U_main = sol_main.y[1]
    p53UU_main = sol_main.y[2]
    Mdm2_cyt_main = sol_main.y[3]
    Mdm2P_cyt_main = sol_main.y[4]
    Mdm2_nuc_main = sol_main.y[5]
    DNAdam_DSB_main = sol_main.y[6]
    DNAdam_UV_main = sol_main.y[7]
    p53_free_main = np.maximum(0, p53_tot_main - (p53U_main + p53UU_main))

    kd2_dsb_effect_main = (DNAdam_DSB_main / (params_combined['Jdam_DSB'] + DNAdam_DSB_main + 1e-9)) * params_combined['kd2__DSB']
    kd2_uv_effect_main = (DNAdam_UV_main / (params_combined['Jdam_UV'] + DNAdam_UV_main + 1e-9)) * params_combined['kd2__UV']
    kd2_for_plot_c = params_combined['kd2_'] + kd2_dsb_effect_main + kd2_uv_effect_main
    mdm2_synthesis_p53_dependent_main = (params_combined['ks2'] * (p53_tot_main**params_combined['m'])) / (params_combined['Js']**params_combined['m'] + p53_tot_main**params_combined['m'] + 1e-9)


    # --- B) Simulations for Number of Pulses vs. Stimulus Duration (Plot D & G & New "Both") ---
    t_simulation_end_pulse_scan = 1500
    t_eval_pulse_scan = np.linspace(0, t_simulation_end_pulse_scan, t_simulation_end_pulse_scan * 2 + 1)

    # --- Run simulations for Plot D (IR duration scan) ---
    stimulus_durations = np.concatenate(([0], np.arange(2, 32, 2)))
    results_ir_scan = [count_p53_pulses_and_check_apoptosis(dur, 'IR', params_combined, y0_combined, (0, t_simulation_end_pulse_scan), t_eval_pulse_scan, p53_mdm2_system_combined_damage)
                    for dur in stimulus_durations]
    pulse_counts_ir = [res[0] for res in results_ir_scan]
    apoptosis_flags_ir = [res[1] for res in results_ir_scan]
    # apoptosis_times_ir = [res[2] for res in results_ir_scan] # Can be used if needed

    # --- Run simulations for Plot G (UV duration scan) ---
    results_uv_scan = [count_p53_pulses_and_check_apoptosis(dur, 'UV', params_combined, y0_combined, (0, t_simulation_end_pulse_scan), t_eval_pulse_scan, p53_mdm2_system_combined_damage)
                    for dur in stimulus_durations]
    pulse_counts_uv = [res[0] for res in results_uv_scan]
    apoptosis_flags_uv = [res[1] for res in results_uv_scan]

    # --- Run simulations for "BOTH" duration scan (for the combined pulse plot) ---
    results_both_scan = [count_p53_pulses_and_check_apoptosis(dur, 'BOTH', params_combined, y0_combined, (0, t_simulation_end_pulse_scan), t_eval_pulse_scan, p53_mdm2_system_combined_damage)
                        for dur in stimulus_durations]
    pulse_counts_both = [res[0] for res in results_both_scan]
    apoptosis_flags_both = [res[1] for res in results_both_scan]


    # ---------------------------------------------------------------------------
    # 4. CREATE SUBPLOTS (3x2 grid)
    # ---------------------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(25, 21))
    color_p53 = 'dodgerblue'; color_mdm2nuc = 'red'; color_mdm2cyt = 'forestgreen'; color_mdm2pcyt = 'darkorange'
    color_dnadam_dsb_plot = 'purple'; color_dnadam_uv_plot = 'magenta'; color_kd2_plot = 'sienna'
    color_p53_free = 'royalblue'; color_p53u = 'deepskyblue'; color_p53uu = 'lightskyblue'
    color_mdm2_synth_rate = 'gold'; color_plot_d_ir_line = 'teal'; color_plot_g_uv_line = 'darkcyan'
    color_phase_ir = 'grey'; color_phase_uv = 'darkgreen'; color_phase_both = 'indigo'

    # --- Row 0: Time series from sol_main ---
    axs[0, 0].plot(t_values_main, p53_tot_main, label='p53_total', color=color_p53, lw=2)
    axs[0, 0].plot(t_values_main, Mdm2_nuc_main, label='Mdm2_nuc', color=color_mdm2nuc, lw=2)
    axs[0, 0].set_title('A: Total p53 & Nuc Mdm2 (Main Sim)'); axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.5); axs[0, 0].set_ylim(0, 1.6); axs[0, 0].set_xlabel('Time (min)')
    axs[0, 0].set_ylabel('Concentration')


    axs[0, 1].plot(t_values_main, Mdm2_cyt_main, label='Mdm2_cyt', color=color_mdm2cyt, lw=2)
    axs[0, 1].plot(t_values_main, Mdm2P_cyt_main, label='Mdm2P_cyt', color=color_mdm2pcyt, lw=2)
    axs[0, 1].plot(t_values_main, Mdm2_nuc_main, label="Mdm2_nuc", color='darkviolet', lw=1.5, linestyle='--')
    axs[0, 1].set_title('B: Cytoplasmic Mdm2 (Main Sim)'); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.5)
    axs[0, 1].set_ylim(0, 0.35)
    axs[0, 1].set_xlabel('Time (min)')
    axs[0, 1].set_ylabel('Concentration')

    # --- Row 1: Time series from sol_main ---
    ax_c1 = axs[1, 0]; ax_c1.plot(t_values_main, DNAdam_DSB_main, color=color_dnadam_dsb_plot, lw=2.5, label='DNAdam_DSB (IR)')
    if params_combined.get('ampl_UV',0) > 0 and params_combined.get('UV_end',-1) > params_combined.get('UV_start',-1):
        ax_c1.plot(t_values_main, DNAdam_UV_main, color=color_dnadam_uv_plot, linestyle='--', lw=2.0, label='DNAdam_UV')
        ax_c1.plot(t_values_main, DNAdam_DSB_main + DNAdam_UV_main, color='darkviolet', lw=2.5, label='Total DNA Damage (IR+UV)')
    ax_c1.set_ylabel('DNA Damage Levels')
    ax_c1.tick_params(axis='y')
    ax_c1.set_ylim(bottom=0)
    ax_c2 = ax_c1.twinx()
    ax_c2.plot(t_values_main, kd2_for_plot_c, color=color_kd2_plot, lw=2.5, label='Total kd2')
    ax_c2.set_ylabel('Total kd2 Rate', color=color_kd2_plot)
    ax_c2.tick_params(axis='y', labelcolor=color_kd2_plot)
    ax_c2.set_ylim(bottom=0, top=0.025)
    ax_c1.set_title('C: DNA Damage & kd2 (Main Sim)')
    lines_c1, labels_c1 = ax_c1.get_legend_handles_labels()
    lines_c2, labels_c2 = ax_c2.get_legend_handles_labels()
    ax_c1.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc='upper right')
    ax_c1.grid(True, ls=':', alpha=0.3)
    ax_c1.set_xlabel('Time (min)')

    axs[1, 1].plot(t_values_main, p53_free_main, label='p53 (Free)', color=color_p53_free, lw=2); axs[1, 1].plot(t_values_main, p53U_main, label='p53U (Mono-Ub)', color=color_p53u, lw=2); axs[1, 1].plot(t_values_main, p53UU_main, label='p53UU (Poly-Ub)', color=color_p53uu, lw=2)
    axs[1, 1].set_title('D: p53 Ubiquitination (Main Sim)'); axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.5); axs[1, 1].set_xlabel('Time (min)'); axs[1, 1].set_ylabel('Concentration')

    # --- Row 2: Time series from sol_main and Combined Pulse Counts ---
    ax_f1 = axs[2, 0]; ax_f1.plot(t_values_main, p53_tot_main, label='p53_total', color=color_p53, lw=2)
    ax_f1.set_ylabel('p53_total Conc.', color=color_p53)
    ax_f1.tick_params(axis='y', labelcolor=color_p53)
    ax_f2 = ax_f1.twinx()
    ax_f2.plot(t_values_main, mdm2_synthesis_p53_dependent_main, label='Mdm2 Synth. (p53-dep.)', color=color_mdm2_synth_rate, ls='--', lw=2)
    ax_f2.set_ylabel('Mdm2 Synth. Rate', color=color_mdm2_synth_rate)
    ax_f2.tick_params(axis='y', labelcolor=color_mdm2_synth_rate)
    lines_f1, labels_f1 = ax_f1.get_legend_handles_labels()
    lines_f2, labels_f2 = ax_f2.get_legend_handles_labels()
    ax_f1.legend(lines_f1 + lines_f2, labels_f1 + labels_f2, loc='upper right')
    ax_f1.grid(True, alpha=0.5)
    ax_f1.set_title('E: Mdm2 Synthesis (Main Sim)')
    ax_f1.set_xlabel('Time (min)')

    ax_pulse_compare = axs[2, 1]
    ax_pulse_compare.clear() # Clear if re-running cell

    # IR Dose Response
    ax_pulse_compare.plot(stimulus_durations, pulse_counts_ir, marker='o', ls='-', color='teal', lw=1.5, ms=6, label='IR Pulses')
    for i, dur in enumerate(stimulus_durations):
        ax_pulse_compare.text(dur, pulse_counts_ir[i] + 0.25, str(pulse_counts_ir[i]), ha='center', va='bottom', fontsize=8, color='teal')
        if apoptosis_flags_ir[i]:
            ax_pulse_compare.plot(dur, pulse_counts_ir[i], marker='x', markersize=10, color='red', markeredgewidth=2) # Mark apoptosis

    # UV Dose Response
    ax_pulse_compare.plot(stimulus_durations, pulse_counts_uv, marker='s', ls='--', color='darkcyan', lw=1.5, ms=6, label='UV Pulses')
    for i, dur in enumerate(stimulus_durations):
        ax_pulse_compare.text(dur, pulse_counts_uv[i] - 0.35, str(pulse_counts_uv[i]), ha='center', va='top', fontsize=8, color='darkcyan') # Adjust text position
        if apoptosis_flags_uv[i]:
            ax_pulse_compare.plot(dur, pulse_counts_uv[i], marker='x', markersize=10, color='red', markeredgewidth=2)

    # BOTH IR and UV Dose Response
    ax_pulse_compare.plot(stimulus_durations, pulse_counts_both, marker='^', ls=':', color='orangered', lw=1.5, ms=6, label='IR+UV Pulses')
    for i, dur in enumerate(stimulus_durations):
        ax_pulse_compare.text(dur, pulse_counts_both[i] + 0.25, str(pulse_counts_both[i]), ha='center', va='bottom', fontsize=8, color='orangered', alpha=0.7) # Slightly offset
        if apoptosis_flags_both[i]:
            ax_pulse_compare.plot(dur, pulse_counts_both[i], marker='x', markersize=10, color='darkred', markeredgewidth=2) # Different color for "both" apoptosis


    ax_pulse_compare.set_title('F: Pulses vs. Stimulus Duration (X marks apoptosis)')
    ax_pulse_compare.legend(loc='best')
    ax_pulse_compare.grid(True, ls=':', alpha=0.7)
    ax_pulse_compare.set_xticks(np.arange(0, max(stimulus_durations) + 5, 5))
    all_pulse_counts_for_ylim = pulse_counts_ir + pulse_counts_uv + pulse_counts_both
    if all_pulse_counts_for_ylim: max_pulses_overall = max(all_pulse_counts_for_ylim)
    else: max_pulses_overall = 4 # Default
    ax_pulse_compare.set_yticks(np.arange(0, max_pulses_overall + 2, 1))
    ax_pulse_compare.set_ylim(bottom=-0.5)
    ax_pulse_compare.set_xlabel('Stimulus Duration (min)')
    ax_pulse_compare.set_ylabel('Number of p53 Pulses')



    # --- Final Layout ---
    # print environment information
    print("Environment Information:")
    print(f"UV Start: {params_combined['UV_start']}, UV End: {params_combined['UV_end']}, UV Amplitude: {params_combined['ampl_UV']}")
    print(f"IR Start: {params_combined['IR_start']}, IR End: {params_combined['IR_end']}, IR Amplitude: {params_combined['ampl_IR']}")


    # chacked for apoptosis in main simulation
    apoptosis_triggered_main, time_of_death_main, reason = check_for_apoptosis_comprehensive(
        time_array=t_values_main, 
        p53_total_array=p53_tot_main,
        DNAdam_DSB_array=DNAdam_DSB_main,
        DNAdam_UV_array=DNAdam_UV_main,
        p_params=params_combined,

    )

    print("Cell information:")
    if apoptosis_triggered_main:
        print(f"Apoptosis triggered in main simulation at time: {time_of_death_main:.2f} minutes")
        print(f"Reason for apoptosis: {reason}")
    else:
        print("No apoptosis triggered in main simulation.")

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    plt.suptitle("P53-Mdm2 Network Dynamics (Combined IR/UV Damage Model)", fontsize=18, y=1.00)
    plt.show()


def plot_p53_full_simulation(
    sol_main,
    params_used_for_sol_main,
    stimulus_durations_scan,
    pulse_counts_ir_scan,
    apoptosis_flags_ir_scan,
    pulse_counts_uv_scan,
    apoptosis_flags_uv_scan,
    pulse_counts_both_scan,
    apoptosis_flags_both_scan,
    save_fig=False,
    cell_type="",
    ):
    """
    Generates a 5x2 subplot grid for the p53_full_system simulations.
    Plot F (axs[2,0]) is now Cell Cycle State.
    """

    t_values_main = sol_main.t
    p53_tot_main   = sol_main.y[0]; p53U_main      = sol_main.y[1]; p53UU_main     = sol_main.y[2]
    Mdm2_cyt_main  = sol_main.y[3]; Mdm2P_cyt_main = sol_main.y[4]; Mdm2_nuc_main  = sol_main.y[5]
    DNAdam_DSB_main = sol_main.y[6]; DNAdam_UV_main  = sol_main.y[7]
    ATM_active_main = sol_main.y[8]; ATR_active_main = sol_main.y[9]
    p21_main       = sol_main.y[10]; Wip1_main      = sol_main.y[11]
    ApoptFactor_main = sol_main.y[12] if sol_main.y.shape[0] > 12 else np.zeros_like(t_values_main)

    p53_free_main  = np.maximum(0, p53_tot_main - (p53U_main + p53UU_main))

    kd2_atm_effect_main = (ATM_active_main / (params_used_for_sol_main['J_atm_kd2'] + ATM_active_main + 1e-9)) * params_used_for_sol_main['kd2__ATM_max']
    kd2_atr_effect_main = (ATR_active_main / (params_used_for_sol_main['J_atr_kd2'] + ATR_active_main + 1e-9)) * params_used_for_sol_main['kd2__ATR_max']
    kd2_for_plot_c = params_used_for_sol_main['kd2_'] + kd2_atm_effect_main + kd2_atr_effect_main

    apoptosis_triggered_main_sim, time_of_death_main_sim, reason_main_sim = check_apoptosis_by_factor(
        t_values_main, ApoptFactor_main,
        params_used_for_sol_main.get('ApoptFactor_threshold', 1.0),
        params_used_for_sol_main.get('ApoptFactor_duration_min', 300) # Ensure this matches params
    )
    apoptosis_info_main_sim = (apoptosis_triggered_main_sim, time_of_death_main_sim, reason_main_sim)
    necrosis_triggered_main_sim, time_of_necrosis_main_sim, reason_necrosis_main_sim = check_for_necrosis(t_values_main,DNAdam_DSB_main, DNAdam_UV_main,params_used_for_sol_main)
    necrosis_info_main_sim = (necrosis_triggered_main_sim, time_of_necrosis_main_sim, reason_necrosis_main_sim)

    # --- Calculate Cell Cycle States for Plot F ---
    cell_cycle_states_numerical = np.zeros_like(t_values_main, dtype=int)
    cell_cycle_state_labels_for_plot = []



    for i, t_i in enumerate(t_values_main):
        y_timepoint = sol_main.y[:, i]
        current_state_str = determine_cell_cycle_state(y_timepoint, params_used_for_sol_main, t_i, apoptosis_info_main_sim, necrosis_info_main_sim)
        cell_cycle_states_numerical[i] = CELL_CYCLE_STATE_MAP[current_state_str]
        if not cell_cycle_state_labels_for_plot or cell_cycle_state_labels_for_plot[-1][1] != current_state_str:
             cell_cycle_state_labels_for_plot.append((t_i, current_state_str))





    # ---------------------------------------------------------------------------
    # 2. CREATE SUBPLOTS (5x2 grid)
    # ---------------------------------------------------------------------------
    fig, axs = plt.subplots(5, 2, figsize=(20, 30), )
    # ... (your color definitions) ...
    color_p53 = 'dodgerblue'; color_mdm2nuc = 'red'; color_mdm2cyt = 'forestgreen'; color_mdm2pcyt = 'darkorange'
    color_dnadam_dsb_plot = 'purple'; color_dnadam_uv_plot = 'magenta'; color_kd2_plot = 'sienna'
    color_p53_free = 'royalblue'; color_p53u = 'deepskyblue'; color_p53uu = 'lightskyblue'
    # color_mdm2_synth_rate = 'gold'; # No longer used for plot F
    color_plot_ir_line = 'teal'; color_plot_uv_line = 'darkcyan'; color_plot_both_line = 'orangered'
    color_phase_ir = 'grey'; color_phase_uv = 'darkgreen'; color_phase_both = 'indigo'
    color_atm = 'darkred'; color_atr = 'darkslateblue'; color_p21 = 'olive'; color_wip1 = 'chocolate'
    color_apopt_factor = 'firebrick'

    # --- Row 0, Row 1 (Plots A, B, C, E) ---
    # These remain unchanged from your existing plot_p53_full_simulation
    axs[0, 0].plot(t_values_main, p53_tot_main, label='p53_total', color=color_p53, lw=2)
    axs[0, 0].plot(t_values_main, Mdm2_nuc_main, label='Mdm2_nuc', color=color_mdm2nuc, lw=2)
    axs[0, 0].set_title('A: p53_total & Mdm2_nuc'); axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.5); axs[0, 0].set_xlabel('Time (min)'); axs[0, 0].set_ylabel('Concentration')

    axs[0, 1].plot(t_values_main, Mdm2_cyt_main, label='Mdm2_cyt', color=color_mdm2cyt, lw=2)
    axs[0, 1].plot(t_values_main, Mdm2P_cyt_main, label='Mdm2P_cyt', color=color_mdm2pcyt, lw=2)
    axs[0, 1].set_title('B: Cytoplasmic Mdm2'); axs[0, 1].legend(); axs[0, 1].grid(True, alpha=0.5); axs[0, 1].set_xlabel('Time (min)'); axs[0, 1].set_ylabel('Concentration')

    ax_c1 = axs[1, 0]
    ax_c1.plot(t_values_main, DNAdam_DSB_main, color=color_dnadam_dsb_plot, lw=2.5, label='DNAdam_DSB (IR)')
    if params_used_for_sol_main.get('ampl_UV',0) > 0 and params_used_for_sol_main.get('UV_end',-1) > params_used_for_sol_main.get('UV_start',-1):
        ax_c1.plot(t_values_main, DNAdam_UV_main, color=color_dnadam_uv_plot, linestyle='--', lw=2.0, label='DNAdam_UV')
    ax_c1.set_ylabel('DNA Damage Levels'); ax_c1.tick_params(axis='y'); ax_c1.set_ylim(bottom=0)
    ax_c2 = ax_c1.twinx()
    ax_c2.plot(t_values_main, kd2_for_plot_c, color=color_kd2_plot, lw=2.5, label='Total kd2 (from ATM/ATR)')
    ax_c2.set_ylabel('Total kd2 Rate', color=color_kd2_plot); ax_c2.tick_params(axis='y', labelcolor=color_kd2_plot); ax_c2.set_ylim(bottom=params_used_for_sol_main['kd2_'] * 0.9)
    ax_c1.set_title('C: DNA Damage & kd2'); lines_c1, labels_c1 = ax_c1.get_legend_handles_labels(); lines_c2, labels_c2 = ax_c2.get_legend_handles_labels(); ax_c1.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc='upper right'); ax_c1.grid(True, ls=':', alpha=0.3); ax_c1.set_xlabel('Time (min)')

    axs[1, 1].plot(t_values_main, p53_free_main, label='p53 (Free)', color=color_p53_free, lw=2)
    axs[1, 1].plot(t_values_main, p53U_main, label='p53U (Mono-Ub)', color=color_p53u, lw=2)
    axs[1, 1].plot(t_values_main, p53UU_main, label='p53UU (Poly-Ub)', color=color_p53uu, lw=2)
    axs[1, 1].set_title('D: p53 Ubiquitination'); axs[1, 1].legend(); axs[1, 1].grid(True, alpha=0.5); axs[1, 1].set_xlabel('Time (min)'); axs[1, 1].set_ylabel('Concentration')


    # --- Row 2: NEW Plot F (Cell Cycle State) and Plot G (Active Kinases) ---
    ax_f_new = axs[2, 0]
    unique_states = sorted(CELL_CYCLE_STATE_MAP.keys(), key=lambda k: CELL_CYCLE_STATE_MAP[k])
    for state_str in unique_states:
        state_val = CELL_CYCLE_STATE_MAP[state_str]
        ax_f_new.fill_between(t_values_main, state_val - 0.45, state_val + 0.45,
                              where=(cell_cycle_states_numerical == state_val),
                              color=CELL_CYCLE_STATE_COLORS[state_str],
                              label=state_str, step='post', alpha=0.7)

    ax_f_new.set_yticks(list(CELL_CYCLE_STATE_MAP.values()))
    ax_f_new.set_yticklabels(list(CELL_CYCLE_STATE_MAP.keys()))
    ax_f_new.set_ylim(-0.5, len(CELL_CYCLE_STATE_MAP) - 0.5)
    ax_f_new.set_title('E: Inferred Cell Cycle State')
    ax_f_new.legend(loc='upper right', fontsize='small')
    ax_f_new.grid(True, alpha=0.3, axis='x') 
    ax_f_new.set_xlabel('Time (min)')
    ax_f_new.set_ylabel('State')

    # Plot G (Active Kinases) remains the same
    axs[2, 1].plot(t_values_main, ATM_active_main, label='ATM_active', color=color_atm, lw=2)
    axs[2, 1].plot(t_values_main, ATR_active_main, label='ATR_active', color=color_atr, linestyle='--', lw=2)
    axs[2, 1].set_title('F: Active Kinases'); axs[2, 1].legend(); axs[2, 1].grid(True, alpha=0.5); axs[2, 1].set_ylim(bottom=0);
    axs[2, 1].set_xlabel('Time (min)'); axs[2, 1].set_ylabel('Active Kinase Level')


    # --- Row 3 (Plots H, I) ---
    axs[3, 0].plot(t_values_main, p21_main, label='p21_protein', color=color_p21, lw=2)
    axs[3, 0].plot(t_values_main, Wip1_main, label='Wip1_protein', color=color_wip1, linestyle='--', lw=2)
    axs[3, 0].set_title('G: p21 & Wip1 Levels'); axs[3, 0].legend(); axs[3, 0].grid(True, alpha=0.5); axs[3, 0].set_ylim(bottom=0);
    axs[3, 0].set_xlabel('Time (min)'); axs[3, 0].set_ylabel('Protein Level')

    ax_pulse_compare = axs[3, 1]
    ax_pulse_compare.plot(stimulus_durations_scan, pulse_counts_ir_scan, marker='o', ls='-', color=color_plot_ir_line, lw=1.5, ms=6, label='IR Pulses')
    ax_pulse_compare.plot(stimulus_durations_scan, pulse_counts_uv_scan, marker='s', ls='--', color=color_plot_uv_line, lw=1.5, ms=6, label='UV Pulses')
    ax_pulse_compare.plot(stimulus_durations_scan, pulse_counts_both_scan, marker='^', ls=':', color=color_plot_both_line, lw=1.5, ms=6, label='IR+UV Pulses')
    for i, dur in enumerate(stimulus_durations_scan): # Add apoptosis markers
        if apoptosis_flags_ir_scan[i]: ax_pulse_compare.plot(dur, pulse_counts_ir_scan[i], marker='x', markersize=8, color='red', mew=1.5)
        if apoptosis_flags_uv_scan[i]: ax_pulse_compare.plot(dur, pulse_counts_uv_scan[i], marker='x', markersize=8, color='red', mew=1.5)
        if apoptosis_flags_both_scan[i]: ax_pulse_compare.plot(dur, pulse_counts_both_scan[i], marker='x', markersize=8, color='darkred', mew=1.5)
    ax_pulse_compare.set_title('H: Pulses vs. Stimulus Duration'); ax_pulse_compare.legend(loc='best'); ax_pulse_compare.grid(True, ls=':', alpha=0.7)
    ax_pulse_compare.set_xticks(np.arange(0, max(stimulus_durations_scan, default=30) + 5, 5)) # Added default for empty scan
    all_pulse_counts = pulse_counts_ir_scan + pulse_counts_uv_scan + pulse_counts_both_scan
    if all_pulse_counts: max_pulses_overall = max(all_pulse_counts)
    else: max_pulses_overall = 4
    ax_pulse_compare.set_yticks(np.arange(0, max_pulses_overall + 2, 1)); ax_pulse_compare.set_ylim(bottom=-0.5)
    ax_pulse_compare.set_xlabel('Stimulus Duration (min)'); ax_pulse_compare.set_ylabel('Number of p53 Pulses')


    # --- Row 4 (Plots K, J) ---
    ax_k1 = axs[4,0]
    if sol_main.y.shape[0] > 12 :
        ax_k1.plot(t_values_main, ApoptFactor_main, label='ApoptFactor', color=color_apopt_factor, lw=2)
        if 'ApoptFactor_threshold' in params_used_for_sol_main: 
            ax_k1.axhline(params_used_for_sol_main.get('ApoptFactor_threshold', 1.0), color=color_apopt_factor, linestyle=':', lw=1.5, label='Apoptosis Threshold')
    else:
        ax_k1.text(0.5, 0.5, "ApoptFactor not in sol_main", ha='center', va='center', transform=ax_k1.transAxes)
    ax_k1.set_title('I: Pro-Apoptotic Factor'); ax_k1.legend(); ax_k1.grid(True, alpha=0.5); ax_k1.set_ylim(bottom=0);
    ax_k1.set_xlabel('Time (min)'); ax_k1.set_ylabel('ApoptFactor Level')

    ax_j_all = axs[4, 1]
    ax_j_all.plot(t_values_main, p53_free_main, label='p53_free', color=color_p53, lw=1.5)
    ax_j_all.plot(t_values_main, p53U_main, label='p53U', color=color_p53u, lw=1.5)
    ax_j_all.plot(t_values_main, p53UU_main, label='p53UU', color=color_p53uu, lw=1.5)
    ax_j_all.plot(t_values_main, Mdm2_cyt_main, label='Mdm2_cyt', color=color_mdm2cyt, lw=1.5) # Corrected label
    ax_j_all.plot(t_values_main, Mdm2P_cyt_main, label='Mdm2P_cyt', color=color_mdm2pcyt, lw=1.5)
    ax_j_all.plot(t_values_main, Mdm2_nuc_main, label='Mdm2_nuc', color=color_mdm2nuc, lw=1.5) # Corrected color
    ax_j_all.plot(t_values_main, ATM_active_main, label='ATM_act', color=color_atm, lw=1.5, ls='--')
    ax_j_all.plot(t_values_main, ATR_active_main, label='ATR_act', color=color_atr, lw=1.5, ls='--')
    ax_j_all.plot(t_values_main, p21_main, label='p21', color=color_p21, lw=1.5)
    ax_j_all.plot(t_values_main, Wip1_main, label='Wip1', color=color_wip1, lw=1.5)
    if sol_main.y.shape[0] > 12: 
        ax_j_all.plot(t_values_main, ApoptFactor_main, label='ApoptFactor', color=color_apopt_factor, lw=1.5)

    ax_j_all.set_title('J: All Species Dynamics (Main Sim)')
    ax_j_all.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax_j_all.grid(True, alpha=0.5)
    ax_j_all.set_xlabel('Time (min)')
    ax_j_all.set_ylabel('Concentration / Level')


    # --- Final Layout & Print Info ---
    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=3.0) # Adjusted w_pad for J legend
    plt.suptitle(f"P53 Network Dynamics ({cell_type})", fontsize=18, y=1.00) # y slightly adjusted
    plt.show()

    # Print Environment and Cell Info (using the pre-calculated apoptosis_info_main_sim)
    print("Environment Information:")
    print(f"UV Start: {params_used_for_sol_main.get('UV_start', 'N/A')}, UV End: {params_used_for_sol_main.get('UV_end', 'N/A')}, UV Amplitude: {params_used_for_sol_main.get('ampl_UV', 0.0)}")
    print(f"IR Start: {params_used_for_sol_main.get('IR_start', 'N/A')}, IR End: {params_used_for_sol_main.get('IR_end', 'N/A')}, IR Amplitude: {params_used_for_sol_main.get('ampl_IR', 0.0)}")

    print("Cell information (Main Simulation):")
    if apoptosis_triggered_main_sim:
        print(f"Apoptosis triggered at time: {time_of_death_main_sim:.2f} minutes")
        print(f"Reason for apoptosis: {reason_main_sim}")
    else:
        print("No apoptosis triggered in main simulation.")

    # --- Final Layout ---
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.5)
    plt.suptitle(f"P53 Network Dynamics ({cell_type})", fontsize=18, y=1.00) # y slightly adjusted
    plt.show()

    if save_fig:
        fig.savefig(f'p53_full_simulation_{cell_type}.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as 'p53_full_simulation_{cell_type}.png'.")