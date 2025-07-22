import copy
import numpy as np

class BaseCellParameters:
    """
    Contains core parameters for the p53-Mdm2 regulatory network,
    primarily based on the Ciliberto et al. (2005) model and
    the simpler 'p53_mdm2_system_combined_damage' ODE.
    """
    def __init__(self):
        self.params = {
            # --- p53 Dynamics ---
            'ks53': 0.055,       # Basal synthesis rate of total p53 (p53_tot/min). Paper Table 2.
            'kd53_': 0.0055,     # Basal degradation rate constant for all p53 forms (p53_tot, p53U, p53UU) (min^-1). Paper: kd53'.
            'kd53': 8.0,         # Enhanced degradation rate constant of poly-ubiquitinated p53 (p53UU) (min^-1). Paper Table 2.
            'kf': 8.8,           # Rate constant for Mdm2_nuc-mediated p53 ubiquitination (p53 -> p53U, p53U -> p53UU) ((concentration*min)^-1). Paper: kt (k_transcription/ubiquitination).
            'kr': 2.5,           # Rate constant for p53 de-ubiquitination (p53UU -> p53U, p53U -> p53) (min^-1). Paper Table 2.

            # --- Mdm2 Dynamics ---
            'ks2_': 0.0015,      # Basal synthesis rate of Mdm2_cyt (Mdm2/min). Paper: ks2'.
            'ks2': 0.006,        # Max p53-dependent synthesis rate of Mdm2_cyt (Mdm2/min). Paper Table 2.
            'm': 3,              # Hill coefficient for p53-dependent Mdm2 synthesis (dimensionless). Paper Table 2.
            'Js': 1.2,           # p53_tot concentration for half-maximal Mdm2 synthesis (concentration units). Paper: J_sm (saturation Mdm2).

            'kph': 0.05,         # Max rate of Mdm2_cyt phosphorylation to Mdm2P_cyt, inhibited by p53_tot (Mdm2_cyt/min). Paper Table 2.
            'J': 0.01,           # p53_tot concentration for half-maximal inhibition of Mdm2_cyt phosphorylation (concentration units). Paper: J_ph (phosphorylation).
            'kdeph': 6.0,        # Rate constant for Mdm2P_cyt dephosphorylation to Mdm2_cyt (min^-1). Paper Table 2.

            'ki': 14.0,          # Rate constant for Mdm2P_cyt import into the nucleus (Mdm2_nuc) (min^-1). Paper Table 2.
            'ko': 0.5,           # Rate constant for Mdm2_nuc export from the nucleus (to Mdm2P_cyt) (min^-1). Paper Table 2.
            'Vratio': 15.0,      # Volume ratio (Cytoplasm/Nucleus) (dimensionless). Paper Table 2.
            'kd2_': 0.01,        # Basal degradation rate constant for all Mdm2 forms (Mdm2_cyt, Mdm2P_cyt, Mdm2_nuc) (min^-1). Paper: k'd2.
                                 # In the paper Table 1, Mdm2_nuc degradation rate 'kd2[Mdm2_nuc]' term uses 'kd2' which is composed of k'd2 and damage-dependent terms.
                                 # Here, kd2_ is the basal part, and damage terms are added in ODE.

            # --- DSB (IR-like) Damage Parameters ---
            'ampl_IR': 1.0,      # Amplitude of IR signal (when on) (dimensionless, typically 0 or 1). Paper Table 2 (ampl).
            'IR_start': -1.0,    # Start time for IR exposure (min). (Set to negative if not used by default).
            'IR_end': -1.0,      # End time for IR exposure (min). (Set to negative if not used by default).
            'kDNA_DSB': 0.18,    # Production rate constant of DNAdam_DSB by IR signal (DNAdam/min per IR_signal unit). Paper: kDNA.
            'kdDNA_DSB': 0.017,  # p53-dependent repair rate constant of DNAdam_DSB ( (p53_tot_concentration * min)^-1 effectively, when multiplied by p53_tot). Paper: kdDNA.
            'JDNA_DSB': 1.0,     # Saturation constant for p53-dependent DNAdam_DSB repair (DNAdam concentration). Paper: Jdna.
            'Jdam_DSB': 0.2,     # Saturation constant for DNAdam_DSB effect on Mdm2_nuc degradation (DNAdam concentration). Paper: Jdam.
            'kd2__DSB': 0.01,    # Max additional degradation rate constant for Mdm2_nuc due to DNAdam_DSB (min^-1). Paper: kd2 (coefficient for damage term).

            # --- SSB (UV-like) Damage Parameters (Values are examples, adjust as needed) ---
            'ampl_UV': 1.0,      # Amplitude of UV signal (when on) (dimensionless, typically 0 or 1).
            'UV_start': -1.0,    # Start time for UV exposure (min).
            'UV_end': -1.0,      # End time for UV exposure (min).
            'kDNA_UV': 0.15,     # Production rate constant of DNAdam_UV by UV signal (DNAdam/min per UV_signal unit).
            'kdDNA_UV': 0.035,   # p53-dependent repair rate constant of DNAdam_UV ( (p53_tot_concentration * min)^-1 effectively).
            'JDNA_UV': 0.7,      # Saturation constant for p53-dependent DNAdam_UV repair (DNAdam concentration).
            'Jdam_UV': 0.25,     # Saturation constant for DNAdam_UV effect on Mdm2_nuc degradation (DNAdam concentration).
            'kd2__UV': 0.008,    # Max additional degradation rate constant for Mdm2_nuc due to DNAdam_UV (min^-1).

            # --- Basic Interpretation Thresholds (Example, can be expanded) ---
            'p53_tot_oscillation_threshold': 0.5, # Example threshold to detect p53 oscillations (p53_tot concentration)
            'DNAdam_DSB_high_threshold': 1.0,     # Example threshold for "high" DSB damage (DNAdam_DSB concentration)
            'DNAdam_UV_high_threshold': 1.0,      # Example threshold for "high" UV damage (DNAdam_UV concentration)

            'p53_apoptosis_duration_min': 1000,  # Minimum duration p53_tot must stay above its specific apoptosis threshold for commitment (min).
            'p53_thresh_apopt_DSB': 1.0,        # p53_tot level for apoptosis if DNAdam_DSB is dominant damage type (p53_tot concentration).
            'p53_thresh_apopt_UV': 1.2,         # p53_tot level for apoptosis if DNAdam_UV is dominant damage type (p53_tot concentration).
            'p53_thresh_apopt_BOTH': 0.8,       # p53_tot level for apoptosis if both damage types are significant (p53_tot concentration).
        }

class AdvancedCellParameters(BaseCellParameters):
    """
    Extends BaseCellParameters with parameters for ATM, ATR, p21, Wip1,
    a generic Pro-Apoptotic Factor, and various interpretation thresholds
    for the 'p53_full_system' ODE.
    """
    def __init__(self):
        super().__init__() # Initialize base parameters

        advanced_params = {
            # --- ATM Pathway (DSB response) ---
            'ATM_total': 1.0,            # Total concentration of ATM (arbitrary concentration units).
            'k_act_atm_dsb': 0.5,        # Rate constant for ATM activation by DNAdam_DSB ( (DNAdam_concentration * min)^-1 ).
            'k_inact_atm_wip1': 0.2,     # Rate constant for ATM_active inactivation by Wip1_protein ( (Wip1_concentration * min)^-1 ).
            'J_atm_kd2': 0.2,            # Saturation constant for ATM_active effect on Mdm2_nuc degradation (ATM_active concentration).
            'kd2__ATM_max': 0.01,        # Max additional degradation rate constant for Mdm2_nuc due to ATM_active (min^-1).
                                         # This replaces direct DNAdam_DSB effect on kd2 from BaseCell if ATM is active.

            # --- ATR Pathway (SSB/stalled replication response) ---
            'ATR_total': 1.0,            # Total concentration of ATR (arbitrary concentration units).
            'k_act_atr_uv': 0.6,         # Rate constant for ATR activation by DNAdam_UV ( (DNAdam_concentration * min)^-1 ).
            'k_inact_atr': 0.1,          # Basal inactivation rate constant for ATR_active (min^-1).
            'J_atr_kd2': 0.2,            # Saturation constant for ATR_active effect on Mdm2_nuc degradation (ATR_active concentration).
            'kd2__ATR_max': 0.01,        # Max additional degradation rate constant for Mdm2_nuc due to ATR_active (min^-1).
                                         # This replaces direct DNAdam_UV effect on kd2 from BaseCell if ATR is active.

            # --- p53-Mdm2 Interaction Modulation by Kinases ---
            'K_inhibit_kf_atm': 0.2,     # Inhibition constant: ATM_active level at which kf is halved due to ATM (ATM_active concentration). Higher value means less inhibition.
            'K_inhibit_kf_atr': 0.2,     # Inhibition constant: ATR_active level at which kf is halved due to ATR (ATR_active concentration). Higher value means less inhibition.

            # --- p21 Dynamics (Cell Cycle Arrest) ---
            'ks_p21': 0.1,               # Max synthesis rate of p21 protein by p53 (p21_concentration/min).
            'J_p21': 0.5,                # p53_tot concentration for half-maximal p21 synthesis (p53_tot concentration).
            'h_p21': 2,                  # Hill coefficient for p53-dependent p21 synthesis (dimensionless).
            'kd_p21': 0.02,              # Degradation rate constant of p21 protein (min^-1).

            # --- Wip1 Dynamics (Negative Feedback on DDR) ---
            'ks_wip1': 0.05,             # Max synthesis rate of Wip1 protein by p53 (Wip1_concentration/min).
            'J_wip1': 0.6,               # p53_tot concentration for half-maximal Wip1 synthesis (p53_tot concentration).
            'h_wip1': 2,                 # Hill coefficient for p53-dependent Wip1 synthesis (dimensionless).
            'kd_wip1': 0.01,             # Degradation rate constant of Wip1 protein (min^-1).

            # --- Pro-Apoptotic Factor Dynamics ---
            'ks_apopt': 0.03,            # Max synthesis rate of ApoptFactor by p53 (ApoptFactor_concentration/min).
            'J_apopt': 0.8,              # p53_tot concentration for half-maximal p53-driven ApoptFactor synthesis (p53_tot concentration).
            'h_apopt': 3,                # Hill coefficient for p53-driven ApoptFactor synthesis (dimensionless).
            'ks_apopt_atm': 0.002,       # Max synthesis rate of ApoptFactor driven by ATM_active (ApoptFactor_concentration/min).
            'J_apopt_atm': 0.1,          # ATM_active concentration for half-maximal ATM-driven ApoptFactor synthesis (ATM_active concentration).
            'ks_apopt_atr': 0.002,       # Max synthesis rate of ApoptFactor driven by ATR_active (ApoptFactor_concentration/min).
            'J_apopt_atr': 0.1,          # ATR_active concentration for half-maximal ATR-driven ApoptFactor synthesis (ATR_active concentration).
            'kd_apopt': 0.005,           # Degradation rate constant of ApoptFactor (min^-1).

            # --- Interpretation Thresholds (for simulation analysis, not directly in ODEs) ---
            'ApoptFactor_threshold': 1.0,       # Level of ApoptFactor above which apoptosis is considered triggered (ApoptFactor concentration).
            'ApoptFactor_duration_min': 300,    # Minimum duration ApoptFactor must stay above threshold for apoptosis commitment (min).

            'p21_G1_arrest_threshold': 0.4,     # Level of p21_protein above which cell is considered in G1 arrest (p21_protein concentration).
            'p53_stress_arrest_threshold': 0.8, # Level of p53_tot considered indicative of stress-induced arrest (p53_tot concentration).
            'ATM_active_stress_threshold': 0.3, # Level of ATM_active considered indicative of stress (ATM_active concentration).
            'ATR_active_stress_threshold': 0.3, # Level of ATR_active considered indicative of stress (ATR_active concentration).

            'DNAdam_DSB_necro_thresh': 5.0,     # Level of DNAdam_DSB above which necrosis is considered triggered (DNAdam_DSB concentration).
            'DNAdam_UV_necro_thresh': 6.0,      # Level of DNAdam_UV above which necrosis is considered triggered (DNAdam_UV concentration).
            'DNAdam_necro_duration_min': 180,   # Minimum duration DNAdam (either type) must stay above its threshold for necrosis commitment (min).

            'DNAdam_DSB_apopt_thresh': 1.5,     # Level of DNAdam_DSB for direct apoptosis (potentially bypassing p53 level checks) (DNAdam_DSB concentration).
            'DNAdam_UV_apopt_thresh': 2.0,      # Level of DNAdam_UV for direct apoptosis (DNAdam_UV concentration).
            'DNAdam_apopt_duration_min': 200,   # Minimum duration high DNAdam (either type) for direct apoptosis commitment (min).
        }
        # Update base params with advanced ones, ensuring advanced take precedence if names overlap (though they shouldn't here)
        self.params.update(advanced_params)
        self.params['p53_apoptosis_duration_min'] = 120

# --- Standardized Cell Type Classes ---

class BasicCell:
    """
    A basic cell model using the core p53-Mdm2 regulatory network
    and direct DNA damage effects, simulated by 'p53_mdm2_system_combined_damage'.
    This model does not include explicit ATM, ATR, p21, Wip1, or ApoptFactor dynamics.
    """
    def __init__(self):
        self.name = "BasicCell_Baseline"
        self.description = (
            "A baseline cell model using the core p53-Mdm2 dynamics described by ",
            "Ciliberto et al. (2005), with DNA damage directly influencing Mdm2 degradation ",
            "and p53-dependent repair. Does not include detailed kinase pathways or downstream effectors.",
        )
        # Initialize with all default base parameters
        self.params = copy.deepcopy(BaseCellParameters().params)
        # No further modifications needed for the baseline BasicCell itself

class AdvancedCell:
    def __init__(self):
        self.name = "AdvancedCell_Baseline"
        self.description = (
            "A baseline cell model with a fully functional p53-Mdm2 regulatory network, ",
            "DNA damage response pathways (ATM/ATR), and downstream effectors like p21, Wip1, ",
            "and an apoptotic factor. Represents a typical, healthy, non-specialized ",
            "mammalian cell capable of DNA repair, cell cycle arrest, and apoptosis.",
        )
        # Initialize with all default advanced parameters
        self.params = copy.deepcopy(AdvancedCellParameters().params)
        # No further modifications needed for the baseline AdvancedCell itself

class UVResistantCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "UVResistantCell"
        self.description = (
            "An AdvancedCell variant with enhanced UV resistance. Features include faster UV damage repair, ",
            "more robust ATR activation, stronger ATR-mediated Mdm2 degradation, potentially ",
            "more sensitive p21 induction, and a higher threshold for apoptosis, ",
            "prioritizing repair over cell death in response to UV.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for UV Resistance ---
        self.params['kdDNA_UV'] = 0.070
        self.params['JDNA_UV'] = 0.5
        self.params['k_act_atr_uv'] = 0.9
        self.params['kd2__ATR_max'] = 0.015
        self.params['ks_p21'] = 0.12
        self.params['J_p21'] = 0.4
        self.params['ks_apopt'] = 0.025
        self.params['J_apopt'] = 1.0
        self.params['ApoptFactor_threshold'] = 1.5
        self.params['ApoptFactor_duration_min'] = 360
        self.params['p21_G1_arrest_threshold'] = 0.35
        self.params['ATR_active_stress_threshold'] = 0.25

class ErythrocyteCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "ErythrocyteCell"
        self.description = (
            "Represents a mature mammalian red blood cell (erythrocyte), modeled as an AdvancedCell ",
            "with its p53 synthesis, DNA damage sensing, kinase signaling, and downstream ",
            "p53 targets effectively silenced due to its anucleated state.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Erythrocyte (Anucleated, No DNA, No p53 pathway) ---
        self.params['ks53'] = 0.0
        self.params['kf'] = 0.0
        self.params['kd53'] = 0.0
        self.params['kd53_'] = 0.01
        self.params['ks2_'] = 0.0
        self.params['ks2'] = 0.0
        self.params['kph'] = 0.0
        self.params['ki'] = 0.0
        self.params['ko'] = 0.0
        self.params['kd2_'] = 0.01
        self.params['kDNA_DSB'] = 0.0
        self.params['kdDNA_DSB'] = 0.0
        self.params['kDNA_UV'] = 0.0
        self.params['kdDNA_UV'] = 0.0
        self.params['ATM_total'] = 0.0
        self.params['k_act_atm_dsb'] = 0.0
        self.params['ATR_total'] = 0.0
        self.params['k_act_atr_uv'] = 0.0
        self.params['kd2__ATM_max'] = 0.0
        self.params['kd2__ATR_max'] = 0.0
        self.params['K_inhibit_kf_atm'] = 1e6
        self.params['K_inhibit_kf_atr'] = 1e6
        self.params['ks_p21'] = 0.0
        self.params['ks_wip1'] = 0.0
        self.params['ks_apopt'] = 0.0
        self.params['ks_apopt_atm'] = 0.0
        self.params['ks_apopt_atr'] = 0.0
        self.params['ApoptFactor_threshold'] = 1000.0
        self.params['ApoptFactor_duration_min'] = 1e6
        self.params['p53_thresh_apopt_DSB'] = 1000.0
        self.params['p53_thresh_apopt_UV'] = 1000.0
        self.params['p53_thresh_apopt_BOTH'] = 1000.0
        self.params['DNAdam_DSB_apopt_thresh'] = 1000.0
        self.params['DNAdam_UV_apopt_thresh'] = 1000.0
        self.params['p21_G1_arrest_threshold'] = 1000.0
        self.params['p53_stress_arrest_threshold'] = 1000.0
        self.params['ATM_active_stress_threshold'] = 1000.0
        self.params['ATR_active_stress_threshold'] = 1000.0

class MonocyteCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "MonocyteCell"
        self.description = (
            "Models a monocyte, an immune cell typically in G0/G1, as an AdvancedCell. Exhibits robust DNA damage ",
            "sensing and kinase activation, potentially strong p53 stabilization, efficient ",
            "p21 induction for cell cycle arrest, and a moderate apoptosis threshold.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Monocyte ---
        self.params['K_inhibit_kf_atm'] = 0.15
        self.params['K_inhibit_kf_atr'] = 0.15
        self.params['k_act_atm_dsb'] = 0.6
        self.params['k_act_atr_uv'] = 0.7
        self.params['kd2__ATM_max'] = 0.012
        self.params['kd2__ATR_max'] = 0.012
        self.params['ks_p21'] = 0.12
        self.params['J_p21'] = 0.4
        self.params['kd_p21'] = 0.015
        self.params['p21_G1_arrest_threshold'] = 0.3
        self.params['ApoptFactor_threshold'] = 0.9
        self.params['ApoptFactor_duration_min'] = 270
        self.params['ks_apopt_atm'] = 0.003
        self.params['ks_apopt_atr'] = 0.003
        self.params['J_apopt_atm'] = 0.08
        self.params['J_apopt_atr'] = 0.08

class CancerCellp53Mutant(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "CancerCell_p53Mutant"
        self.description = (
            "Simulates a cancer cell with a mutant/deficient p53, modeled as an AdvancedCell ",
            "where p53's ability to transactivate downstream targets is severely impaired. ",
            "Leads to loss of p53-mediated cell cycle arrest and apoptosis.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for p53 Deficiency/Mutation ---
        self.params['ks2'] = self.params['ks2_'] * 0.1
        self.params['ks_p21'] = 0.001
        self.params['ks_wip1'] = 0.001
        self.params['ks_apopt'] = 0.001
        self.params['ApoptFactor_threshold'] = 2.5
        self.params['ApoptFactor_duration_min'] = 600
        self.params['ks_apopt_atm'] = 0.0005
        self.params['ks_apopt_atr'] = 0.0005
        self.params['p53_thresh_apopt_DSB'] = 5.0
        self.params['p53_thresh_apopt_UV'] = 5.0
        self.params['p53_thresh_apopt_BOTH'] = 5.0
        self.params['p21_G1_arrest_threshold'] = 1.0

class StemCellEmbryonic(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "StemCell_Embryonic"
        self.description = (
            "Represents an embryonic-like stem cell, modeled as an AdvancedCell. Characterized by high sensitivity to DNA ",
            "damage, prioritizing apoptosis. Features rapid p53 induction, potent apoptotic response, ",
            "and less emphasis on prolonged cell cycle arrest.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Embryonic Stem Cell-like Properties ---
        self.params['ks_apopt'] = 0.06
        self.params['J_apopt'] = 0.3
        self.params['h_apopt'] = 4
        self.params['ApoptFactor_threshold'] = 0.5
        self.params['ApoptFactor_duration_min'] = 120
        self.params['ks_apopt_atm'] = 0.005
        self.params['J_apopt_atm'] = 0.05
        self.params['ks_apopt_atr'] = 0.005
        self.params['J_apopt_atr'] = 0.05
        self.params['K_inhibit_kf_atm'] = 0.1
        self.params['K_inhibit_kf_atr'] = 0.1
        self.params['kd2__ATM_max'] = 0.015
        self.params['kd2__ATR_max'] = 0.015
        self.params['ks_p21'] = 0.05
        self.params['J_p21'] = 0.6
        self.params['kd_p21'] = 0.03
        self.params['k_act_atm_dsb'] = 0.7
        self.params['k_act_atr_uv'] = 0.8

class SenescentCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "SenescentCell"
        self.description = (
            "Models a senescent cell as an AdvancedCell. Exhibits permanent cell cycle arrest ",
            "via high/stable p21, and high resistance to apoptosis.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Senescent Cell Properties ---
        self.params['ks_p21'] = 0.15
        self.params['kd_p21'] = 0.005
        self.params['p21_G1_arrest_threshold'] = 0.2
        self.params['ks_apopt'] = 0.005
        self.params['J_apopt'] = 1.5
        self.params['ApoptFactor_threshold'] = 3.0
        self.params['ApoptFactor_duration_min'] = 720
        self.params['ks_apopt_atm'] = 0.0001
        self.params['ks_apopt_atr'] = 0.0001

class CancerCellRestoredp53(CancerCellp53Mutant): # Inherits from CancerCellp53Mutant (which inherits from AdvancedCell)
    def __init__(self):
        super().__init__() # Get p53 mutant params first
        self.name = "CancerCell_Restored_p53"
        self.description = (
            "A p53-mutant cancer cell (AdvancedCell based) where p53 transactivation function is partially restored. ",
            "Investigates how re-activating p53 impacts cell fate in a cancerous background.",
        )
        adv_cell_ref_params = AdvancedCellParameters().params # For reference to normal values

        # --- Modifications from CancerCellp53Mutant ---
        self.params['ks2'] = adv_cell_ref_params['ks2'] * 0.75
        self.params['ks_p21'] = adv_cell_ref_params['ks_p21'] * 0.75
        self.params['ks_wip1'] = adv_cell_ref_params['ks_wip1'] * 0.5
        self.params['ks_apopt'] = adv_cell_ref_params['ks_apopt'] * 0.6
        self.params['ApoptFactor_threshold'] = 1.6
        self.params['ApoptFactor_duration_min'] = 400
        self.params['ks_apopt_atm'] = adv_cell_ref_params['ks_apopt_atm'] * 0.5
        self.params['ks_apopt_atr'] = adv_cell_ref_params['ks_apopt_atr'] * 0.5
        self.params['p53_thresh_apopt_DSB'] = adv_cell_ref_params['p53_thresh_apopt_DSB'] + 0.2
        self.params['p53_thresh_apopt_UV'] = adv_cell_ref_params['p53_thresh_apopt_UV'] + 0.2
        self.params['p53_thresh_apopt_BOTH'] = adv_cell_ref_params['p53_thresh_apopt_BOTH'] + 0.2

class RadioresistantCancerCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "RadioresistantCancerCell_p53wt"
        self.description = (
            "An AdvancedCell model of a cancer cell with wild-type p53 that exhibits radioresistance. ",
            "Achieved through enhanced DSB repair and increased apoptosis thresholds.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Radioresistance ---
        self.params['kdDNA_DSB'] = 0.040
        self.params['JDNA_DSB'] = 0.3
        self.params['k_act_atm_dsb'] = 0.65
        self.params['ApoptFactor_threshold'] = 1.8
        self.params['ApoptFactor_duration_min'] = 450
        self.params['ks_apopt'] = 0.02
        self.params['ks_apopt_atm'] = 0.001
        self.params['ks_p21'] = 0.15
        self.params['J_p21'] = 0.3
        self.params['kd_p21'] = 0.015
        self.params['DNAdam_DSB_necro_thresh'] = 6.5

class FibroblastCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "FibroblastCell"
        self.description = (
            "An AdvancedCell model of a differentiated fibroblast, typically quiescent (G0). ",
            "Possesses a functional p53 pathway for robust DNA damage response.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Fibroblast ---
        self.params['p21_G1_arrest_threshold'] = 0.35
        self.params['ks_p21'] = 0.11
        self.params['J_p21'] = 0.45
        self.params['ApoptFactor_threshold'] = 1.1
        self.params['ApoptFactor_duration_min'] = 330

class HepatocyteCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "HepatocyteCell"
        self.description = (
            "An AdvancedCell model of a hepatocyte, reflecting high metabolic activity and regenerative potential. ",
            "Balances robust DNA repair and apoptosis with capacity for cell cycle re-entry.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Hepatocyte ---
        self.params['kdDNA_DSB'] = 0.020
        self.params['kdDNA_UV'] = 0.040
        self.params['ApoptFactor_threshold'] = 1.4
        self.params['ApoptFactor_duration_min'] = 350
        self.params['ks_apopt'] = 0.028
        self.params['kd_p21'] = 0.025
        self.params['ks_wip1'] = 0.055

class NeuronCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "NeuronCell"
        self.description = (
            "An AdvancedCell model of a post-mitotic neuron. p53 primarily governs survival versus apoptosis. ",
            "Cell cycle arrest mechanisms via p21 are largely disabled.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Neuron ---
        self.params['ks_p21'] = 0.0001
        self.params['p21_G1_arrest_threshold'] = 10.0
        self.params['ApoptFactor_threshold'] = 0.75
        self.params['ApoptFactor_duration_min'] = 240
        self.params['ks_apopt'] = 0.035
        self.params['ks_apopt_atm'] = 0.003
        self.params['J_apopt_atm'] = 0.08
        self.params['ks_apopt_atr'] = 0.003
        self.params['J_apopt_atr'] = 0.08
        self.params['DNAdam_DSB_necro_thresh'] = 4.0
        self.params['DNAdam_necro_duration_min'] = 120

class MelanocyteCell(AdvancedCell): # Now inherits from AdvancedCell
    def __init__(self):
        super().__init__() # Initialize with AdvancedCell parameters
        self.name = "MelanocyteCell"
        self.description = (
            "An AdvancedCell model of a melanocyte, primarily responding to UV radiation. ",
            "Features a robust ATR pathway, balancing UV damage repair and apoptosis.",
        )
        # self.params are already a deepcopy of AdvancedCellParameters.params

        # --- Modifications for Melanocyte ---
        self.params['kDNA_UV'] = 0.13
        self.params['k_act_atr_uv'] = 0.75
        self.params['ATR_total'] = 1.1
        self.params['kd2__ATR_max'] = 0.013
        self.params['kdDNA_UV'] = 0.045
        self.params['JDNA_UV'] = 0.6
        self.params['K_inhibit_kf_atr'] = 0.15
        self.params['ks_p21'] = 0.11
        self.params['J_p21'] = 0.45
        self.params['p21_G1_arrest_threshold'] = 0.35
        self.params['ApoptFactor_duration_min'] = 280
        self.params['ks_apopt_atr'] = 0.0025
        self.params['J_apopt_atr'] = 0.09
        self.params['p53_thresh_apopt_UV'] = 1.0
        self.params['p53_thresh_apopt_DSB'] = 1.1
        self.params['ks_wip1'] = 0.055
        self.params['DNAdam_UV_necro_thresh'] = 5.0
        self.params['DNAdam_necro_duration_min'] = 150