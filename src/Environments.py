class BasicEnvironment:
    def __init__(self, ):
        self.name = "BasicEnvironment"
        self.params = {
            # --- UV (SSB) Damage Parameters ---
            'UV_start': 0,        # Example UV start time
            'UV_end': 0,          # Example UV duration (same as IR example above for comparison)
            'ampl_UV': 0.0,         # Reduced amplitude compared to IR (was 0.3, this might be too low, let's try a bit higher but still < ampl_IR)

            'IR_start': 10,         # Default IR start for Fig 2A-C like sims
            'IR_end': 20,           # Default IR end
            'ampl_IR': 1.0,         # Amplitude of IR signal (was 'ampl')

        }
    

class AdvancedEnvironment:
    def __init__(self, ):
        self.name = "AdvancedEnvironment"
        self.params = {
            # --- UV (SSB) Damage Parameters ---
            'UV_start': 10,        # Example UV start time
            'UV_end': 700,          # Example UV duration (same as IR example above for comparison)
            'ampl_UV': 0.001,         # Reduced amplitude compared to IR (was 0.3, this might be too low, let's try a bit higher but still < ampl_IR)

            'IR_start': 10,         # Default IR start for Fig 2A-C like sims
            'IR_end': 20,           # Default IR end
            'ampl_IR': 1.,         # Amplitude of IR signal (was 'ampl')
        }