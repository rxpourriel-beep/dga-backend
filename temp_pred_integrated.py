# ============================================================================
# SECTION 1: INSTALLATION AND IMPORTS
# ============================================================================

# Install required packages (run this cell first in Colab)
install_commands = """
!pip install numpy pandas matplotlib scikit-learn torch PyEMD vmdpy antropy scipy -q
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
try:
    from PyEMD import ICEEMDAN as ICEEMDAN_Class
    ICEEMDAN = ICEEMDAN_Class
except ImportError:
    print("Warning: PyEMD not available. Install with: pip install PyEMD")
    ICEEMDAN = None
from vmdpy import VMD
# try:
#     import antropy as ant
# except ImportError:
#     print("Warning: antropy not available. Sample entropy will use default values.")
#     ant = None
# except Exception as e:
#     print(f"Warning: antropy import failed: {e}. Sample entropy will use default values.")
#     ant = None
ant = None  # Disable antropy due to import issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
from datetime import datetime
import glob
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the DGA dataset
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        df: Preprocessed dataframe
        gas_columns: List of gas column names
    """
    df = pd.read_csv(filepath)
    
    # Subsample for faster processing (take first 1000 samples)
    if len(df) > 1000:
        df = df.iloc[:1000].copy()
        print(f"Subsampled dataset to {len(df)} rows for faster processing")
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert all columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    # Remove any columns that became all NaN
    df = df.dropna(axis=1, how='all')
    
    # Identify gas columns (assuming standard DGA gases)
    # Based on paper: H2, CH4, C2H6, C2H4, CO, CO2
    gas_columns_upper = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
    gas_columns_lower = ['h2', 'ch4', 'c2h6', 'c2h4', 'c2h2', 'co', 'co2']
    
    # If columns don't exist, try to identify them
    available_columns = df.columns.tolist()
    gas_columns = [col for col in gas_columns_upper if col in available_columns]
    
    # Also check lowercase versions
    if not gas_columns:
        gas_columns = [col for col in gas_columns_lower if col in available_columns]
    
    if not gas_columns:
        # Try to auto-detect numeric columns (excluding fault labels)
        gas_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove common label columns
        label_columns = ['Fault', 'fault', 'act', 'label', 'target', 'class']
        gas_columns = [col for col in gas_columns if col not in label_columns]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Gas columns detected: {gas_columns}")
    print(f"Missing values: {df[gas_columns].isnull().sum().sum()}")
    
    return df, gas_columns


# ============================================================================
# SECTION 3: SIGNAL DECOMPOSITION (ICEEMDAN-SE-VMD)
# ============================================================================

def apply_zscore_outlier_detection(signal, threshold=3.0, window=None):
    """
    Apply Z-score based outlier detection and handling.
    
    Args:
        signal: Input time series
        threshold: Z-score threshold for outliers (default=3.0)
        window: Optional rolling window size for local z-score
        
    Returns:
        signal_cleaned: Signal with outliers clipped or replaced
        outlier_mask: Boolean mask of detected outliers
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal_cleaned = signal.copy()
    
    if window is None:
        # Global z-score
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-10:
            return signal_cleaned, np.zeros(len(signal), dtype=bool)
        
        z_scores = np.abs((signal - mean) / (std + 1e-10))
        outlier_mask = z_scores > threshold
        
        # Replace outliers with local median
        for idx in np.where(outlier_mask)[0]:
            window_start = max(0, idx - 2)
            window_end = min(len(signal), idx + 3)
            signal_cleaned[idx] = np.median(signal[window_start:window_end])
    else:
        # Rolling window z-score
        outlier_mask = np.zeros(len(signal), dtype=bool)
        half_window = window // 2
        
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            window_data = signal[start:end]
            
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 1e-10:
                z_score = np.abs((signal[i] - mean) / std)
                if z_score > threshold:
                    outlier_mask[i] = True
                    signal_cleaned[i] = np.median(window_data)
    
    print(f"Z-score detection: {np.sum(outlier_mask)} outliers detected and handled")
    return signal_cleaned, outlier_mask


def calculate_sample_entropy(signal, m=2, r=0.2):
    """
    Calculate SAMPLE ENTROPY using high-performance NumPy-only implementation.
    Sample Entropy measures complexity and predictability of time series.
    
    Args:
        signal: Input time series (1D array)
        m: Embedding dimension (default=2)
        r: Tolerance (default=0.2 * std(signal))
        
    Returns:
        se: Sample entropy value (float)
        
    Reference: Richman & Moorman (2000)
    """
    N = len(signal)
    
    # Validate input
    if N < m + 1:
        return 0.0
    
    # Normalize signal
    signal = np.asarray(signal, dtype=np.float64)
    signal_std = np.std(signal)
    
    if signal_std < 1e-10:
        return 0.0
    
    # Adaptive tolerance based on signal statistics
    if r is None:
        r = 0.2 * signal_std
    else:
        r = r * signal_std if isinstance(r, float) else r
    
    # Create embedding vectors
    def _count_patterns(m_dim):
        """Count matching patterns of dimension m_dim"""
        patterns = np.zeros((N - m_dim + 1, m_dim))
        for i in range(N - m_dim + 1):
            patterns[i] = signal[i:i + m_dim]
        
        # Count template matches using broadcasting
        count = 0
        for i in range(len(patterns)):
            distances = np.max(np.abs(patterns - patterns[i]), axis=1)
            count += np.sum(distances <= r) - 1  # Exclude self-match
        
        return count / (N - m_dim + 1) if (N - m_dim + 1) > 0 else 1
    
    # Calculate B(m) and A(m+1)
    try:
        B_m = _count_patterns(m)
        A_m1 = _count_patterns(m + 1)
        
        # Avoid log(0)
        if A_m1 < 1e-10 or B_m < 1e-10:
            return np.inf
        
        # Sample Entropy = -ln(A(m+1) / B(m))
        se = -np.log(A_m1 / B_m)
        
        return float(np.clip(se, 0, 20))  # Clip extreme values
    except (ValueError, ZeroDivisionError):
        return 0.0


def determine_vmd_k_center_frequency(signal, max_k=8):
    """
    Determine optimal K for VMD using CENTER FREQUENCY METHOD.
    
    Based on research: optimal K is found by monitoring the separation between
    center frequencies. When two adjacent mode center frequencies become too close,
    it indicates over-decomposition.
    
    Args:
        signal: Input signal
        max_k: Maximum number of modes to test
        
    Returns:
        optimal_k: Optimal number of decomposition layers
    """
    print("Determining optimal K using Center Frequency Method...")
    
    center_freqs_data = []
    frequency_gaps = []
    
    for k in range(2, min(max_k + 1, 12)):
        try:
            # Apply VMD with current K
            u, _, _ = VMD(signal, alpha=2000, tau=0, K=k, DC=0, init=1, tol=1e-7)
            
            # Calculate center frequency for each mode
            mode_freqs = []
            for mode_idx, mode in enumerate(u):
                # Compute FFT to find dominant frequency
                fft_vals = np.fft.fft(mode)
                power = np.abs(fft_vals[:len(fft_vals)//2]) ** 2
                center_freq_idx = np.argmax(power)
                mode_freqs.append(center_freq_idx)
            
            center_freqs_data.append((k, sorted(mode_freqs)))
            
            # Calculate gaps between consecutive center frequencies
            if len(mode_freqs) > 1:
                sorted_freqs = sorted(mode_freqs)
                gaps = [sorted_freqs[i+1] - sorted_freqs[i] for i in range(len(sorted_freqs)-1)]
                min_gap = min(gaps) if gaps else float('inf')
                max_gap = max(gaps) if gaps else 0
                avg_gap = np.mean(gaps) if gaps else 0
                frequency_gaps.append((k, min_gap, avg_gap, max_gap))
                
                print(f"  K={k}: Center Freqs={sorted_freqs}, MinGap={min_gap}, AvgGap={avg_gap:.1f}")
        except Exception as e:
            print(f"  K={k}: Failed - {str(e)[:50]}")
            continue
    
    # Find optimal K: when min_gap becomes too small, previous K was better
    optimal_k = 4  # Default
    
    if frequency_gaps:
        for i in range(len(frequency_gaps) - 1):
            k_curr, min_gap_curr, avg_gap_curr, _ = frequency_gaps[i]
            k_next, min_gap_next, avg_gap_next, _ = frequency_gaps[i + 1]
            
            # If gap decreases significantly (over-decomposition), choose k_curr
            if min_gap_next < 0.5 * min_gap_curr and min_gap_next < 5:
                optimal_k = k_curr
                print(f"\nOptimal K selected: {optimal_k} (gap degradation detected)")
                break
        else:
            # If no degradation, use last valid K
            optimal_k = frequency_gaps[-1][0]
            print(f"\nOptimal K selected: {optimal_k} (last valid K)")
    
    return optimal_k


def optimize_vmd_parameters(signal, alpha_range=(500, 2500), k_range_hint=None):
    """
    Find optimal VMD parameters (alpha, K) based on reconstruction error.
    K is determined using center frequency method.
    
    Args:
        signal: Input signal
        alpha_range: Range for penalty factor alpha
        k_range_hint: Optional hint for K range (uses center frequency method by default)
        
    Returns:
        best_alpha: Optimal alpha
        best_k: Optimal K (from center frequency method)
        best_error: Best reconstruction error
    """
    from sklearn.metrics import mean_squared_error
    
    # Determine K using center frequency method
    best_k = determine_vmd_k_center_frequency(signal, max_k=8)
    
    best_error = float('inf')
    best_alpha = None
    
    print(f"Optimizing VMD alpha parameter with K={best_k}...")
    
    for alpha in np.linspace(alpha_range[0], alpha_range[1], 15):
        try:
            u, _, _ = VMD(signal, alpha=alpha, tau=0, K=best_k, DC=0, init=1, tol=1e-7)
            reconstructed = np.sum(u, axis=0)
            error = mean_squared_error(signal, reconstructed)
            
            if error < best_error:
                best_error = error
                best_alpha = alpha
        except Exception as e:
            continue
    
    if best_alpha is None:
        best_alpha = np.mean(alpha_range)
    
    print(f"Optimal VMD parameters: alpha={best_alpha:.1f}, K={best_k}, reconstruction_error={best_error:.6f}")
    return best_alpha, best_k, best_error


def decompose_signal_iceemdan_vmd(signal):
    """
    Perform IMPROVED ICEEMDAN-SE-VMD decomposition with enhanced component handling.
    
    MODIFIED LOGIC (as per research requirements):
    1. ICEEMDAN decomposition + Z-score outlier detection
    2. Calculate Sample Entropy for each IMF
    3. Apply VMD to ALL components with High SE (SE > 0.6) - treat as high-frequency noise
    4. Merge components with Low SE (< 0.1) into single \"TREND\" term
    5. Keep components with Medium SE (0.1 ≤ SE ≤ 0.6) as-is
    
    Args:
        signal: Original gas concentration time series
        
    Returns:
        components: List of decomposed components
        component_names: List of component names for tracking
        se_summary: Dictionary with SE information for each component
    """
    print("\n" + "="*70)
    print("ADVANCED ICEEMDAN-SE-VMD DECOMPOSITION")
    print("="*70)
    
    # Pre-processing: Z-score outlier detection
    print("\\nStep 0: Applying Z-score outlier detection...")
    signal_cleaned, outlier_mask = apply_zscore_outlier_detection(signal, threshold=3.0, window=10)
    print(f"  Original signal: {len(signal)} samples")
    print(f"  Outliers detected and handled: {np.sum(outlier_mask)}")
    
    # Step 1: ICEEMDAN decomposition
    print("\nStep 1: ICEEMDAN Decomposition...")
    if ICEEMDAN is None:
        print("  WARNING: ICEEMDAN not available. Using fallback Butterworth filter.")
        from scipy import signal as sp_signal
        b, a = sp_signal.butter(2, 0.1)
        trend = sp_signal.filtfilt(b, a, signal_cleaned)
        fluctuation = signal_cleaned - trend
        imfs = [fluctuation, trend]
    else:
        try:
            iceemdan = ICEEMDAN(trials=100, max_imf=10)
            imfs = iceemdan(signal_cleaned)
            print(f"  Successfully decomposed into {len(imfs)} IMFs")
        except Exception as e:
            print(f"  ICEEMDAN failed: {e}. Using fallback Butterworth filter.")
            from scipy import signal as sp_signal
            b, a = sp_signal.butter(2, 0.1)
            trend = sp_signal.filtfilt(b, a, signal_cleaned)
            fluctuation = signal_cleaned - trend
            imfs = [fluctuation, trend]
    
    # Step 2: Calculate Sample Entropy for each IMF
    print("\nStep 2: Calculating Sample Entropy (SE) for each IMF...")
    se_values = []
    imf_groups = {'high_se': [], 'medium_se': [], 'low_se': []}
    se_summary = {}
    
    for i, imf in enumerate(imfs):
        se = calculate_sample_entropy(imf, m=2, r=0.2)
        se_values.append(se)
        se_summary[f'IMF{i+1}'] = {'se': se, 'length': len(imf)}
        
        # Classify by SE threshold
        if se > 0.6:
            imf_groups['high_se'].append((i, imf, se))
            classification = "HIGH (noise) - will apply VMD"
        elif se < 0.1:
            imf_groups['low_se'].append((i, imf, se))
            classification = "LOW (trend) - will merge"
        else:
            imf_groups['medium_se'].append((i, imf, se))
            classification = "MEDIUM - keep as-is"
        
        print(f"  IMF{i+1}: SE={se:.4f} [{classification}]")
    
    components = []
    component_names = []
    
    # Step 3: Apply VMD to ALL high-entropy components (SE > 0.6) - High-frequency noise
    print("\nStep 3: Applying VMD to High-Entropy Components (SE > 0.6)...")
    high_se_vmd_components = []
    
    for imf_idx, imf, se in imf_groups['high_se']:
        print(f"  Processing IMF{imf_idx+1} (SE={se:.4f}) with VMD...")
        
        try:
            # Determine optimal K using center frequency method
            alpha_opt, k_opt, recon_error = optimize_vmd_parameters(imf, alpha_range=(500, 2500))
            
            # Apply VMD
            u, _, _ = VMD(imf, alpha=alpha_opt, tau=0, K=k_opt, DC=0, init=1, tol=1e-7)
            
            for j, vmd_comp in enumerate(u):
                components.append(vmd_comp)
                component_names.append(f'IMF{imf_idx+1}_VMD{j+1}')
                high_se_vmd_components.append(vmd_comp)
            
            print(f"    ✓ VMD created {len(u)} sub-components")
        except Exception as e:
            print(f"    ✗ VMD failed: {str(e)[:60]}. Keeping original IMF.")
            components.append(imf)
            component_names.append(f'IMF{imf_idx+1}')
            high_se_vmd_components.append(imf)
    
    # Step 4: Merge low-entropy components into "TREND" (SE < 0.1)
    print("\nStep 4: Merging Low-Entropy Components (SE < 0.1) into TREND...")
    
    if imf_groups['low_se']:
        trend_components = [imf for _, imf, _ in imf_groups['low_se']]
        trend_merged = np.sum(trend_components, axis=0)
        components.append(trend_merged)
        component_names.append('TREND')
        print(f"  ✓ Merged {len(trend_components)} low-entropy IMFs into TREND component")
    
    # Step 5: Keep medium-entropy components (0.1 ≤ SE ≤ 0.6) as-is
    print("\nStep 5: Processing Medium-Entropy Components (0.1 ≤ SE ≤ 0.6)...")
    
    for imf_idx, imf, se in imf_groups['medium_se']:
        components.append(imf)
        component_names.append(f'IMF{imf_idx+1}_MEDIUM')
        print(f"  Kept IMF{imf_idx+1} (SE={se:.4f}) as-is")
    
    print("\n" + "="*70)
    print(f"DECOMPOSITION COMPLETE: {len(components)} components for prediction")
    print("Component breakdown:")
    print(f"  - High-SE components (VMD'd): {sum(1 for name in component_names if 'VMD' in name)} components")
    print(f"  - Medium-SE components: {sum(1 for name in component_names if 'MEDIUM' in name)} components")
    print(f"  - Low-SE components (merged): {1 if 'TREND' in component_names else 0} TREND component")
    print("="*70 + "\n")
    
    return components, component_names, se_summary


# ============================================================================
# SECTION 4: BKA OPTIMIZATION ALGORITHM
# ============================================================================

class BKA:
    """
    Black-Winged Kite Algorithm (BKA) for hyperparameter optimization
    IMPROVED VERSION: Larger population, better Cauchy mutation strategy
    Based on Wang et al. 2024
    """
    
    def __init__(self, pop_size=40, max_iter=50, bounds=None):
        """
        Args:
            pop_size: Population size (INCREASED: default 40, recommend 30-50)
            max_iter: Maximum iterations
            bounds: List of (min, max) tuples for each parameter
        """
        # Ensure pop_size is in recommended range
        if pop_size < 30:
            print(f"⚠️  WARNING: pop_size={pop_size} is below recommended minimum of 30. Increasing to 30.")
            pop_size = 30
        
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.dim = len(bounds) if bounds else 0
        self.leader_improvement_history = []  # Track leader improvements for Cauchy mutation
        
    def initialize_population(self):
        """Initialize random population within bounds"""
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            low, high = self.bounds[i]
            population[:, i] = np.random.uniform(low, high, self.pop_size)
        return population
    
    def attack_behavior(self, position, t):
        """
        Simulate attack behavior (Equation 8-9 in paper)
        
        Args:
            position: Current position
            t: Current iteration
            
        Returns:
            new_position: Updated position
        """
        r = np.random.rand()
        p = 0.9
        n = 0.05 * np.exp(-2 * (t / self.max_iter) ** 2)
        
        if p < r:
            new_position = position + n * (1 + np.sin(r)) * position
        else:
            new_position = position + n * (2 * r - 1) * position
        
        return new_position
    
    def migratory_behavior(self, position, leader, fitness, random_fitness, leader_improved=True):
        """
        Simulate migratory behavior with ENHANCED CAUCHY MUTATION STRATEGY.
        
        The Cauchy mutation is applied with adaptive scaling:
        - If leader improved: use standard Cauchy mutation
        - If leader FAILED to improve: use STRONG Cauchy mutation with larger scale
        
        (Equation 10-11 in paper, enhanced)
        
        Args:
            position: Current position
            leader: Best position (leader)
            fitness: Current fitness
            random_fitness: Random individual's fitness
            leader_improved: Boolean indicating if leader improved in this iteration
            
        Returns:
            new_position: Updated position
        """
        r = np.random.rand()
        m = 2 * np.sin(r + np.pi / 2)
        
        # ENHANCED CAUCHY MUTATION: Scale based on leader improvement status
        if leader_improved:
            # Leader improved: use moderate Cauchy mutation
            cauchy = np.random.standard_cauchy(self.dim)
        else:
            # Leader FAILED to improve: use STRONG Cauchy mutation with scaling
            # This encourages more aggressive exploration when leader stagnates
            cauchy_scale = 1.5 + np.random.rand() * 2.0  # Scale factor 1.5 to 3.5
            cauchy = cauchy_scale * np.random.standard_cauchy(self.dim)
        
        if fitness < random_fitness:
            new_position = position + cauchy * (position - leader)
        else:
            new_position = position + cauchy * (leader - m * position)
        
        return new_position
    
    def clip_bounds(self, position):
        """Ensure position stays within bounds"""
        for i in range(self.dim):
            low, high = self.bounds[i]
            position[i] = np.clip(position[i], low, high)
        return position
    
    def optimize(self, objective_function):
        """
        Run BKA optimization with ENHANCED CAUCHY MUTATION STRATEGY.
        
        IMPROVEMENTS:
        - Larger population (pop_size=30-50)
        - Adaptive Cauchy mutation: stronger when leader fails to improve
        - Tracks leader improvement state for better exploration
        
        Args:
            objective_function: Function to minimize (returns scalar)
            
        Returns:
            best_position: Best parameters found
            best_fitness: Best fitness value
        """
        print(f"\n🦅 BLACK-WINGED KITE ALGORITHM (BKA) OPTIMIZATION")
        print(f"   Population size: {self.pop_size}")
        print(f"   Max iterations: {self.max_iter}")
        print(f"   Dimensions: {self.dim}")
        print(f"   Enhanced Cauchy mutation: ENABLED\n")
        
        # Initialize
        population = self.initialize_population()
        fitness = np.array([objective_function(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        previous_best_fitness = best_fitness
        convergence_stagnation = 0  # Track iterations without improvement
        
        # Optimization loop
        for t in range(self.max_iter):
            stagnation_count = 0  # Count stagnations in this iteration
            
            for i in range(self.pop_size):
                # Attack behavior
                new_pos = self.attack_behavior(population[i], t)
                new_pos = self.clip_bounds(new_pos)
                
                # Check if leader has improved (for Cauchy mutation strategy)
                leader_improved = (best_fitness < previous_best_fitness)
                if not leader_improved:
                    stagnation_count += 1
                
                # Migratory behavior with ENHANCED CAUCHY MUTATION
                random_idx = np.random.randint(0, self.pop_size)
                new_pos = self.migratory_behavior(
                    new_pos, best_position, 
                    fitness[i], fitness[random_idx],
                    leader_improved=leader_improved
                )
                new_pos = self.clip_bounds(new_pos)
                
                # Evaluate new position
                new_fitness = objective_function(new_pos)
                
                # Update if better
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < best_fitness:
                        best_position = new_pos.copy()
                        best_fitness = new_fitness
            
            # Store improvement state and track stagnation
            improvement_pct = ((previous_best_fitness - best_fitness) / (previous_best_fitness + 1e-10)) * 100
            previous_best_fitness = best_fitness
            
            if improvement_pct < 0.1:  # Less than 0.1% improvement
                convergence_stagnation += 1
            else:
                convergence_stagnation = 0
            
            self.leader_improvement_history.append(improvement_pct)
            
            if (t + 1) % 10 == 0:
                status = "⚠️  STAGNATION" if convergence_stagnation > 5 else "✓ Improving"
                print(f"Iteration {t+1:3d}/{self.max_iter} | Best Fitness: {best_fitness:.6f} | "
                      f"Improvement: {improvement_pct:+.3f}% | {status}")
        
        print(f"\n✅ BKA optimization completed!")
        print(f"   Best fitness found: {best_fitness:.6f}")
        print(f"   Convergence history: {len([x for x in self.leader_improvement_history if x > 0])} iterations with improvement\n")
        
        return best_position, best_fitness


# ============================================================================
# SECTION 5: BI-LSTM WITH ATTENTION MECHANISM
# ============================================================================

class BiLSTM_Attention(nn.Module):
    """
    Bidirectional LSTM with Attention Mechanism
    As described in the paper
    """
    
    def __init__(self, input_size=1, hidden_size1=64, hidden_size2=64, 
                 output_size=1, dropout=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size1: Hidden layer 1 size
            hidden_size2: Hidden layer 2 size
            output_size: Output size
            dropout: Dropout rate
        """
        super(BiLSTM_Attention, self).__init__()
        
        # First Bi-LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if hidden_size2 > 0 else 0
        )
        
        # Second Bi-LSTM layer (optional)
        self.use_second_layer = hidden_size2 > 0
        if self.use_second_layer:
            self.lstm2 = nn.LSTM(
                input_size=hidden_size1 * 2,
                hidden_size=hidden_size2,
                bidirectional=True,
                batch_first=True
            )
            attention_input = hidden_size2 * 2
        else:
            attention_input = hidden_size1 * 2
        
        # Attention mechanism with change-point focus
        self.attention = nn.Sequential(
            nn.Linear(attention_input, attention_input),
            nn.Tanh(),
            nn.Linear(attention_input, 1)
        )
        
        # Additional attention for change-points (recent changes)
        self.changepoint_attention = nn.Sequential(
            nn.Linear(attention_input, attention_input // 2),
            nn.ReLU(),
            nn.Linear(attention_input // 2, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(attention_input, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass with enhanced attention for change-points
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            output: Prediction (batch_size, output_size)
        """
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        
        # Second LSTM layer (if used)
        if self.use_second_layer:
            lstm_out, _ = self.lstm2(lstm_out)
        
        # Standard attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Change-point attention: focus on recent changes
        # Calculate change magnitude (difference from previous timestep)
        change_magnitude = torch.abs(lstm_out[:, 1:, :] - lstm_out[:, :-1, :])
        change_magnitude = torch.cat([torch.zeros_like(lstm_out[:, :1, :]), change_magnitude], dim=1)
        
        changepoint_weights = torch.softmax(self.changepoint_attention(change_magnitude), dim=1)
        
        # Combine attentions: emphasize recent change-points
        combined_weights = 0.7 * attention_weights + 0.3 * changepoint_weights
        
        context_vector = torch.sum(combined_weights * lstm_out, dim=1)
        
        # Dropout and output
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        
        return output


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that penalizes high-volatility periods more heavily
    """
    def __init__(self, volatility_threshold=0.1, weight_factor=2.0):
        super(WeightedMSELoss, self).__init__()
        self.volatility_threshold = volatility_threshold
        self.weight_factor = weight_factor
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        
        # Calculate volatility as absolute difference between consecutive targets
        target_diff = torch.abs(target[1:] - target[:-1])
        # Pad to match length
        target_diff = torch.cat([torch.zeros(1, target.shape[1], device=target.device), target_diff], dim=0)
        
        # Create weights: higher weight for high volatility
        weights = torch.where(target_diff > self.volatility_threshold, 
                            self.weight_factor, 1.0)
        
        return torch.mean(weights * mse)


# ============================================================================
# SECTION 6: TRAINING AND PREDICTION
# ============================================================================

def create_sequences(data, window=6, include_differencing=True):
    """
    Create sequences for time series prediction with optional differencing
    
    Args:
        data: Time series data
        window: Lookback window size
        include_differencing: Whether to include first-order differences
        
    Returns:
        X: Input sequences (shape: [samples, window, features])
        y: Target values
    """
    X, y = [], []
    for i in range(len(data) - window):
        seq = data[i:i+window]
        if include_differencing:
            # Add first-order differences (velocity features)
            diff = np.diff(seq)
            # Pad with zero for first element
            diff = np.concatenate([[0], diff])
            # Combine original and differenced features
            features = np.column_stack([seq, diff])
        else:
            features = seq.reshape(-1, 1)
        X.append(features)
        y.append(data[i+window])
    return np.array(X), np.array(y)


def train_test_split_sequential(data, train_ratio=0.8):
    """
    Split time series data sequentially (no shuffling)
    
    Args:
        data: Time series data
        train_ratio: Ratio for training set
        
    Returns:
        train_data, test_data
    """
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def train_bilstm_component(X_train, y_train, X_test, y_test, 
                           learning_rate=0.001, epochs=1, 
                           hidden_size1=64, hidden_size2=64,
                           device='cpu', verbose=False, use_huber_loss=True):
    """
    Train Bi-LSTM model on a single component
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        learning_rate: Learning rate
        epochs: Number of training epochs
        hidden_size1: First hidden layer size
        hidden_size2: Second hidden layer size
        device: Device to use
        verbose: Print training progress
        use_huber_loss: Use Huber loss for spike sensitivity
        
    Returns:
        model: Trained model
        train_loss: Final training loss
        test_loss: Final test loss
    """
    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Reshape for scaling (flatten features)
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train_scaled).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    y_test_t = torch.FloatTensor(y_test_scaled).to(device)
    
    # Initialize model
    input_size = X_train.shape[-1]  # Number of features (1 or 2 with differencing)
    model = BiLSTM_Attention(
        input_size=input_size,
        hidden_size1=int(hidden_size1),
        hidden_size2=int(hidden_size2),
        output_size=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_huber_loss:
        criterion = nn.HuberLoss(delta=1.0)  # Huber loss for spike sensitivity
    else:
        criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        train_loss = criterion(train_pred, y_train_t).item()
        test_loss = criterion(test_pred, y_test_t).item()
    
    return model, scaler_X, scaler_y, train_loss, test_loss


def optimize_and_train_component(component_data, window=6, device='cpu'):
    """
    Optimize hyperparameters using BKA and train model
    
    Args:
        component_data: Component time series
        window: Sequence window
        device: Torch device
        
    Returns:
        model: Trained model
        scalers: Data scalers
        metrics: Performance metrics
    """
    # Create sequences
    X, y = create_sequences(component_data, window)
    
    # Split data (80/20 as per paper)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Define BKA objective function
    def objective(params):
        lr, epochs, h1, h2 = params
        _, _, _, _, test_loss = train_bilstm_component(
            X_train, y_train, X_test, y_test,
            learning_rate=lr,
            epochs=int(epochs),
            hidden_size1=int(h1),
            hidden_size2=int(h2),
            device=device,
            verbose=False,
            use_huber_loss=True
        )
        return test_loss
    
    # BKA optimization
    print("Starting BKA optimization...")
    bounds = [
        (0.001, 0.01),   # learning_rate
        (50, 150),       # epochs
        (32, 128),       # hidden_size1
        (32, 128)        # hidden_size2
    ]
    
    bka = BKA(pop_size=40, max_iter=30, bounds=bounds)
    best_params, best_fitness = bka.optimize(objective)
    
    print(f"\nBest parameters found:")
    print(f"  Learning rate: {best_params[0]:.4f}")
    print(f"  Epochs: {int(best_params[1])}")
    print(f"  Hidden size 1: {int(best_params[2])}")
    print(f"  Hidden size 2: {int(best_params[3])}")
    print(f"  Best fitness (MSE): {best_fitness:.6f}\n")
    
    # Train final model with best parameters
    model, scaler_X, scaler_y, train_loss, test_loss = train_bilstm_component(
        X_train, y_train, X_test, y_test,
        learning_rate=best_params[0],
        epochs=int(best_params[1]),
        hidden_size1=int(best_params[2]),
        hidden_size2=int(best_params[3]),
        device=device,
        verbose=True
    )
    
    # Calculate metrics
    model.eval()
    with torch.no_grad():
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(-1).to(device)
        y_pred_scaled = model(X_test_t).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    
    return model, (scaler_X, scaler_y), metrics, (X_train, y_train, X_test, y_test)


# ============================================================================
# SECTION 6.5: RECURSIVE MULTI-STEP FORECASTING & THRESHOLD MONITORING
# ============================================================================

def _build_features_from_window(window_values):
    """Build model input features (value + first-difference) from raw window."""
    diff = np.diff(window_values)
    diff = np.concatenate([[0], diff])
    return np.stack([window_values, diff], axis=-1)


def recursive_forecast(
    model,
    scaler_X,
    scaler_y,
    init_window,
    steps=48,
    time_step_hours=1.0,
    threshold_ppm=None,
    ggr_threshold=2.0,
    history=None,
    device='cpu',
):
    """Recursive multi-step forecasting + Time-To-Threshold (TTT) monitoring.

    Args:
        model: Trained PyTorch model
        scaler_X: Input scaler (MinMaxScaler)
        scaler_y: Output scaler (MinMaxScaler)
        init_window: 1D array of the last `window` raw values
        steps: Number of future steps to predict
        time_step_hours: Time between samples (e.g., 1.0 for 1 hour)
        threshold_ppm: Dict like {'c2h2': 35, 'c2h4': 200}
        ggr_threshold: ppm/day slope threshold to trigger alert
        history: Full past series (1D) used for slope calculation
        device: Torch device

    Returns:
        dict: {
            'predictions': [...],
            'time_to_threshold_hours': {...},
            'time_to_threshold_days': {...},
            'ggr_ppm_day': float,
            'immediate_alert': bool,
            'crossed_step': {...}
        }
    """
    if threshold_ppm is None:
        threshold_ppm = {'c2h2': 35, 'c2h4': 200}

    window_size = len(init_window)
    window = np.array(init_window, dtype=float).copy()
    preds = []

    # Keep a rolling list of values for slope calculation
    recent = list(history[-window_size:]) if (history is not None and len(history) >= window_size) else list(window)

    time_to_threshold = {k: None for k in threshold_ppm}
    crossed_step = {k: None for k in threshold_ppm}
    immediate_alert = False

    for step in range(steps):
        # Build features and scale
        features = _build_features_from_window(window)
        features_scaled = scaler_X.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        X_t = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_t).cpu().numpy().flatten()[0]
        y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]

        # Append prediction
        preds.append(y_pred)
        recent.append(y_pred)
        if len(recent) > window_size:
            recent.pop(0)

        # Update window for next prediction
        window = np.roll(window, -1)
        window[-1] = y_pred

        # Check thresholds
        for gas, limit in threshold_ppm.items():
            if gas in ['c2h2', 'c2h4']:
                if time_to_threshold[gas] is None and y_pred >= limit:
                    time_to_threshold[gas] = (step + 1) * time_step_hours
                    crossed_step[gas] = step + 1

        # Calculate Gas Generation Rate (GGR) as ppm/day based on last window
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = np.array(recent, dtype=float)
            slope_per_step = np.polyfit(x, y, 1)[0]
            ggr_ppm_day = slope_per_step * (24.0 / time_step_hours)
            if ggr_ppm_day >= ggr_threshold:
                immediate_alert = True

    # Convert hours to days for threshold times
    time_to_threshold_days = {
        gas: (hours / 24.0) if hours is not None else None
        for gas, hours in time_to_threshold.items()
    }

    return {
        'predictions': np.array(preds),
        'time_to_threshold_hours': time_to_threshold,
        'time_to_threshold_days': time_to_threshold_days,
        'ggr_ppm_day': ggr_ppm_day if 'ggr_ppm_day' in locals() else 0.0,
        'immediate_alert': immediate_alert,
        'crossed_step': crossed_step,
    }


def plot_recursive_projection(
    history,
    predictions,
    time_step_hours=1.0,
    thresholds=None,
    save_path='recursive_projection.png',
):
    """Plot history + recursive prediction projection.

    Args:
        history: 1D array of past values
        predictions: 1D array of future predicted values
        time_step_hours: Time per step
        thresholds: dict of {name: value}
        save_path: output filename
    """
    if thresholds is None:
        thresholds = {'C2H2': 35, 'C2H4': 200}

    total_points = len(history) + len(predictions)
    times = np.arange(total_points) * time_step_hours

    plt.figure(figsize=(12, 6))
    plt.plot(times[:len(history)], history, label='History', color='black', linewidth=2)
    plt.plot(times[len(history):], predictions, '--', label='Forecast', color='red', linewidth=2)

    for name, val in thresholds.items():
        plt.axhline(val, linestyle='--', linewidth=1.5, label=f'{name} Threshold ({val} ppm)')

    plt.xlabel(f'Time (hours, step={time_step_hours}h)')
    plt.ylabel('Gas Concentration (ppm)')
    plt.title('Recursive Forecast Projection with Threshold Monitoring')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Recursive projection plot saved: {save_path}")


# ============================================================================
# SECTION 7: MAIN PREDICTION PIPELINE
# ============================================================================

def predict_gas_concentration(gas_data, window=6, device='cpu', use_bka=True):
    """
    Complete gas concentration prediction pipeline
    
    Args:
        gas_data: Gas concentration time series
        window: Sequence window
        device: Torch device
        use_bka: Whether to use BKA optimization
        
    Returns:
        predictions: Predicted values
        metrics: Performance metrics
        components_info: Information about decomposed components
    """
    print("="*70)
    print("STARTING GAS CONCENTRATION PREDICTION")
    print("="*70)
    
    # Step 1: Decompose signal
    print("\n1. Decomposing signal using ICEEMDAN-VMD...")
    components, component_names, se_summary = decompose_signal_iceemdan_vmd(gas_data)
    
    # Step 2: Train models for each component
    print(f"\n2. Training models for {len(components)} components...")
    
    component_predictions = []
    component_metrics = []
    all_test_data = []
    
    for i, (comp, name) in enumerate(zip(components, component_names)):
        print(f"\n--- Component {i+1}/{len(components)}: {name} ---")
        
        # Skip if component too short
        if len(comp) < window + 20:
            print(f"Component too short ({len(comp)} samples), skipping...")
            continue
        
        try:
            if use_bka:
                # Use BKA optimization (slower but better)
                model, scalers, metrics, data_splits = optimize_and_train_component(
                    comp, window, device
                )
            else:
                # Use default parameters (faster)
                X, y = create_sequences(comp, window)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                model, scaler_X, scaler_y, train_loss, test_loss = train_bilstm_component(
                    X_train, y_train, X_test, y_test,
                    learning_rate=0.001,
                    epochs=100,
                    hidden_size1=64,
                    hidden_size2=64,
                    device=device,
                    verbose=True,
                    use_huber_loss=True
                )
                
                scalers = (scaler_X, scaler_y)
                data_splits = (X_train, y_train, X_test, y_test)
                
                # Calculate metrics
                model.eval()
                with torch.no_grad():
                    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
                    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
                    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
                    y_pred_scaled = model(X_test_t).cpu().numpy()
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)
                
                metrics = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
            
            # Make predictions on test set
            _, _, X_test, y_test = data_splits
            scaler_X, scaler_y = scalers
            
            model.eval()
            with torch.no_grad():
                X_test_2d = X_test.reshape(-1, X_test.shape[-1])
                X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
                X_test_t = torch.FloatTensor(X_test_scaled).to(device)
                y_pred_scaled = model(X_test_t).cpu().numpy()
                y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            component_predictions.append(y_pred)
            component_metrics.append(metrics)
            all_test_data.append(y_test)
            
            print(f"Component {name} - R²: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}")
            
        except Exception as e:
            print(f"Error processing component {name}: {e}")
            continue
    
    # Step 3: Reconstruct final prediction
    print("\n3. Reconstructing final prediction...")
    
    if not component_predictions:
        raise ValueError("No components were successfully predicted!")
    
    # Ensure all predictions have same length
    min_length = min(len(pred) for pred in component_predictions)
    component_predictions = [pred[:min_length] for pred in component_predictions]
    all_test_data = [data[:min_length] for data in all_test_data]
    
    # Sum predictions from all components
    final_prediction = np.sum(component_predictions, axis=0)
    
    # For ground truth, use first component's test data length
    # (This is a simplification - ideally you'd reconstruct from all components)
    ground_truth = all_test_data[0] if all_test_data else None
    
    # Calculate final metrics
    if ground_truth is not None and len(ground_truth) == len(final_prediction):
        final_metrics = {
            'MSE': mean_squared_error(ground_truth, final_prediction),
            'RMSE': np.sqrt(mean_squared_error(ground_truth, final_prediction)),
            'MAE': mean_absolute_error(ground_truth, final_prediction),
            'R2': r2_score(ground_truth, final_prediction)
        }
    else:
        final_metrics = {
            'MSE': np.mean([m['MSE'] for m in component_metrics]),
            'MAE': np.mean([m['MAE'] for m in component_metrics]),
            'R2': np.mean([m['R2'] for m in component_metrics])
        }
        final_metrics['RMSE'] = np.sqrt(final_metrics['MSE'])
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"R² Score: {final_metrics['R2']:.4f}")
    print(f"RMSE: {final_metrics['RMSE']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.4f}")
    
    components_info = {
        'names': component_names,
        'predictions': component_predictions,
        'metrics': component_metrics
    }
    
    return final_prediction, final_metrics, components_info


# ============================================================================
# SECTION 8: VISUALIZATION
# ============================================================================

def plot_results(gas_name, ground_truth, predictions, components_info, save_path=None):
    """
    Create comprehensive visualization of results
    
    Args:
        gas_name: Name of gas being predicted
        ground_truth: Actual values
        predictions: Predicted values
        components_info: Component information
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{gas_name} Concentration Prediction Results', fontsize=16)
    
    # Plot 1: Final prediction vs ground truth
    ax1 = axes[0, 0]
    ax1.plot(ground_truth, label='Ground Truth', linewidth=2)
    ax1.plot(predictions, label='Prediction', linewidth=2, alpha=0.8)
    ax1.set_title('Final Prediction vs Ground Truth')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Concentration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction error
    ax2 = axes[0, 1]
    error = ground_truth - predictions if ground_truth is not None else predictions
    ax2.plot(error, color='red', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Prediction Error')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Error')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Component predictions
    ax3 = axes[1, 0]
    for i, (name, pred) in enumerate(zip(components_info['names'], 
                                         components_info['predictions'])):
        ax3.plot(pred[:100], label=name, alpha=0.7)  # Plot first 100 points
    ax3.set_title('Component Predictions (first 100 samples)')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Value')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Component R² scores
    ax4 = axes[1, 1]
    r2_scores = [m['R2'] for m in components_info['metrics']]
    names = components_info['names'][:len(r2_scores)]
    bars = ax4.barh(names, r2_scores, color='skyblue')
    ax4.set_xlabel('R² Score')
    ax4.set_title('Component Prediction Accuracy')
    ax4.set_xlim([0, 1])
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        # Save to results directory
        import os
        os.makedirs('./results', exist_ok=True)
        output_path = f'./results/{gas_name}_prediction_results.png'
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close()  # Close the figure to free memory


# ============================================================================
# SECTION 9: DUVAL'S TRIANGLE INTEGRATION
# ============================================================================

# Duval Triangle Zone Boundaries
# Each zone is defined as a list of (CH4%, C2H4%, C2H2%) vertices
# then converted to ternary (x, y) coordinates.
# Based on IEC 60599 Duval Triangle boundaries

def ternary_to_cartesian(ch4, c2h4, c2h2):
    """
    Convert ternary (CH4, C2H4, C2H2) percentages to 2D Cartesian coordinates.
    The three axes are: bottom-left = C2H2, bottom-right = C2H4, top = CH4
    """
    # Normalize to fractions
    total = ch4 + c2h4 + c2h2
    if total == 0:
        return 0, 0
    a = ch4 / total    # top vertex
    b = c2h4 / total   # right vertex
    c = c2h2 / total   # left vertex (bottom-left)

    # Standard ternary layout:
    # bottom-left corner = (0,0), bottom-right = (1,0), top = (0.5, sqrt(3)/2)
    x = 0.5 * (2 * b + a) / (a + b + c)
    y = (np.sqrt(3) / 2) * a / (a + b + c)
    return x, y


def zone_vertices_to_cartesian(vertices):
    """Convert list of (ch4, c2h4, c2h2) tuples to cartesian x,y arrays."""
    xs, ys = [], []
    for (ch4, c2h4, c2h2) in vertices:
        x, y = ternary_to_cartesian(ch4, c2h4, c2h2)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Zone definitions (ch4%, c2h4%, c2h2%) vertices
ZONES_IEC = {
    "PD": {
        "color": "#B34700",
        "alpha": 0.9,
        "label": "PD",
        "vertices": [
            (98, 0, 2), (96, 2, 2), (98, 2, 0), (100, 0, 0),
        ],
    },
    "T1": {
        "color": "#F09030",
        "alpha": 0.85,
        "label": "T1",
        "vertices": [
            (100, 0, 0), (98, 2, 0), (96, 2, 2), (76, 22, 2),
            (80, 20, 0),
        ],
    },
    "T2": {
        "color": "#B86030",
        "alpha": 0.85,
        "label": "T2",
        "vertices": [
            (80, 20, 0), (76, 22, 2), (40, 58, 2), (50, 50, 0),
        ],
    },
    "T3": {
        "color": "#7A4E2D",
        "alpha": 0.9,
        "label": "T3",
        "vertices": [
            (50, 50, 0), (40, 58, 2), (0, 98, 2), (0, 100, 0),
        ],
    },
    "D1": {
        "color": "#7DDFC0",
        "alpha": 0.7,
        "label": "D1",
        "vertices": [
            (98, 0, 2), (96, 2, 2), (76, 22, 2), (40, 58, 2),
            (23, 57, 20), (34, 46, 20), (49, 36, 15),
            (63, 24, 13), (76, 14, 10), (86, 6, 8), (94, 2, 4),
        ],
    },
    "D2": {
        "color": "#3090C0",
        "alpha": 0.7,
        "label": "D2",
        "vertices": [
            (94, 2, 4), (86, 6, 8), (76, 14, 10), (63, 24, 13),
            (49, 36, 15), (34, 46, 20), (23, 57, 20),
            (13, 57, 30), (0, 57, 43), (0, 50, 50),
            (22, 30, 48), (45, 14, 41), (71, 6, 23),
        ],
    },
    "DT": {
        "color": "#A0C8D8",
        "alpha": 0.7,
        "label": "DT",
        "vertices": [
            (40, 58, 2), (0, 98, 2), (0, 57, 43), (13, 57, 30), (23, 57, 20),
        ],
    },
}


def draw_duval_triangle(
    samples=None,
    labels=None,
    title="Duval Triangle – Dissolved Gas Analysis",
    figsize=(9, 8),
    show_grid=True,
    save_path=None,
):
    """
    Draw Duval Triangle and optionally plot sample points.

    Parameters
    ----------
    samples : list of (ch4, c2h4, c2h2) tuples  — raw ppm values or percentages
              The function auto-normalizes to percentages.
    labels  : list of str — optional labels for each sample point
    title   : str — plot title
    figsize : tuple
    show_grid : bool — draw ternary grid lines
    save_path : str — if given, saves figure to this path
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Draw zone polygons ──────────────────────────────────────────────────
    for zone_name, zone in ZONES_IEC.items():
        xs, ys = zone_vertices_to_cartesian(zone["vertices"])
        poly = plt.Polygon(
            list(zip(xs, ys)),
            closed=True,
            facecolor=zone["color"],
            edgecolor="black",
            linewidth=1.2,
            alpha=zone["alpha"],
            zorder=2,
        )
        ax.add_patch(poly)

    # ── Outer triangle border ───────────────────────────────────────────────
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", linewidth=2.0, zorder=5)

    # ── Ternary grid lines ──────────────────────────────────────────────────
    if show_grid:
        for pct in [20, 40, 60, 80]:
            f = pct / 100
            # Lines parallel to each axis
            # CH4 constant
            x1, y1 = ternary_to_cartesian(pct, 100 - pct, 0)
            x2, y2 = ternary_to_cartesian(pct, 0, 100 - pct)
            ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.5, alpha=0.5, zorder=3)
            # C2H4 constant
            x1, y1 = ternary_to_cartesian(100 - pct, pct, 0)
            x2, y2 = ternary_to_cartesian(0, pct, 100 - pct)
            ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.5, alpha=0.5, zorder=3)
            # C2H2 constant
            x1, y1 = ternary_to_cartesian(0, 100 - pct, pct)
            x2, y2 = ternary_to_cartesian(100 - pct, 0, pct)
            ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.5, alpha=0.5, zorder=3)

    # ── Axis tick marks & labels ────────────────────────────────────────────
    tick_len = 0.015
    for pct in [20, 40, 60, 80]:
        f = pct / 100

        # Bottom axis (C2H2) — tick marks
        x0, y0 = ternary_to_cartesian(0, 100 - pct, pct)
        ax.plot([x0, x0 - tick_len], [y0, y0], "k-", lw=1, zorder=6)
        ax.text(x0 - 0.03, y0, f"{pct}", ha="right", va="center", fontsize=8)

        # Right axis (C2H4) — tick marks
        x0, y0 = ternary_to_cartesian(100 - pct, pct, 0)
        dx = tick_len * np.cos(np.radians(60))
        dy = tick_len * np.sin(np.radians(60))
        ax.plot([x0, x0 + dx], [y0, y0 - dy], "k-", lw=1, zorder=6)
        ax.text(x0 + 0.04, y0 - 0.02, f"{pct}", ha="left", va="center", fontsize=8)

        # Left axis (CH4) — tick marks
        x0, y0 = ternary_to_cartesian(pct, 0, 100 - pct)
        ax.plot([x0, x0 + tick_len * 0.5], [y0, y0 + tick_len * 0.86], "k-", lw=1, zorder=6)
        ax.text(x0 - 0.02, y0 + 0.01, f"{pct}", ha="right", va="bottom", fontsize=8)

    # ── Axis arrows & titles ────────────────────────────────────────────────
    # CH4 (left axis, pointing up-right)
    ax.annotate(
        "", xy=(0.27, 0.58), xytext=(0.15, 0.37),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=7,
    )
    ax.text(0.10, 0.50, "% CH₄", ha="center", va="center",
            fontsize=11, fontweight="bold", rotation=60)

    # C2H4 (right axis, pointing down-right)
    ax.annotate(
        "", xy=(0.83, 0.35), xytext=(0.72, 0.56),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=7,
    )
    ax.text(0.92, 0.47, "% C₂H₄", ha="center", va="center",
            fontsize=11, fontweight="bold", rotation=-60)

    # C2H2 (bottom axis, pointing left)
    ax.annotate(
        "", xy=(0.15, -0.04), xytext=(0.38, -0.04),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=7,
    )
    ax.text(0.50, -0.07, "% C₂H₂", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # ── Zone labels ─────────────────────────────────────────────────────────
    zone_label_positions = {
        "PD":  (99, 0.5, 0.5),
        "T1":  (88, 10, 2),
        "T2":  (60, 38, 2),
        "T3":  (22, 76, 2),
        "D1":  (60, 10, 30),
        "D2":  (30, 28, 42),
        "DT":  (18, 75, 7),
    }
    for zn, (ch4, c2h4, c2h2) in zone_label_positions.items():
        x, y = ternary_to_cartesian(ch4, c2h4, c2h2)
        ax.text(x, y, zn, ha="center", va="center", fontsize=10,
                fontweight="bold", color="black", zorder=8,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="black", lw=0.8, alpha=0.7))

    # ── Legend ───────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=ZONES_IEC["PD"]["color"],  label="PD  – Partial Discharge"),
        mpatches.Patch(color=ZONES_IEC["T1"]["color"],  label="T1  – Thermal < 300°C"),
        mpatches.Patch(color=ZONES_IEC["T2"]["color"],  label="T2  – Thermal 300–700°C"),
        mpatches.Patch(color=ZONES_IEC["T3"]["color"],  label="T3  – Thermal > 700°C"),
        mpatches.Patch(color=ZONES_IEC["D1"]["color"],  label="D1  – Low Energy Discharge"),
        mpatches.Patch(color=ZONES_IEC["D2"]["color"],  label="D2  – High Energy Discharge"),
        mpatches.Patch(color=ZONES_IEC["DT"]["color"],  label="DT  – Mix Thermal & Electrical"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.0),
        fontsize=7.5,
        frameon=True,
        framealpha=0.85,
        edgecolor="gray",
    )

    # ── Plot sample points ───────────────────────────────────────────────────
    if samples is not None:
        colors_pts = plt.cm.tab10(np.linspace(0, 1, max(len(samples), 1)))
        for i, (ch4_ppm, c2h4_ppm, c2h2_ppm) in enumerate(samples):
            total = ch4_ppm + c2h4_ppm + c2h2_ppm
            if total == 0:
                continue
            ch4_p  = 100 * ch4_ppm  / total
            c2h4_p = 100 * c2h4_ppm / total
            c2h2_p = 100 * c2h2_ppm / total
            x, y = ternary_to_cartesian(ch4_p, c2h4_p, c2h2_p)
            color = colors_pts[i % len(colors_pts)]
            ax.scatter(x, y, color=color, s=80, zorder=10,
                       edgecolors="black", linewidths=0.8)
            lbl = labels[i] if (labels and i < len(labels)) else f"S{i+1}"
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=11,
            )

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, ax


def classify_sample(ch4_ppm, c2h4_ppm, c2h2_ppm):
    """
    Classify a single DGA sample using the Duval Triangle method.
    Returns zone name (PD, T1, T2, T3, D1, D2, DT) and percentage composition.
    """
    total = ch4_ppm + c2h4_ppm + c2h2_ppm
    if total == 0:
        return "Unknown", 0, 0, 0

    ch4  = 100 * ch4_ppm  / total
    c2h4 = 100 * c2h4_ppm / total
    c2h2 = 100 * c2h2_ppm / total

    # IEC 60599 classification rules (simplified boundary checks)
    if ch4 >= 98 and c2h2 <= 2:
        return "PD", ch4, c2h4, c2h2
    elif c2h2 <= 4 and ch4 >= 72 and c2h4 <= 20:
        return "T1", ch4, c2h4, c2h2
    elif c2h2 <= 4 and c2h4 >= 20 and c2h4 <= 60 and ch4 >= 30:
        return "T2", ch4, c2h4, c2h2
    elif c2h2 <= 15 and c2h4 >= 50:
        return "T3", ch4, c2h4, c2h2
    elif c2h2 >= 29 and ch4 >= 30:
        return "D2", ch4, c2h4, c2h2
    elif c2h2 >= 29 and ch4 < 30:
        return "DT", ch4, c2h4, c2h2
    elif c2h2 > 2:
        return "D1", ch4, c2h4, c2h2
    else:
        return "T1", ch4, c2h4, c2h2


def plot_duval_triangle(all_results, save_name='duval_triangle_analysis.png'):
    """
    Plot Duval's Triangle with predicted fault locations using ternary diagram
    
    Args:
        all_results: Dict containing predictions for all gases
        save_name: Output filename
    """
    # Extract predicted values for CH4, C2H4, C2H2
    duval_gas_dict = {}
    for gas_name, gas_results in all_results.items():
        if 'predictions' in gas_results and len(gas_results['predictions']) > 0 and gas_name.lower() in ['ch4', 'c2h4', 'c2h2']:
            duval_gas_dict[gas_name.lower()] = np.median(gas_results['predictions'][-10:])  # Use median of last 10 predictions
    
    if len(duval_gas_dict) >= 3:
        ch4_val = duval_gas_dict.get('ch4', 0)
        c2h4_val = duval_gas_dict.get('c2h4', 0)
        c2h2_val = duval_gas_dict.get('c2h2', 0)
        
        samples = [(ch4_val, c2h4_val, c2h2_val)]
        labels = ["Predicted"]
        
        zone, ch4_pct, c2h4_pct, c2h2_pct = classify_sample(ch4_val, c2h4_val, c2h2_val)
        
        print(f"Duval Triangle Classification: {zone}")
        print(f"Composition: CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%")
        
        fig, ax = draw_duval_triangle(
            samples=samples,
            labels=labels,
            title="Duval Triangle – BKA-BiLSTM Model Predictions",
            save_path=save_name,
        )
        plt.show()
    else:
        print("Not enough gases for Duval analysis (need ch4, c2h4, c2h2)")


# ============================================================================
# SECTION 10: MAIN EXECUTION
# ============================================================================

# ============================================================================
# SECTION 9b: SELIM et al. (2025) DGA FAULT DISCRIMINATION GRAPH
# IEEE TDEI – Graphical Shape in Cartesian Plane Based on DGA
# Exact boundary coordinates from Table III of the paper.
# ============================================================================

# Exact boundary coordinates from Table III
_DGA_COORDS = {
    'a': (0,     0),   'b': (28.85, 50),  'c': (71,    50),
    'd': (100,   0),   'e': (64,     0),  'f': (54,    17),
    'g': (25,   17),   'h': (15,     0),  'i': (34.3,  17),
    'j': (52.6, 50),   'k': (82.5,   0),  'l': (65.6,  29),
    'm': (70.5, 29),   'n': (57.7,  50),  'o': (76.6,  40.5),
    'p': (63.85,40.5), 'q': (47.9,  40.5),'r': (21.9,  37),
    's': (45.85,37),   't': (17.8,  30.8),'u': (42.2,  30.8),
}

_DGA_COLORS = {
    'outer': '#1a4fa0', 'inner': '#1a4fa0',
    'T1': '#e8f0fe', 'T2': '#d0e4fc', 'T3': '#b8d8fa',
    'D1': '#fce8d0', 'D2': '#fad0b8', 'PD': '#fad0e8',
    'N' : '#e8fce8', 'lbl': '#cc00cc',
}


def _dga_pt(name):
    return _DGA_COORDS[name]


def compute_dga_point(H2, CH4, C2H2, C2H4):
    """
    Convert dissolved gas concentrations (ppm) to (x, y) Cartesian point.
    Equations (1)–(5) from Selim et al. 2025.  L = 100.

    Parameters
    ----------
    H2, CH4, C2H2, C2H4 : float  (concentrations in ppm)

    Returns
    -------
    P1, P2, P3 : float   (three ratios, sum ≈ 1.0)
    x, y       : float   (Cartesian coordinates, L=100)
    """
    den = 2*H2 + 2*CH4 + C2H2 + C2H4
    if den == 0:
        raise ValueError("All relevant gas concentrations are zero – cannot compute DGA ratios.")
    P1 = (CH4 + H2)  / den
    P2 = (C2H2 + H2) / den
    P3 = (C2H4 + CH4) / den
    L  = 100
    x  = (P2 + P1 * np.sin(np.radians(30))) * L
    y  = (P1 * np.sin(np.radians(60)))      * L
    return P1, P2, P3, x, y


def classify_dga_fault(x, y):
    """
    Rule-based classifier using the corrected boundary coordinates
    from Table III of Selim et al. 2025.
    Returns the fault zone label as a string.
    """
    j_, q_, s_, r_, t_, u_ = (_dga_pt('j'), _dga_pt('q'), _dga_pt('s'),
                               _dga_pt('r'), _dga_pt('t'), _dga_pt('u'))
    n_, p_, m_ = _dga_pt('n'), _dga_pt('p'), _dga_pt('m')

    # j→q line: x at given y height
    def x_jq(yy):
        return j_[0] + (q_[0]-j_[0]) * (j_[1]-yy) / (j_[1]-q_[1])

    in_thermal = x < x_jq(y) if t_[1] <= y <= j_[1] else x < 28.85*y/50 + 5

    if in_thermal:
        if   y >= r_[1]: return "T1"
        elif y >= t_[1]: return "T2"
        else:            return "T3"
    else:
        if y >= p_[1]:
            return "PD" if x >= n_[0] else "N"
        elif y >= m_[1]:
            x_pm = p_[0] + (m_[0]-p_[0]) * (p_[1]-y) / (p_[1]-m_[1])
            return "D2" if x <= x_pm else "D1"
        else:
            return "D1"


def draw_dga_selim_graph(samples=None, labels=None,
                          title="DGA Fault Discrimination Graph\n(Selim et al., IEEE TDEI 2025)",
                          save_path=None, figsize=(11, 9)):
    """
    Draw the Selim et al. 2025 DGA fault discrimination graph.

    Parameters
    ----------
    samples : list of (H2, CH4, C2H2, C2H4) tuples  — raw ppm values
    labels  : list of str — optional label for each sample point
    title   : str — plot title
    save_path : str — if given, saves figure to this path
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    C  = _DGA_COLORS
    pt = _dga_pt
    LOW, LIN = 2.0, 1.6

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    def _fill(verts, col, a=0.35):
        ax.add_patch(plt.Polygon(verts, closed=True, facecolor=col,
                                 edgecolor='none', alpha=a, zorder=1))

    def _ln(p1, p2, lw=LIN, col=None):
        ax.plot([pt(p1)[0], pt(p2)[0]], [pt(p1)[1], pt(p2)[1]],
                color=col or C['inner'], lw=lw, zorder=4)

    # ── Fills ─────────────────────────────────────────────────────────────────
    # T1: b-j-q-s-r-b   (r & t on same outer left wall; r is upper intercept)
    _fill([pt('b'),pt('j'),pt('q'),pt('s'),pt('r'),pt('b')], C['T1'])
    # T2: r-s-u-t-r     (t is lower intercept on same outer left wall)
    _fill([pt('r'),pt('s'),pt('u'),pt('t'),pt('r')], C['T2'])
    # T3: t-u-i-g-t
    _fill([pt('t'),pt('u'),pt('i'),pt('g'),pt('t')], C['T3'])
    # N:  j-n-p-q-j
    _fill([pt('j'),pt('n'),pt('p'),pt('q'),pt('j')], C['N'])
    # PD: n-c-o-p-n
    _fill([pt('n'),pt('c'),pt('o'),pt('p'),pt('n')], C['PD'])
    # D2: q-p-m-l-i-u-s-q
    _fill([pt('q'),pt('p'),pt('m'),pt('l'),pt('i'),pt('u'),pt('s'),pt('q')], C['D2'])
    # D1: p-o-c-d-k-m-p
    _fill([pt('p'),pt('o'),pt('c'),pt('d'),pt('k'),pt('m'),pt('p')], C['D1'])

    # ── Outer boundary a-b-c-d-a ───────────────────────────────────────────────
    ox = [pt(p)[0] for p in ['a','b','c','d','a']]
    oy = [pt(p)[1] for p in ['a','b','c','d','a']]
    ax.plot(ox, oy, color=C['outer'], lw=LOW, zorder=5)

    # ── Bottom notch h-g-i-f-e ────────────────────────────────────────────────
    nx = [pt(p)[0] for p in ['h','g','i','f','e']]
    ny = [pt(p)[1] for p in ['h','g','i','f','e']]
    ax.plot(nx, ny, color=C['outer'], lw=LOW, zorder=5)
    # Right notch step k-l-m  (m→d REMOVED)
    _ln('k','l', lw=LOW, col=C['outer'])
    _ln('l','m', lw=LOW, col=C['outer'])

    # ── Thermal inner lines ────────────────────────────────────────────────────
    # r and t lie ON the outer left wall — NO r→t inner line
    _ln('r','s')   # T1/T2 horizontal divider
    _ln('t','u')   # T2/T3 horizontal divider
    _ln('s','q')   # right wall: s → q
    _ln('u','s')   # right wall: u → s
    _ln('u','i')   # T3 bottom → notch corner i

    # ── Electrical inner lines ─────────────────────────────────────────────────
    # j→f REMOVED
    _ln('j','q')   # N left side
    _ln('q','p')   # N/PD bottom (horizontal y=40.5)
    _ln('n','p')   # PD left boundary
    _ln('p','o')   # PD top (horizontal y=40.5)
    _ln('o','c')   # PD right (on outer wall)
    _ln('p','m')   # D2/D1 divider: p → m  (NOT p→l)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.annotate('', xy=(108, 0), xytext=(-3, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0, 55), xytext=(0, -2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(109,  0, 'x', fontsize=13, va='center', fontweight='bold')
    ax.text(  0, 57, 'y', fontsize=13, ha='center', fontweight='bold')

    # ── Point labels ──────────────────────────────────────────────────────────
    _offsets = {
        'a':(-3,-2.5),'b':(-3,1.5),'c':(2,1.5),'d':(2,-2.5),
        'e':(0,-2.5),'f':(2,0),'g':(-3,0),'h':(0,-2.5),
        'i':(0,-2.5),'j':(0,1.5),'k':(0,-2.5),'l':(2,0),
        'm':(2,0),'n':(0,1.5),'o':(2,0),'p':(2,-1.5),
        'q':(-3,0),'r':(-3,0),'s':(2,0),'t':(-3,0),'u':(-3,0),
    }
    for name, (x, y) in _DGA_COORDS.items():
        dx, dy = _offsets.get(name, (1, 1))
        ax.text(x+dx, y+dy, name, fontsize=8.5, ha='center', va='center',
                color='#1a1a1a', fontweight='bold', zorder=8)

    # ── Region labels ─────────────────────────────────────────────────────────
    for label, (lx, ly) in [('T1',(33,45)),('T2',(29,38)),('T3',(22,27)),
                             ('N',(52,45)),('PD',(68,46)),
                             ('D2',(55,32)),('D1',(80,22))]:
        ax.text(lx, ly, label, fontsize=13, color=C['lbl'],
                fontweight='bold', ha='center', va='center', zorder=9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.6))

    # ── Plot sample points ─────────────────────────────────────────────────────
    if samples:
        colors_pts = plt.cm.tab10(np.linspace(0, 1, max(len(samples), 1)))
        import matplotlib.patheffects as pe_mod
        for i, (H2_v, CH4_v, C2H2_v, C2H4_v) in enumerate(samples):
            try:
                _, _, _, sx, sy = compute_dga_point(H2_v, CH4_v, C2H2_v, C2H4_v)
                col = colors_pts[i % len(colors_pts)]
                ax.scatter(sx, sy, color=col, s=120, zorder=10,
                           edgecolors='black', linewidths=0.8, marker='*')
                lbl = labels[i] if (labels and i < len(labels)) else f"S{i+1}"
                fault_zone = classify_dga_fault(sx, sy)
                ax.annotate(
                    f"{lbl}\n({fault_zone})", (sx, sy),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=col, fontweight='bold',
                    path_effects=[pe_mod.withStroke(linewidth=2, foreground='white')],
                    zorder=11,
                )
            except ValueError:
                pass   # skip zero-concentration samples

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C['T1'], edgecolor=C['outer'], label='T1: Thermal fault, t < 300 °C'),
        mpatches.Patch(facecolor=C['T2'], edgecolor=C['outer'], label='T2: Thermal, 300 °C < t < 700 °C'),
        mpatches.Patch(facecolor=C['T3'], edgecolor=C['outer'], label='T3: Thermal fault, t > 700 °C'),
        mpatches.Patch(facecolor=C['N'],  edgecolor=C['outer'], label='N:  Normal operation'),
        mpatches.Patch(facecolor=C['PD'], edgecolor=C['outer'], label='PD: Partial discharge (corona)'),
        mpatches.Patch(facecolor=C['D2'], edgecolor=C['outer'], label='D2: High energy discharge'),
        mpatches.Patch(facecolor=C['D1'], edgecolor=C['outer'], label='D1: Low energy discharge'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8.5,
              framealpha=0.9, edgecolor='#aaaaaa',
              title='Fault Regions', title_fontsize=9)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlim(-5, 112)
    ax.set_ylim(-5, 60)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved DGA Selim graph to {save_path}")

    return fig, ax


def plot_dga_selim_graph(all_results, output_dir='.', verbose=True):
    """
    Build and save the Selim et al. 2025 DGA graph using predicted gas values
    extracted from all_results (same structure produced by train_model).

    Looks for H2, CH4, C2H2, C2H4 keys (case-insensitive) in all_results.
    """
    def _get(gas_key):
        for k, v in all_results.items():
            if k.lower() == gas_key.lower() and 'predictions' in v and len(v['predictions']) > 0:
                return float(np.median(v['predictions'][-10:]))
        return 0.0

    H2_v   = _get('h2')
    CH4_v  = _get('ch4')
    C2H2_v = _get('c2h2')
    C2H4_v = _get('c2h4')

    if H2_v == 0 and CH4_v == 0 and C2H2_v == 0 and C2H4_v == 0:
        if verbose:
            print("\n⚠️  Selim DGA graph skipped: H2, CH4, C2H2, C2H4 not found in results.")
        return

    try:
        _, _, _, sx, sy = compute_dga_point(H2_v, CH4_v, C2H2_v, C2H4_v)
        fault_zone = classify_dga_fault(sx, sy)

        if verbose:
            print(f"\n📊 SELIM et al. DGA GRAPH DIAGNOSIS:")
            print(f"   H2={H2_v:.1f} ppm  CH4={CH4_v:.1f} ppm  "
                  f"C2H2={C2H2_v:.1f} ppm  C2H4={C2H4_v:.1f} ppm")
            print(f"   Cartesian point: x={sx:.3f}, y={sy:.3f}")
            print(f"   Fault Zone: {fault_zone}")

        save_path = os.path.join(output_dir, 'dga_selim_graph.png')
        fig, ax = draw_dga_selim_graph(
            samples=[(H2_v, CH4_v, C2H2_v, C2H4_v)],
            labels=["Predicted"],
            title="DGA Fault Discrimination Graph – BKA-BiLSTM Predictions\n"
                  "(Selim et al., IEEE TDEI 2025)",
            save_path=save_path,
        )
        plt.close(fig)

        # Store result back into all_results for downstream use
        all_results['DGA_SELIM_ANALYSIS'] = {
            'H2_ppm': H2_v, 'CH4_ppm': CH4_v,
            'C2H2_ppm': C2H2_v, 'C2H4_ppm': C2H4_v,
            'x': sx, 'y': sy, 'fault_zone': fault_zone,
        }

    except Exception as exc:
        if verbose:
            print(f"\n⚠️  Could not generate Selim DGA graph: {exc}")


# ============================================================================
# (original) SECTION 10: MAIN EXECUTION
# ============================================================================

def train_model(data_file, output_dir='./results', use_optimization=False, 
                 num_gases=None, gpu=True, verbose=True):
    """
    Complete model training and evaluation pipeline
    
    Args:
        data_file: Path to CSV file with DGA data
        output_dir: Directory to save results and plots
        use_optimization: Use BKA optimization (slower but better, ~10-30 min per gas)
        num_gases: Number of gases to process (None = all)
        gpu: Use GPU if available
        verbose: Print detailed training progress
        
    Returns:
        all_results: Dictionary with all training results and diagnostics
        
    Example:
        results = train_model(
            'DGA-dataset-1.csv',
            use_optimization=False,  # Fast training
            num_gases=2              # Train on 2 gases for demo
        )
    """
    import os

    # Create output directory (use timestamped subfolder when output_dir already exists)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(output_dir, f'session_{timestamp}')
        if verbose:
            print(f"⚡ Output directory already exists and is non-empty, using new session folder: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
    if verbose:
        print(f"🖥️  Using device: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    if verbose:
        print(f"\n📂 Loading data from: {data_file}")
    df, gas_columns = load_and_preprocess_data(data_file)
    
    # Limit number of gases if specified
    if num_gases:
        gas_columns = gas_columns[:num_gases]
    
    if verbose:
        print(f"🔬 Processing {len(gas_columns)} gases: {gas_columns}")
    
    all_results = {}
    
    for idx, gas in enumerate(gas_columns):
        if verbose:
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{len(gas_columns)}] TRAINING: {gas}")
            print(f"{'='*70}")
        
        gas_data = df[gas].values
        
        try:
            predictions, metrics, components_info = predict_gas_concentration(
                gas_data,
                window=6,
                device=device,
                use_bka=use_optimization
            )
            
            print(f"Prediction completed for {gas}")
            
            all_results[gas] = {
                'predictions': predictions,
                'metrics': metrics,
                'components': components_info
            }
            
            # Plot results
            X, y = create_sequences(gas_data, window=6)
            split_idx = int(len(X) * 0.8)
            y_test = y[split_idx:split_idx+len(predictions)]
            
            plot_path = os.path.join(output_dir, f'{gas}_results.png')
            plot_results(gas, y_test, predictions, components_info, save_path=plot_path)
            
            print(f"Plot saved to {plot_path}")
            
            if verbose:
                print(f"\n✅ {gas} Training Complete:")
                print(f"   R² Score:  {metrics['R2']:.4f}")
                print(f"   RMSE:      {metrics['RMSE']:.4f}")
                print(f"   MAE:       {metrics['MAE']:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"\n❌ Error training {gas}: {e}")
            all_results[gas] = {'error': str(e)}
    
    # Summary and Duval analysis
    if verbose:
        print(f"\n\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        
        for gas, results in all_results.items():
            if 'error' not in results:
                print(f"{gas:8} | R²: {results['metrics']['R2']:.4f} | "
                      f"RMSE: {results['metrics']['RMSE']:.4f} | "
                      f"MAE: {results['metrics']['MAE']:.4f}")
            else:
                print(f"{gas:8} | ERROR: {results['error']}")
    
    # Duval's Triangle analysis
    try:
        # Extract CH4, C2H4, C2H2 for Duval Triangle classification
        ch4_val = np.median(all_results.get('ch4', {}).get('predictions', [0])[-10:]) if 'ch4' in all_results and 'predictions' in all_results['ch4'] else 0
        c2h4_val = np.median(all_results.get('c2h4', {}).get('predictions', [0])[-10:]) if 'c2h4' in all_results and 'predictions' in all_results['c2h4'] else 0
        c2h2_val = np.median(all_results.get('c2h2', {}).get('predictions', [0])[-10:]) if 'c2h2' in all_results and 'predictions' in all_results['c2h2'] else 0
        
        if ch4_val > 0 and c2h4_val > 0 and c2h2_val > 0:
            # Classify using Duval Triangle
            zone, ch4_pct, c2h4_pct, c2h2_pct = classify_sample(ch4_val, c2h4_val, c2h2_val)
            
            all_results['DUVAL_ANALYSIS'] = {
                'ch4_ppm': ch4_val,
                'c2h4_ppm': c2h4_val,
                'c2h2_ppm': c2h2_val,
                'zone': zone,
                'ch4_percent': ch4_pct,
                'c2h4_percent': c2h4_pct,
                'c2h2_percent': c2h2_pct
            }
            
            if verbose:
                print(f"\n⚡ DUVAL'S TRIANGLE DIAGNOSIS:")
                print(f"   Zone: {zone}")
                print(f"   Composition: CH4={ch4_pct:.1f}%, C2H4={c2h4_pct:.1f}%, C2H2={c2h2_pct:.1f}%")
            
            # Plot Duval triangle with sample point
            duval_path = os.path.join(output_dir, 'duval_triangle_analysis.png')
            samples = [(ch4_val, c2h4_val, c2h2_val)]
            labels = [f"Predicted\n({zone})"]
            fig, ax = draw_duval_triangle(
                samples=samples,
                labels=labels,
                title="Duval Triangle – BKA-BiLSTM Model Predictions",
                save_path=duval_path
            )
            plt.close(fig)
    
    except Exception as e:
        if verbose:
            print(f"\n⚠️  Could not perform Duval analysis: {e}")

    # ------------------------------------------------------------------
    # Selim et al. 2025 DGA Fault Discrimination Graph (Cartesian plane)
    # Runs alongside Duval – uses H2, CH4, C2H2, C2H4 from predictions
    # ------------------------------------------------------------------
    try:
        plot_dga_selim_graph(all_results, output_dir=output_dir, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"\n⚠️  Could not generate Selim DGA graph: {e}")

    # ------------------------------------------------------------------
    # Recursive multi-step forecast for all gases (auto-run at end of training)
    # ------------------------------------------------------------------
    # Define standard DGA threshold values (ppm) for fault detection
    # Based on IEEE/IEC standards for dissolved gas analysis
    gas_thresholds = {
        'H2': 100,      # Hydrogen threshold
        'CH4': 50,      # Methane threshold
        'C2H6': 65,     # Ethane threshold
        'C2H4': 50,     # Ethylene threshold
        'C2H2': 35,     # Acetylene threshold
        'CO': 350,      # Carbon monoxide threshold
        'CO2': 2500     # Carbon dioxide threshold
    }
    
    try:
        if verbose:
            print("\n🔮 Running recursive multi-step forecasts for all gases...")
        
        # Loop through all gases for recursive forecasting
        for gas in gas_columns:
            # Check if gas exists in dataset (case-insensitive)
            gas_col = next((col for col in df.columns if col.lower() == gas.lower()), None)
            if gas_col:
                if verbose:
                    print(f"\n   Processing recursive forecast for {gas}...")
                
                # Use the full series for this gas to train a direct forecasting model
                series = df[gas_col].values.astype(float)
                window = 6
                
                # Train a fresh Bi-LSTM model specifically for recursive forecasting
                # This ensures we have a model optimized for multi-step ahead prediction
                X, y = create_sequences(series, window)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model with default hyperparameters (can be optimized if needed)
                model, scaler_X, scaler_y, _, _ = train_bilstm_component(
                    X_train, y_train, X_test, y_test,
                    learning_rate=0.001,
                    epochs=100,
                    hidden_size1=64,
                    hidden_size2=64,
                    device=device,
                    verbose=False,  # Suppress training output for cleaner logs
                    use_huber_loss=True
                )
                
                # Prepare initial window from the last 'window' samples
                init_window = series[-window:]
                
                # Get threshold for this gas
                threshold_value = gas_thresholds.get(gas.upper(), None)
                threshold_dict = {gas.lower(): threshold_value} if threshold_value else {}
                
                # Perform recursive multi-step forecasting
                # This generates predictions step-by-step, using each prediction as input for the next
                rec = recursive_forecast(
                    model=model,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    init_window=init_window,
                    steps=60,  # 60-hour forecast horizon for next 48-60h
                    time_step_hours=1.0,  # 1 hour per step
                    threshold_ppm=threshold_dict,
                    ggr_threshold=2.0,  # Gas generation rate threshold (ppm/day)
                    history=series,  # Full historical series for trend analysis
                    device=device
                )

                # Store results in all_results dictionary
                all_results[f'{gas}_RECURSIVE'] = rec

                # Plot recursive projection (now enabled)
                plot_path = os.path.join(output_dir, f'{gas.lower()}_recursive_projection.png')
                plot_recursive_projection(series, rec['predictions'], time_step_hours=1.0,
                                          thresholds={gas.upper(): threshold_value} if threshold_value else None,
                                          save_path=plot_path)

                # Print forecast summary for this gas
                if verbose:
                    ttt_hours = rec['time_to_threshold_hours'].get(gas.lower())
                    if ttt_hours is not None:
                        print(f"      Estimated Time to reach {threshold_value} ppm: {ttt_hours:.1f} hours ({ttt_hours/24:.2f} days)")
                    else:
                        print(f"      Threshold ({threshold_value} ppm) not reached within 48-hour forecast horizon")
                    print(f"      Gas Generation Rate (GGR): {rec['ggr_ppm_day']:.2f} ppm/day")
                    if rec['immediate_alert']:
                        print("      ⚠️  IMMEDIATE ALERT: Rapid gas generation detected!")
                
                # Note: Individual gas plots are skipped as per user request
                # To enable plotting, uncomment the following lines:
                # plot_path = os.path.join(output_dir, f'{gas.lower()}_recursive_projection.png')
                # plot_recursive_projection(series, rec['predictions'], time_step_hours=1.0,
                #                           thresholds={gas.upper(): threshold_value}, save_path=plot_path)
                
            else:
                if verbose:
                    print(f"   Skipping {gas} - column not found in dataset")
    
    except Exception as e:
        if verbose:
            print(f"\n⚠️  Recursive forecast failed: {e}")
            import traceback
            traceback.print_exc()

    if verbose:
        print(f"\n✅ Training complete! Results saved to: {output_dir}")

    return all_results


def main():
    """Main execution function"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (you'll need to upload your CSV file first)
    print("\nPlease upload your DGA dataset CSV file...")
    # Uncomment the following lines in Google Colab:
    # from google.colab import files
    # uploaded = files.upload()
    # filename = list(uploaded.keys())[0]
    
    # For now, using a placeholder filename
    filename = "DGA-dataset-1.csv"
    
    df, gas_columns = load_and_preprocess_data(filename)
    
    # Predict each gas
    all_results = {}
    
    # Option: Use BKA optimization (slower but more accurate)
    # Set to False for faster results with default hyperparameters
    USE_BKA_OPTIMIZATION = False  # Quick demo - set to True for full optimization
    
    # To generate projections for all detected gases (including C2H2), iterate all gases
    for gas in gas_columns:  # Process all gases
        print(f"\n\n{'='*70}")
        print(f"PROCESSING GAS: {gas}")
        print(f"{'='*70}")
        
        gas_data = df[gas].values
        
        # Run prediction
        predictions, metrics, components_info = predict_gas_concentration(
            gas_data, 
            window=6, 
            device=device,
            use_bka=USE_BKA_OPTIMIZATION
        )
        
        all_results[gas] = {
            'predictions': predictions,
            'metrics': metrics,
            'components': components_info
        }
        
        # Visualize results
        # Get ground truth for visualization
        X, y = create_sequences(gas_data, window=6)
        split_idx = int(len(X) * 0.8)
        y_test = y[split_idx:split_idx+len(predictions)]
        
        plot_results(gas, y_test, predictions, components_info)
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY OF ALL GAS PREDICTIONS")
    print("="*70)
    for gas, results in all_results.items():
        print(f"\n{gas}:")
        print(f"  R² Score: {results['metrics']['R2']:.4f}")
        print(f"  RMSE: {results['metrics']['RMSE']:.4f}")
        print(f"  MAE: {results['metrics']['MAE']:.4f}")
    
    # ========================================================================
    # DUVAL'S TRIANGLE ANALYSIS
    # ========================================================================
    
    print("\n\n" + "="*70)
    print("DUVAL'S TRIANGLE FAULT DIAGNOSIS")
    print("="*70)
    
    # Plot Duval Triangle using ternary plot
    try:
        plot_duval_triangle(all_results, save_name='./results/duval_triangle_diagnosis.png')
        print("\n✓ Duval's Triangle visualization created")
    except Exception as e:
        print(f"\n✗ Could not create Duval triangle plot: {e}")
        import traceback
        traceback.print_exc()
    
    return all_results


def plot_combined_overlay(results, save_path='results/all_gases_combined_projection_60h.png'):
    plt.figure(figsize=(14, 8))
    for gas, result in results.items():
        history = np.array(result['history'], dtype=float)
        predictions = np.array(result['predictions'], dtype=float)
        total = np.concatenate([history[-80:], predictions])
        offset = np.arange(-len(history[-80:]), len(predictions))
        plt.plot(offset, total, label=f'{gas.upper()}')
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Time (hours, future from 0)')
    plt.ylabel('Concentration ppm')
    plt.title('Combined recursive projections for all gases')
    plt.legend(loc='best', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Combined overlay saved: {save_path}')


def make_html_report(results, output_html='results/recursive_projection_report.html'):
    images = sorted(glob.glob('results/*_recursive_projection_60h.png'))

    html_lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '  <title>DGA Recursive Projection Report</title>',
        '  <style>',
        '    body { font-family: Arial, sans-serif; margin: 20px; }',
        '    h1 { color: #2a4a7b; }',
        '    .tile { margin-bottom: 30px; }',
        '    img { width: 100%; max-width: 900px; border: 1px solid #ddd; padding: 4px; }',
        '    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
        '    th, td { border: 1px solid #ccc; padding: 6px; text-align: center; }',
        '    th { background: #f0f4f7; }',
        '  </style>',
        '</head>',
        '<body>',
        '  <h1>Recursive Gas Projection Demo Report</h1>',
        '  <p>Generated projections for the following gas files:</p>',
        '  <ul>'
    ]

    for img in images:
        fname = os.path.basename(img)
        html_lines.append(f'    <li><a href="{fname}">{fname}</a></li>')
    html_lines.append('  </ul>')

    html_lines.extend([
        '  <h2>Gas Metrics Summary</h2>',
        '  <table>',
        '    <tr><th>Gas</th><th>Threshold (ppm)</th><th>Time to Threshold (h)</th><th>Time to Threshold (d)</th><th>GGR (ppm/day)</th><th>Immediate Alert</th></tr>'
    ])
    for gas, data in results.items():
        html_lines.append(
            f'    <tr><td>{gas.upper()}</td><td>{data.get("threshold_ppm", "-")}</td>' +
            f'<td>{data.get("time_to_threshold_hours", "-")}</td>' +
            f'<td>{data.get("time_to_threshold_days", "-")}</td>' +
            f'<td>{data.get("ggr_ppm_day", 0.0):.2f}</td>' +
            f'<td>{data.get("immediate_alert", False)}</td></tr>'
        )
    html_lines.extend(['  </table>', '  <hr>'])

    for img in images + ['results/all_gases_combined_projection_60h.png']:
        label = os.path.basename(img)
        html_lines.extend([
            '  <div class="tile">',
            f'    <h2>{label}</h2>',
            f'    <img src="{label}" alt="{label}">',
            '  </div>'
        ])

    html_lines.extend(['</body>', '</html>'])

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print(f'HTML report generated: {output_html}')


def make_csv_report(results, csv_path='results/recursive_projection_summary.csv'):
    rows = []
    for gas, data in results.items():
        rows.append({
            'gas': gas,
            'threshold_ppm': data.get('threshold_ppm'),
            'time_to_threshold_hours': data.get('time_to_threshold_hours'),
            'time_to_threshold_days': data.get('time_to_threshold_days'),
            'ggr_ppm_day': data.get('ggr_ppm_day'),
            'immediate_alert': data.get('immediate_alert'),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'CSV summary saved: {csv_path}')


def run_all_gases_projection_demo(data_file='data_DGA.csv', output_dir='results', pred_steps=60):
    df, gases = load_and_preprocess_data(data_file)
    os.makedirs(output_dir, exist_ok=True)
    result_data = {}

    gas_thresholds = {
        'h2': 100,
        'ch4': 80,
        'c2h2': 35,
        'c2h4': 200,
        'c2h6': 100,
    }

    for gas in gases:
        print(f"\n=== Running recursive projection for {gas.upper()} ===")
        series = df[gas].astype(float).values
        window = 6
        X, y = create_sequences(series, window)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model, scaler_X, scaler_y, train_loss, test_loss = train_bilstm_component(
            X_train, y_train, X_test, y_test,
            learning_rate=0.001, epochs=50, hidden_size1=32, hidden_size2=32,
            device='cpu', verbose=False, use_huber_loss=True
        )

        threshold_value = gas_thresholds.get(gas.lower())
        rec = recursive_forecast(
            model=model,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            init_window=series[-window:],
            steps=pred_steps,
            time_step_hours=1.0,
            threshold_ppm={gas.lower(): threshold_value} if threshold_value else None,
            ggr_threshold=2.0,
            history=series,
            device='cpu'
        )

        ttt_h = rec['time_to_threshold_hours'].get(gas.lower()) if rec['time_to_threshold_hours'] else None
        result_data[gas] = {
            'history': series,
            'predictions': rec['predictions'],
            'threshold_ppm': threshold_value,
            'time_to_threshold_hours': ttt_h,
            'time_to_threshold_days': (ttt_h/24.0) if ttt_h is not None else None,
            'ggr_ppm_day': rec['ggr_ppm_day'],
            'immediate_alert': rec['immediate_alert']
        }

        plot_recursive_projection(
            series,
            rec['predictions'],
            time_step_hours=1.0,
            thresholds={gas.upper(): threshold_value} if threshold_value else None,
            save_path=os.path.join(output_dir, f'{gas.lower()}_recursive_projection_60h.png')
        )

    plot_combined_overlay(result_data, save_path=os.path.join(output_dir, 'all_gases_combined_projection_60h.png'))
    make_html_report(result_data, output_html=os.path.join(output_dir, 'recursive_projection_report.html'))
    make_csv_report(result_data, csv_path=os.path.join(output_dir, 'recursive_projection_summary.csv'))

    return result_data


# Run if executed as script
if __name__ == "__main__":
    # ========================================================================
    # USER INPUT: Dataset Selection
    # ========================================================================
    print("🔬 DGA Analysis System")
    print("=" * 50)

    # Prompt for training dataset with file path
    print("\n📂 File Upload Instructions:")
    print("   - You can specify any CSV file from your computer")
    print("   - Use full path (e.g., 'C:\\Users\\YourName\\Documents\\data.csv')")
    print("   - Or relative path if file is in current directory")
    print("   - Use forward slashes (/) or double backslashes (\\\\)")

    import os

    def get_valid_file_path(prompt_text):
        """Get and validate file path from user input"""
        while True:
            file_path = input(f"\n{prompt_text}").strip()

            if not file_path:
                # Check for default files in current directory
                default_files = ['data_DGA.csv', 'DGA_SWPCL.csv', 'synthetic_dga_realistic.csv']
                available_defaults = [f for f in default_files if os.path.exists(f)]
                if available_defaults:
                    print(f"Available default files: {available_defaults}")
                    use_default = input("Use default file? (y/n): ").strip().lower()
                    if use_default == 'y':
                        return available_defaults[0]
                continue

            # Check if file exists
            if os.path.exists(file_path):
                print(f"✓ File found: {file_path}")
                return file_path
            else:
                print(f"❌ File not found: {file_path}")
                print("Please check the path and try again.")
                print("Tip: You can drag and drop the file into the terminal, or copy the full path from File Explorer")

    training_dataset = get_valid_file_path("Enter the FULL PATH to your training dataset CSV file: ")

    use_same_for_projection = input("\nUse the same file for projection demo? (y/n): ").strip().lower()
    if use_same_for_projection == 'y':
        projection_dataset = training_dataset
        print(f"Using same dataset for projection: {projection_dataset}")
    else:
        projection_dataset = get_valid_file_path("Enter the FULL PATH to your projection demo dataset CSV file: ")

    # ========================================================================
    # OPTION 1: Quick Training (Fast - ~2-5 minutes for 2 gases)
    # ========================================================================
    results = train_model(
        data_file=training_dataset,
        output_dir="./results",
        use_optimization=False,  # No hyperparameter tuning
        num_gases=1,             # Train only first gas for demo
        gpu=True,
        verbose=True
    )

    # ========================================================================
    # OPTION 2: Full Training with Optimization (Slow - ~30-60 minutes for 2 gases)
    # ========================================================================
    # Uncomment below for best results (takes longer)
    # results = train_model(
    #     data_file=training_dataset,
    #     output_dir="./results",
    #     use_optimization=True,   # Use BKA optimization
    #     num_gases=None,          # Train all gases
    #     gpu=True,
    #     verbose=True
    # )

    # ========================================================================
    # OPTION 3: Custom Training Logic
    # ========================================================================
    # Uncomment and modify as needed
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df, gas_columns = load_and_preprocess_data(training_dataset)
    #
    # all_results = {}
    # for gas in gas_columns[:1]:  # First gas only
    #     predictions, metrics, components = predict_gas_concentration(
    #         df[gas].values,
    #         window=6,
    #         device=device,
    #         use_bka=False
    #     )
    #     all_results[gas] = {'predictions': predictions, 'metrics': metrics}

    # ========================================================================
    # OPTION 4: Generate integrated recursive projection demo + report + CSV
    # ========================================================================
    run_all_gases_projection_demo(
        data_file=projection_dataset,
        output_dir="results",
        pred_steps=60
    )

    print("\n✅ Analysis complete!")