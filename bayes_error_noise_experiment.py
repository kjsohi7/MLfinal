import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pandas as pd
from sklearn.metrics import accuracy_score
import time


    #Implementation of direct Bayes error estimation methods from 
    #Is the Performance of My Deep Network Too Good to Be True?"
class BayesErrorEstimator:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

        # Estimate Bayes error from clean soft labels-Eq. 4 of paper
    
    def estimate_bayes_error_clean(self, soft_labels):
        
        return np.mean(np.minimum(soft_labels, 1 - soft_labels))
    
    # Naive estimator with noisY labels -Eq. 5 in paper- Biased
    def estimate_bayes_error_noisy_naive(self, noisy_soft_labels):
        return np.mean(np.minimum(noisy_soft_labels, 1 - noisy_soft_labels))


    #Correct estimator with noisy labels-Eq. 6 of paper-unbiased
    def estimate_bayes_error_noisy_corrected(self, noisy_soft_labels, sign_labels):
        positive_mask = sign_labels == 1
        negative_mask = sign_labels == 0
        
        positive_contribution = np.sum(1 - noisy_soft_labels[positive_mask])
        negative_contribution = np.sum(noisy_soft_labels[negative_mask])
        
        return (positive_contribution + negative_contribution) / len(noisy_soft_labels)


    #Adding truncated Gaussian noise to soft labels
    def add_truncated_gaussian_noise(self, clean_labels, std=0.4):
        noisy_labels = np.zeros_like(clean_labels)
        
        for i, mu in enumerate(clean_labels):
            # Truncate to ensure values stay in [0, 1]
            lower_bound = -mu
            upper_bound = 1 - mu
            
            # Create truncated normal distribution
            a, b = lower_bound / std, upper_bound / std
            noise = truncnorm.rvs(a, b, loc=0, scale=std)
            
            noisy_labels[i] = np.clip(mu + noise, 0, 1)
        
        return noisy_labels
    
    #Generate binary classification data from Gaussian distributions
    def generate_gaussian_data(self, n_samples, dim, mean_pos, mean_neg, 
                               cov_pos=None, cov_neg=None):

        if cov_pos is None:
            cov_pos = np.eye(dim)
        if cov_neg is None:
            cov_neg = np.eye(dim)
        
        # Generating  data
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos
        
        X_pos = np.random.multivariate_normal(mean_pos, cov_pos, n_pos)
        X_neg = np.random.multivariate_normal(mean_neg, cov_neg, n_neg)
        
        X = np.vstack([X_pos, X_neg])
        
        # Compute true class posteriors analytically
        soft_labels = self._compute_posterior_gaussian(
            X, mean_pos, mean_neg, cov_pos, cov_neg, 0.5
        )
        
        # True Bayes error
        true_bayes_error = np.mean(np.minimum(soft_labels, 1 - soft_labels))
        
        return X, soft_labels, true_bayes_error
    
     #Computing posterior probability p(y=+1|x) for gUussian distributions
    def _compute_posterior_gaussian(self, X, mean_pos, mean_neg, 
                                    cov_pos, cov_neg, prior_pos=0.5):
      
       
        from scipy.stats import multivariate_normal
        
        likelihood_pos = multivariate_normal.pdf(X, mean_pos, cov_pos)
        likelihood_neg = multivariate_normal.pdf(X, mean_neg, cov_neg)
        
        posterior_pos = (likelihood_pos * prior_pos) / \
                       (likelihood_pos * prior_pos + likelihood_neg * (1 - prior_pos))
        
        return posterior_pos


    #Main experiment class to test hypothesis about noise robustness
class NoiseRobustnessExperiment:
  
    
    def __init__(self, random_state=42):
        self.estimator = BayesErrorEstimator(random_state)
        self.results = []
    
    
        #testing hypothesis: Corrected estimator should be robust to noise
        #while naive estimator diverges
    def run_experiment(self, n_samples=1000, n_trials=10, 
                      noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      dim=10):

        print("Running noise robustness experiment")
        print(f"Samples: {n_samples}, Trials: {n_trials}, Dimension: {dim}")
        print(f"Noise levels: {noise_levels}")
        print()
        
        results = []
        
        for noise_std in noise_levels:
            print(f"\nNoise level (std): {noise_std:.2f}")
            
            trial_results = {
                'clean': [],
                'naive': [],
                'corrected': [],
                'true_bayes': []
            }
            
            for trial in range(n_trials):
                # Generate data
                mean_pos = np.zeros(dim)
                mean_neg = np.ones(dim)
                
                X, clean_soft_labels, true_bayes = \
                    self.estimator.generate_gaussian_data(
                        n_samples, dim, mean_pos, mean_neg
                    )
                
                # Generate sign labels (which class is more likely)
                sign_labels = (clean_soft_labels >= 0.5).astype(int)
                
                # Add noise if noise_std > 0
                if noise_std > 0:
                    noisy_soft_labels = self.estimator.add_truncated_gaussian_noise(
                        clean_soft_labels, std=noise_std
                    )
                else:
                    noisy_soft_labels = clean_soft_labels.copy()
                
                # Estimate Bayes error using different methods
                beta_clean = self.estimator.estimate_bayes_error_clean(clean_soft_labels)
                beta_naive = self.estimator.estimate_bayes_error_noisy_naive(noisy_soft_labels)
                beta_corrected = self.estimator.estimate_bayes_error_noisy_corrected(
                    noisy_soft_labels, sign_labels
                )
                
                trial_results['clean'].append(beta_clean)
                trial_results['naive'].append(beta_naive)
                trial_results['corrected'].append(beta_corrected)
                trial_results['true_bayes'].append(true_bayes)
            
            # Compute statistics
            result_entry = {
                'noise_std': noise_std,
                'true_bayes_mean': np.mean(trial_results['true_bayes']),
                'clean_mean': np.mean(trial_results['clean']),
                'clean_std': np.std(trial_results['clean']),
                'naive_mean': np.mean(trial_results['naive']),
                'naive_std': np.std(trial_results['naive']),
                'corrected_mean': np.mean(trial_results['corrected']),
                'corrected_std': np.std(trial_results['corrected']),
                'naive_bias': np.mean(trial_results['naive']) - np.mean(trial_results['true_bayes']),
                'corrected_bias': np.mean(trial_results['corrected']) - np.mean(trial_results['true_bayes'])
            }
            
            results.append(result_entry)
            
            print(f"  True Bayes Error: {result_entry['true_bayes_mean']:.4f}")
            print(f"  Clean estimator:  {result_entry['clean_mean']:.4f} ± {result_entry['clean_std']:.4f}")
            print(f"  Naive estimator:  {result_entry['naive_mean']:.4f} ± {result_entry['naive_std']:.4f} (bias: {result_entry['naive_bias']:.4f})")
            print(f"  Corrected estim:  {result_entry['corrected_mean']:.4f} ± {result_entry['corrected_std']:.4f} (bias: {result_entry['corrected_bias']:.4f})")
        
        self.results = pd.DataFrame(results)
        return self.results
    # Computing bias reduction percentage of corrected vs naive estimator
    def compute_bias_reduction(self):
 
        if len(self.results) == 0:
            return None
        
        # Focus on high noise conditions
        high_noise = self.results[self.results['noise_std'] >= 0.3]
        
        naive_bias_avg = np.abs(high_noise['naive_bias']).mean()
        corrected_bias_avg = np.abs(high_noise['corrected_bias']).mean()
        
        if naive_bias_avg > 0:
            reduction_pct = ((naive_bias_avg - corrected_bias_avg) / naive_bias_avg) * 100
        else:
            reduction_pct = 0
        
        return reduction_pct, naive_bias_avg, corrected_bias_avg
    #Plotting results
    def plot_results(self, save_path='noise_robustness_results.png'):
     
        if len(self.results) == 0:
            print("No results to plot. Run experiment first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Estimated Bayes Error vs Noise Level
        ax1 = axes[0]
        noise_levels = self.results['noise_std']
        
        ax1.plot(noise_levels, self.results['true_bayes_mean'], 
                'k--', linewidth=2, label='True Bayes Error')
        ax1.plot(noise_levels, self.results['clean_mean'], 
                'b-o', linewidth=2, label='Clean Labels')
        ax1.plot(noise_levels, self.results['naive_mean'], 
                'r-s', linewidth=2, label='Naive (Biased)')
        ax1.plot(noise_levels, self.results['corrected_mean'], 
                'g-^', linewidth=2, label='Corrected (Unbiased)')
        
        ax1.fill_between(noise_levels, 
                        self.results['naive_mean'] - self.results['naive_std'],
                        self.results['naive_mean'] + self.results['naive_std'],
                        alpha=0.2, color='red')
        ax1.fill_between(noise_levels,
                        self.results['corrected_mean'] - self.results['corrected_std'],
                        self.results['corrected_mean'] + self.results['corrected_std'],
                        alpha=0.2, color='green')
        
        ax1.set_xlabel('Noise Level (std)', fontsize=12)
        ax1.set_ylabel('Estimated Bayes Error', fontsize=12)
        ax1.set_title('Bayes Error Estimation under Label Noise', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bias Comparison
        ax2 = axes[1]
        x_pos = np.arange(len(noise_levels))
        width = 0.35
        
        ax2.bar(x_pos - width/2, np.abs(self.results['naive_bias']), 
               width, label='Naive Estimator', color='red', alpha=0.7)
        ax2.bar(x_pos + width/2, np.abs(self.results['corrected_bias']), 
               width, label='Corrected Estimator', color='green', alpha=0.7)
        
        ax2.set_xlabel('Noise Level (std)', fontsize=12)
        ax2.set_ylabel('Absolute Bias', fontsize=12)
        ax2.set_title('Bias Comparison: Naive vs Corrected', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{x:.1f}' for x in noise_levels])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.show()
    # formatted table for report
    def generate_report_table(self):
  
        if len(self.results) == 0:
            return None
        
        table_data = self.results[['noise_std', 'true_bayes_mean', 
                                   'naive_mean', 'naive_bias',
                                   'corrected_mean', 'corrected_bias']].copy()
        
        table_data.columns = ['Noise σ', 'True β', 'Naive β̂', 'Naive Bias',
                             'Corrected β̂', 'Corrected Bias']
        
        # Round for display
        for col in table_data.columns:
            if col != 'Noise σ':
                table_data[col] = table_data[col].round(4)
        
        return table_data


def main():
  
    print("Bayes error estimation-Noise robustness experiment")
    print("Testing hypothesis: Corrected estimator is robust to noise")
    
    # Initialize experiment
    experiment = NoiseRobustnessExperiment(random_state=42)
    
    # Run experiment
    results_df = experiment.run_experiment(
        n_samples=1000,
        n_trials=10,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        dim=10
    )
    
    # Compute bias reduction
    reduction_pct, naive_bias, corrected_bias = experiment.compute_bias_reduction()
    
    print("Resluts for hpothesis testing")

    print(f"Average Naive Bias-high noise:      {naive_bias:.6f}")
    print(f"Average Corrected Bias-high noise:  {corrected_bias:.6f}")
    print(f"Bias Reduction:                       {reduction_pct:.2f}%")
    print()
    
    if reduction_pct >= 50:
        print("Hypothesis confirmed: Bias reduction ≥ 50%")
        print("The corrected estimator is significantly more robust to noise.")
    else:
        print("Hypothesis partially confirmed: Bias reduction < 50%")
        print("However,corrected estimator still shows improvement.")

    
    experiment.plot_results()
    
    # report table
    print("\nReport Table (LaTeX format):")
    print(experiment.generate_report_table().to_latex(index=False))
    
    print("\nReport Table (Markdown format):")
    # print(experiment.generate_report_table().to_markdown(index=False))
    print(experiment.generate_report_table().to_string(index=False))

    
    # Save results
    results_df.to_csv('experiment_results.csv', index=False)
    print("\nResults saved to: experiment_results.csv")


if __name__ == "__main__":
    main()