import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import os

# Creating results directory
os.makedirs('results', exist_ok=True)

class CIFAR10BayesErrorEstimator:

    #Bayes Error Estimation on CIFAR-10 with Binary Classification
 
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    #Load CIFAR-10 and convert to binary classification
    def load_cifar10_binary(self, positive_classes, negative_classes):
    
        # print(f"\nLoading CIFAR-10...")
        print(f"Positive classes: {positive_classes}")
        print(f"Negative classes: {negative_classes}")
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        pos_names = [class_names[i] for i in positive_classes]
        neg_names = [class_names[i] for i in negative_classes]
        print(f"Positive: {pos_names}")
        print(f"Negative: {neg_names}")
        
        # Load with data augmentation for more uncertainty
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        
        def filter_binary(dataset, pos_classes, neg_classes):
            data_list = []
            labels_list = []
            
            for img, label in dataset:
                if label in pos_classes:
                    data_list.append(img)
                    labels_list.append(1)
                elif label in neg_classes:
                    data_list.append(img)
                    labels_list.append(0)
            
            data = torch.stack(data_list)
            labels = torch.tensor(labels_list)
            return data, labels
        
        X_train, y_train = filter_binary(trainset, positive_classes, negative_classes)
        X_test, y_test = filter_binary(testset, positive_classes, negative_classes)
        
        print(f"Training samples: {len(y_train)} ({y_train.sum()} positive, {len(y_train)-y_train.sum()} negative)")
        print(f"Test samples: {len(y_test)} ({y_test.sum()} positive, {len(y_test)-y_test.sum()} negative)")
        
        return X_train, y_train, X_test, y_test
    
    def train_classifier_for_soft_labels(self, X_train, y_train, X_test, y_test, 
                                         epochs=15, batch_size=128):  # REDUCED EPOCHS
        #Training a CNN with early stopping for moderate accuracy
        print(f"\nTraining classifier to generate soft labels...")
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        #Simpler model for more uncertainty
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(32 * 8 * 8, 64)  
                self.fc2 = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.3) 
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 8 * 8)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        model = SimpleCNN().to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, "
                      f"Acc: {100*correct/total:.2f}%")
        
        # Get soft labels
        model.eval()
        test_soft_labels = []
        test_hard_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs).squeeze()
                probs = torch.sigmoid(outputs)
                test_soft_labels.extend(probs.cpu().numpy())
                test_hard_labels.extend(labels.numpy())
        
        test_soft_labels = np.array(test_soft_labels)
        test_hard_labels = np.array(test_hard_labels)
        
        test_acc = np.mean((test_soft_labels > 0.5) == test_hard_labels)
        print(f"\nTest Accuracy: {100*test_acc:.2f}%")
        print(f"Soft labels range: [{test_soft_labels.min():.3f}, {test_soft_labels.max():.3f}]")
        
        return test_soft_labels, test_hard_labels
    

   #EstimatE Bayes error from clean soft labels Eq. 4
    def estimate_bayes_error_clean(self, soft_labels):
        return np.mean(np.minimum(soft_labels, 1 - soft_labels))
    

    #Naive estimator with noisy labels Eq. 5-BIased
    def estimate_bayes_error_noisy_naive(self, noisy_soft_labels):
        return np.mean(np.minimum(noisy_soft_labels, 1 - noisy_soft_labels))
    
    #Corrected estimator with noisy labels Eq. 6-Unbiased
    def estimate_bayes_error_noisy_corrected(self, noisy_soft_labels, sign_labels):
        positive_mask = sign_labels == 1
        negative_mask = sign_labels == 0
        
        positive_contribution = np.sum(1 - noisy_soft_labels[positive_mask])
        negative_contribution = np.sum(noisy_soft_labels[negative_mask])
        
        return (positive_contribution + negative_contribution) / len(noisy_soft_labels)
    
    #Adding truncated gaussian noise to soft labels
    def add_truncated_gaussian_noise(self, clean_labels, std=0.4):
        noisy_labels = np.zeros_like(clean_labels)
        
        for i, mu in enumerate(clean_labels):
            lower_bound = -mu
            upper_bound = 1 - mu
            
            a, b = lower_bound / std, upper_bound / std
            noise = truncnorm.rvs(a, b, loc=0, scale=std)
            
            noisy_labels[i] = np.clip(mu + noise, 0, 1)
        
        return noisy_labels


class CIFAR10NoiseRobustnessExperiment:
    
    def __init__(self, random_state=42):
        self.estimator = CIFAR10BayesErrorEstimator(random_state)
        self.results = []
        self.soft_labels = None
        self.hard_labels = None
    # Setup binary classification task
    def setup_dataset(self, setup_name='animals_vs_vehicles'):
       
        setups = {
            'animals_vs_vehicles': {
                'positive': [2, 3, 4, 5, 6, 7],
                'negative': [0, 1, 8, 9],
                'description': 'Animals vs Vehicles'
            }
        }
        
        config = setups[setup_name]
        print(f"Dataset Setup: {config['description']}")
        
        return self.estimator.load_cifar10_binary(
            config['positive'], config['negative']
        )
    #Run the main experiment
    def run_experiment(self, setup_name='animals_vs_vehicles', 
                      n_trials=10, 
                      noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      train_epochs=15): 
     
        print("CIFAR-10 Noise robustness experiment")
        
        X_train, y_train, X_test, y_test = self.setup_dataset(setup_name)
        
        print("Training classifier for soft label")
        
        clean_soft_labels, hard_labels = self.estimator.train_classifier_for_soft_labels(
            X_train, y_train, X_test, y_test, epochs=train_epochs
        )
        
        self.soft_labels = clean_soft_labels
        self.hard_labels = hard_labels
        
        baseline_bayes = self.estimator.estimate_bayes_error_clean(clean_soft_labels)
        print(f"\nBaseline Bayes Error Estimate (clean): {baseline_bayes:.4f}")
        
        print("Testing robustness of noise")
    
        
        results = []
        
        for noise_std in noise_levels:
            print(f"\nNoise level (std): {noise_std:.2f}")
            
            trial_results = {'clean': [], 'naive': [], 'corrected': []}
            
            for trial in range(n_trials):
                sign_labels = (clean_soft_labels >= 0.5).astype(int)
                
                if noise_std > 0:
                    noisy_soft_labels = self.estimator.add_truncated_gaussian_noise(
                        clean_soft_labels, std=noise_std
                    )
                else:
                    noisy_soft_labels = clean_soft_labels.copy()
                
                beta_clean = self.estimator.estimate_bayes_error_clean(clean_soft_labels)
                beta_naive = self.estimator.estimate_bayes_error_noisy_naive(noisy_soft_labels)
                beta_corrected = self.estimator.estimate_bayes_error_noisy_corrected(
                    noisy_soft_labels, sign_labels
                )
                
                trial_results['clean'].append(beta_clean)
                trial_results['naive'].append(beta_naive)
                trial_results['corrected'].append(beta_corrected)
            
            result_entry = {
                'noise_std': noise_std,
                'baseline_bayes': baseline_bayes,
                'clean_mean': np.mean(trial_results['clean']),
                'clean_std': np.std(trial_results['clean']),
                'naive_mean': np.mean(trial_results['naive']),
                'naive_std': np.std(trial_results['naive']),
                'corrected_mean': np.mean(trial_results['corrected']),
                'corrected_std': np.std(trial_results['corrected']),
                'naive_bias': np.mean(trial_results['naive']) - baseline_bayes,
                'corrected_bias': np.mean(trial_results['corrected']) - baseline_bayes
            }
            
            results.append(result_entry)
            
            print(f"  Baseline:         {result_entry['baseline_bayes']:.4f}")
            print(f"  Clean estimator:  {result_entry['clean_mean']:.4f} ± {result_entry['clean_std']:.4f}")
            print(f"  Naive estimator:  {result_entry['naive_mean']:.4f} ± {result_entry['naive_std']:.4f} (bias: {result_entry['naive_bias']:+.4f})")
            print(f"  Corrected estim:  {result_entry['corrected_mean']:.4f} ± {result_entry['corrected_std']:.4f} (bias: {result_entry['corrected_bias']:+.4f})")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    #Compute bias reduction percentage
    def compute_bias_reduction(self):
        
        if len(self.results) == 0:
            return None
        
        high_noise = self.results[self.results['noise_std'] >= 0.3]
        
        naive_bias_avg = np.abs(high_noise['naive_bias']).mean()
        corrected_bias_avg = np.abs(high_noise['corrected_bias']).mean()
        
        if naive_bias_avg > 0:
            reduction_pct = ((naive_bias_avg - corrected_bias_avg) / naive_bias_avg) * 100
        else:
            reduction_pct = 0
        
        return reduction_pct, naive_bias_avg, corrected_bias_avg
    #Create visualization
    def plot_results(self, save_path='results/cifar10_noise_robustness.png'):
      
        if len(self.results) == 0:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        noise_levels = self.results['noise_std']
        
        # Plot 1 - Bayes Error vs Noise
        ax1 = axes[0]
        ax1.plot(noise_levels, self.results['baseline_bayes'], 
                'k--', linewidth=2, label='Baseline (Clean)', marker='o')
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
        ax1.set_title('CIFAR-10: Bayes Error Estimation under Label Noise', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2 - Bias Comparison
        ax2 = axes[1]
        x_pos = np.arange(len(noise_levels))
        width = 0.35
        
        ax2.bar(x_pos - width/2, np.abs(self.results['naive_bias']), 
               width, label='Naive Estimator', color='red', alpha=0.7)
        ax2.bar(x_pos + width/2, np.abs(self.results['corrected_bias']), 
               width, label='Corrected Estimator', color='green', alpha=0.7)
        
        ax2.set_xlabel('Noise Level (std)', fontsize=12)
        ax2.set_ylabel('Absolute Bias', fontsize=12)
        ax2.set_title('CIFAR-10: Bias Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{x:.1f}' for x in noise_levels])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.show()
    
    #Generate table for report
    def generate_report_table(self):
        
        if len(self.results) == 0:
            return None
        
        table_data = self.results[['noise_std', 'baseline_bayes', 
                                   'naive_mean', 'naive_bias',
                                   'corrected_mean', 'corrected_bias']].copy()
        
        table_data.columns = ['Noise σ', 'Baseline β', 'Naive β̂', 'Naive Bias',
                             'Corrected β̂', 'Corrected Bias']
        
        for col in table_data.columns:
            if col != 'Noise σ':
                table_data[col] = table_data[col].round(4)
        
        return table_data


def main():

    print("CIFAR-10 Bayes error estimation")
    print("Testing Hypothesis: Corrected Estimator is Robust to Noise")
    
    experiment = CIFAR10NoiseRobustnessExperiment(random_state=42)
    
    start_time = time.time()
    results_df = experiment.run_experiment(
        setup_name='animals_vs_vehicles',
        n_trials=10,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        train_epochs=15  
    )
    end_time = time.time()
    
    
    reduction_pct, naive_bias, corrected_bias = experiment.compute_bias_reduction()
    
    print("Hypothesis Results")
    print(f"Average Naive Bias--high noise:      {naive_bias:.6f}")
    print(f"Average Corrected Bias--high noise:  {corrected_bias:.6f}")
    print(f"Bias Reduction:                       {reduction_pct:.2f}%")
    print()
    
    if reduction_pct >= 50:
        print("Hypothesis Confirmed: Bias reduction≥50%")
        print("The corrected estimator is significantly more robust to noise.")
    else:
        print("Hypothesis not fully confirmed: Bias reduction<50%")
        print(f"Got {reduction_pct:.1f}% reduction. May need different task difficulty.")
    
    experiment.plot_results(save_path='results/cifar10_noise_robustness.png')
    
    print("\nReport Table:")
    print(experiment.generate_report_table().to_markdown(index=False))
    
    results_df.to_csv('results/cifar10_experiment_results.csv', index=False)
    print("\nResults saved to: results/cifar10_experiment_results.csv")


if __name__ == "__main__":
    main()