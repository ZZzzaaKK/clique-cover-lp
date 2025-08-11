#!/usr/bin/env python3
"""
WP1c: Compare Chalupa heuristics to ILP solutions for existing results
Works with existing result files but provides comprehensive analysis like wp1c_evaluation.py

Author: Analysis for Clique Cover Project
Date: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WP1Analyzer:
    " Analyzer tool that combines existing result parsing with comprehensive analysis"""
    
    def __init__(self, results_dir="results", output_dir="evaluation_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.df = None  # Main dataframe for analysis
        
    def parse_result_file(self, filepath):
        """Parse a result file and extract metrics"""
        results = []
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract individual test results
        test_blocks = content.split('-' * 30)
        
        for block in test_blocks:
            if 'File:' in block:
                result = {}
                
                # Extract metrics using regex
                patterns = {
                    'file': r'File:\s+(.+)',
                    'predicted': r'Predicted:\s+(\d+)',
                    'actual': r'Actual:\s+(\d+)',
                    'deviation': r'Deviation:\s+(-?\d+)',
                    'correct': r'Correct:\s+(True|False)',
                    'time_taken': r'Time taken:\s+([\d.]+)'
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, block)
                    if match:
                        if key in ['predicted', 'actual', 'deviation']:
                            result[key] = int(match.group(1))
                        elif key == 'correct':
                            result[key] = match.group(1) == 'True'
                        elif key == 'time_taken':
                            result[key] = float(match.group(1))
                        else:
                            result[key] = match.group(1)
                
                # Extract additional parameters from filename
                if 'file' in result:
                    params = self.extract_comprehensive_params(result['file'])
                    result.update(params)
                
                if result:
                    results.append(result)
        
        return results
    
    def extract_comprehensive_params(self, filename):
        """Extract comprehensive parameters from filename"""
        params = {}
        
        # Extract numbers after specific patterns
        patterns = {
            'num_cliques': r'n(\d+)',
            'clique_size': r's(\d+)',
            'perturbation': r'r(\d+)|perturbation(\d+)|p(\d{3})',
            'min_size': r'min(\d+)',
            'max_size': r'max(\d+)',
            'nodes': r'nodes(\d+)|vertices(\d+)',
            'edges': r'edges(\d+)',
            'density': r'density(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                # Take the first non-None group
                value = next((g for g in match.groups() if g is not None), None)
                if value:
                    if key == 'perturbation':
                        # Convert perturbation to 0-1 scale
                        params[key] = int(value) / 100 if int(value) > 1 else float(value)
                    elif key == 'density':
                        params[key] = int(value) / 100
                    else:
                        params[key] = int(value)
        
        # Estimate problem size if not directly available
        if 'nodes' not in params and 'num_cliques' in params and 'clique_size' in params:
            params['n_nodes'] = params['num_cliques'] * params['clique_size']
        elif 'nodes' in params:
            params['n_nodes'] = params['nodes']
        
        # Estimate edges if not available
        if 'edges' not in params and 'num_cliques' in params and 'clique_size' in params:
            # Approximate for clique-based graphs
            clique_edges = params['num_cliques'] * (params['clique_size'] * (params['clique_size'] - 1)) // 2
            if 'perturbation' in params:
                # Adjust for perturbation
                params['n_edges'] = int(clique_edges * (1 - params['perturbation'] * 0.5))
            else:
                params['n_edges'] = clique_edges
        elif 'edges' in params:
            params['n_edges'] = params['edges']
        
        # Calculate density if we have nodes and edges
        if 'n_nodes' in params and 'n_edges' in params and params['n_nodes'] > 1:
            max_edges = params['n_nodes'] * (params['n_nodes'] - 1) // 2
            params['density'] = params['n_edges'] / max_edges if max_edges > 0 else 0
        
        return params
    
    def load_all_results(self):
        """Load and merge results for all algorithms into a unified dataframe"""
        all_results = []
        
        for result_file in self.results_dir.glob("*.txt"):
            filename = result_file.name.lower()
            
            # Determine method from filename
            if 'chalupa' in filename:
                method = 'chalupa'
            elif 'reduced_ilp' in filename or 'reduced-ilp' in filename:
                method = 'reduced_ilp'
            elif 'ilp' in filename and 'reduced' not in filename:
                method = 'ilp'
            else:
                continue  # Skip unknown files
            
            # Parse results
            results = self.parse_result_file(result_file)
            
            # Add method identifier
            for r in results:
                r['method'] = method
                all_results.append(r)
        
        if not all_results:
            print("Warning: No results found in", self.results_dir)
            return pd.DataFrame()
        
        # Create main dataframe
        self.df = pd.DataFrame(all_results)
        
        # Calculate additional metrics
        self.calculate_derived_metrics()
        
        print(f"Loaded {len(self.df)} result entries from {len(set(self.df['file'].values))} unique test cases")
        print(f"Methods found: {self.df['method'].unique()}")
        
        return self.df
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics for analysis"""
        if self.df.empty:
            return
        
        # Add quality metrics for each method
        if 'predicted' in self.df.columns and 'actual' in self.df.columns:
            # Quality ratio (predicted/actual) - for clique cover, lower actual is better
            self.df['quality_ratio'] = self.df['predicted'] / self.df['actual']
            self.df['quality_ratio'] = self.df['quality_ratio'].replace([np.inf, -np.inf], np.nan)
            
            # Absolute gap
            self.df['absolute_gap'] = self.df['predicted'] - self.df['actual']
        
        # Add size categories
        if 'n_nodes' in self.df.columns:
            self.df['size_category'] = pd.cut(
                self.df['n_nodes'],
                bins=[0, 20, 50, 100, 200, float('inf')],
                labels=['Tiny (<20)', 'Small (20-50)', 'Medium (50-100)', 
                       'Large (100-200)', 'XLarge (>200)']
            )
    
    def create_comparison_dataframe(self):
        """Create a comparison dataframe with one row per test instance"""
        if self.df.empty:
            return pd.DataFrame()
        
        # Pivot to get methods as columns
        comparison_data = []
        
        for test_file in self.df['file'].unique():
            row = {'file': test_file}
            
            # Get data for this test file
            test_data = self.df[self.df['file'] == test_file]
            
            # Extract parameters (should be same for all methods)
            param_cols = ['n_nodes', 'n_edges', 'density', 'perturbation', 
                         'num_cliques', 'clique_size']
            for col in param_cols:
                if col in test_data.columns:
                    value = test_data[col].iloc[0] if not test_data[col].isna().all() else None
                    row[col] = value
            
            # Extract results for each method
            for method in ['chalupa', 'ilp', 'reduced_ilp']:
                method_data = test_data[test_data['method'] == method]
                
                if not method_data.empty:
                    row[f'{method}_theta'] = method_data['predicted'].iloc[0]
                    row[f'{method}_time'] = method_data['time_taken'].iloc[0] if 'time_taken' in method_data.columns else None
                    row[f'{method}_correct'] = method_data['correct'].iloc[0] if 'correct' in method_data.columns else None
                    
                    if 'actual' in method_data.columns:
                        actual = method_data['actual'].iloc[0]
                        predicted = method_data['predicted'].iloc[0]
                        row[f'{method}_quality_ratio'] = predicted / actual if actual > 0 else None
                        row[f'{method}_absolute_gap'] = predicted - actual
            
            # Store ground truth
            if 'actual' in test_data.columns:
                row['ground_truth'] = test_data['actual'].iloc[0]
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def create_runtime_plots(self):
        """Create comprehensive runtime comparison plots"""
        comparison_df = self.create_comparison_dataframe()
        
        if comparison_df.empty:
            print("No data available for runtime plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Runtime vs Problem Size
        ax = axes[0, 0]
        if 'n_nodes' in comparison_df.columns:
            for method in ['chalupa', 'ilp', 'reduced_ilp']:
                time_col = f'{method}_time'
                if time_col in comparison_df.columns:
                    valid_data = comparison_df.dropna(subset=['n_nodes', time_col])
                    if not valid_data.empty:
                        ax.scatter(valid_data['n_nodes'], valid_data[time_col], 
                                 alpha=0.6, label=method.replace('_', ' ').title(), s=50)
            
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('Runtime vs Problem Size')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 2. Runtime vs Density
        ax = axes[0, 1]
        if 'density' in comparison_df.columns:
            for method in ['chalupa', 'ilp', 'reduced_ilp']:
                time_col = f'{method}_time'
                if time_col in comparison_df.columns:
                    valid_data = comparison_df.dropna(subset=['density', time_col])
                    if not valid_data.empty:
                        ax.scatter(valid_data['density'], valid_data[time_col], 
                                 alpha=0.6, label=method.replace('_', ' ').title(), s=50)
            
            ax.set_xlabel('Graph Density')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('Runtime vs Graph Density')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 3. Speedup Distribution
        ax = axes[1, 0]
        speedups = []
        labels = []
        
        if 'chalupa_time' in comparison_df.columns and 'ilp_time' in comparison_df.columns:
            valid = comparison_df.dropna(subset=['chalupa_time', 'ilp_time'])
            if not valid.empty:
                speedup = valid['ilp_time'] / valid['chalupa_time']
                speedups.append(speedup)
                labels.append('Chalupa vs ILP')
        
        if 'reduced_ilp_time' in comparison_df.columns and 'ilp_time' in comparison_df.columns:
            valid = comparison_df.dropna(subset=['reduced_ilp_time', 'ilp_time'])
            if not valid.empty:
                speedup = valid['ilp_time'] / valid['reduced_ilp_time']
                speedups.append(speedup)
                labels.append('Reduced ILP vs ILP')
        
        if speedups:
            ax.hist(speedups, bins=30, alpha=0.7, label=labels)
            ax.axvline(x=1, color='red', linestyle='--', label='No speedup')
            ax.set_xlabel('Speedup Factor')
            ax.set_ylabel('Frequency')
            ax.set_title('Speedup Distribution')
            ax.legend()
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # 4. Runtime Comparison Box Plot
        ax = axes[1, 1]
        runtime_data = []
        labels = []
        
        for method in ['chalupa', 'ilp', 'reduced_ilp']:
            time_col = f'{method}_time'
            if time_col in comparison_df.columns:
                times = comparison_df[time_col].dropna()
                if not times.empty:
                    runtime_data.append(times)
                    labels.append(method.replace('_', ' ').title())
        
        if runtime_data:
            bp = ax.boxplot(runtime_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_ylabel('Runtime (seconds)')
            ax.set_title('Runtime Distribution Comparison')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Runtime Analysis: Comprehensive Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / f"runtime_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved runtime plots to {output_path}")
        plt.show()
    
    def create_quality_plots(self):
        """Create comprehensive solution quality comparison plots"""
        comparison_df = self.create_comparison_dataframe()
        
        if comparison_df.empty:
            print("No data available for quality plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Quality Ratio Distribution
        ax = axes[0, 0]
        for method in ['chalupa', 'reduced_ilp']:
            ratio_col = f'{method}_quality_ratio'
            if ratio_col in comparison_df.columns:
                ratios = comparison_df[ratio_col].dropna()
                if not ratios.empty:
                    ax.hist(ratios, bins=30, alpha=0.5, label=method.replace('_', ' ').title())
        
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Optimal')
        ax.set_xlabel('Quality Ratio (Method θ / Optimal θ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Solution Quality Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Quality vs Problem Size
        ax = axes[0, 1]
        if 'n_nodes' in comparison_df.columns:
            for method in ['chalupa', 'reduced_ilp']:
                ratio_col = f'{method}_quality_ratio'
                if ratio_col in comparison_df.columns:
                    valid = comparison_df.dropna(subset=['n_nodes', ratio_col])
                    if not valid.empty:
                        scatter = ax.scatter(valid['n_nodes'], valid[ratio_col],
                                           alpha=0.6, label=method.replace('_', ' ').title(), s=50)
            
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel('Quality Ratio')
            ax.set_title('Solution Quality vs Problem Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Absolute Gap Distribution
        ax = axes[1, 0]
        for method in ['chalupa', 'reduced_ilp']:
            gap_col = f'{method}_absolute_gap'
            if gap_col in comparison_df.columns:
                gaps = comparison_df[gap_col].dropna()
                if not gaps.empty:
                    ax.hist(gaps, bins=30, alpha=0.5, label=method.replace('_', ' ').title())
        
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Optimal')
        ax.set_xlabel('Absolute Gap (Method θ - Optimal θ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Absolute Gap Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Success Rate Analysis
        ax = axes[1, 1]
        thresholds = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
        
        for method in ['chalupa', 'reduced_ilp']:
            ratio_col = f'{method}_quality_ratio'
            if ratio_col in comparison_df.columns:
                ratios = comparison_df[ratio_col].dropna()
                if not ratios.empty:
                    success_rates = []
                    for threshold in thresholds:
                        success_rate = (ratios <= threshold).mean() * 100
                        success_rates.append(success_rate)
                    
                    ax.plot(thresholds, success_rates, marker='o', linewidth=2, 
                           markersize=8, label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Quality Threshold')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate at Different Quality Thresholds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Solution Quality Analysis: Comprehensive Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / f"quality_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved quality plots to {output_path}")
        plt.show()
    
    def create_perturbation_analysis(self):
        """Analyze effect of perturbation strength on algorithm performance"""
        comparison_df = self.create_comparison_dataframe()
        
        if comparison_df.empty or 'perturbation' not in comparison_df.columns:
            print("No perturbation data available for analysis")
            return
        
        # Filter data with perturbation values
        pert_df = comparison_df.dropna(subset=['perturbation'])
        
        if pert_df.empty:
            print("No valid perturbation data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Quality vs Perturbation
        ax = axes[0, 0]
        for method in ['chalupa', 'reduced_ilp']:
            ratio_col = f'{method}_quality_ratio'
            if ratio_col in pert_df.columns:
                valid = pert_df.dropna(subset=[ratio_col])
                if not valid.empty:
                    # Group by perturbation level and calculate mean
                    grouped = valid.groupby('perturbation')[ratio_col].mean()
                    ax.plot(grouped.index * 100, grouped.values, 
                           marker='o', linewidth=2, markersize=8, 
                           label=method.replace('_', ' ').title())
        
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Optimal')
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('Mean Quality Ratio')
        ax.set_title('Solution Quality vs Perturbation Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Runtime vs Perturbation
        ax = axes[0, 1]
        for method in ['chalupa', 'ilp', 'reduced_ilp']:
            time_col = f'{method}_time'
            if time_col in pert_df.columns:
                valid = pert_df.dropna(subset=[time_col])
                if not valid.empty:
                    grouped = valid.groupby('perturbation')[time_col].mean()
                    ax.plot(grouped.index * 100, grouped.values,
                           marker='o', linewidth=2, markersize=8,
                           label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('Mean Runtime (seconds)')
        ax.set_title('Runtime vs Perturbation Strength')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 3. Absolute θ values vs Perturbation
        ax = axes[1, 0]
        for method in ['chalupa', 'ilp', 'reduced_ilp']:
            theta_col = f'{method}_theta'
            if theta_col in pert_df.columns:
                valid = pert_df.dropna(subset=[theta_col])
                if not valid.empty:
                    grouped = valid.groupby('perturbation')[theta_col].mean()
                    ax.plot(grouped.index * 100, grouped.values,
                           marker='o', linewidth=2, markersize=8,
                           label=method.replace('_', ' ').title())
        
        if 'ground_truth' in pert_df.columns:
            grouped = pert_df.groupby('perturbation')['ground_truth'].mean()
            ax.plot(grouped.index * 100, grouped.values,
                   marker='s', linewidth=2, markersize=8,
                   label='Ground Truth', color='green')
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('Clique Cover Number θ(G)')
        ax.set_title('Clique Cover Number vs Perturbation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Success Rate vs Perturbation
        ax = axes[1, 1]
        for method in ['chalupa', 'reduced_ilp']:
            correct_col = f'{method}_correct'
            if correct_col in pert_df.columns:
                valid = pert_df.dropna(subset=[correct_col])
                if not valid.empty:
                    grouped = valid.groupby('perturbation')[correct_col].mean() * 100
                    ax.plot(grouped.index * 100, grouped.values,
                           marker='o', linewidth=2, markersize=8,
                           label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Optimal Solution Rate vs Perturbation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Perturbation Strength Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / f"perturbation_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved perturbation plots to {output_path}")
        plt.show()
    
    def perform_statistical_analysis(self):
        """Perform statistical tests and correlation analysis"""
        comparison_df = self.create_comparison_dataframe()
        
        if comparison_df.empty:
            print("No data available for statistical analysis")
            return {}
        
        stats_results = {}
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Test if Chalupa quality ratio is significantly different from 1.0
        if 'chalupa_quality_ratio' in comparison_df.columns:
            ratios = comparison_df['chalupa_quality_ratio'].dropna()
            if len(ratios) > 1:
                t_stat, p_value = stats.ttest_1samp(ratios, 1.0)
                stats_results['chalupa_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_ratio': ratios.mean(),
                    'std_ratio': ratios.std()
                }
                
                print(f"\nOne-sample t-test for Chalupa (H0: quality_ratio = 1.0):")
                print(f"  Mean ratio: {ratios.mean():.4f}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.6f}")
                
                if p_value < 0.05:
                    print("  ✗ Chalupa is statistically different from optimal (p < 0.05)")
                else:
                    print("  ✓ No significant difference from optimal (p ≥ 0.05)")
        
        # Paired t-test for runtimes
        if 'chalupa_time' in comparison_df.columns and 'ilp_time' in comparison_df.columns:
            valid = comparison_df.dropna(subset=['chalupa_time', 'ilp_time'])
            if len(valid) > 1:
                t_stat, p_value = stats.ttest_rel(valid['chalupa_time'], valid['ilp_time'])
                stats_results['runtime_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_speedup': (valid['ilp_time'] / valid['chalupa_time']).mean()
                }
                
                print(f"\nPaired t-test for runtime difference:")
                print(f"  Mean speedup: {(valid['ilp_time'] / valid['chalupa_time']).mean():.1f}x")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.6f}")
        
        # Correlation analysis
        if len(comparison_df) > 2:
            print("\n" + "-"*40)
            print("CORRELATION ANALYSIS")
            print("-"*40)
            
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            key_cols = [col for col in ['n_nodes', 'n_edges', 'density', 'perturbation', 
                                       'chalupa_quality_ratio', 'chalupa_time', 'ilp_time'] 
                       if col in numeric_cols]
            
            if len(key_cols) > 1:
                corr_matrix = comparison_df[key_cols].corr()
                stats_results['correlations'] = corr_matrix.to_dict()
                
                # Print key correlations
                if 'chalupa_quality_ratio' in corr_matrix.columns:
                    print("\nCorrelations with Chalupa quality ratio:")
                    for col in ['n_nodes', 'n_edges', 'density', 'perturbation']:
                        if col in corr_matrix.columns:
                            corr = corr_matrix.loc['chalupa_quality_ratio', col]
                            print(f"  {col}: {corr:.3f}")
        
        return stats_results
    
    def generate_comprehensive_report(self, stats_results=None):
        """Generate comprehensive reports in multiple formats"""
        comparison_df = self.create_comparison_dataframe()
        
        if comparison_df.empty:
            print("No data available for report generation")
            return
        
        # Generate text report
        self.generate_text_report(comparison_df, stats_results)
        
        # Generate markdown report
        self.generate_markdown_report(comparison_df, stats_results)
        
        # Generate LaTeX table
        self.generate_latex_table(comparison_df)
        
        # Save processed data
        self.save_processed_data(comparison_df)
    
    def generate_text_report(self, df, stats_results=None):
        """Generate comprehensive text summary report"""
        report_path = self.output_dir / f"summary_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WP1.c EVALUATION SUMMARY REPORT (FROM EXISTING RESULTS)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total test instances: {len(df)}\n")
            
            # Method availability
            methods_available = []
            for method in ['chalupa', 'ilp', 'reduced_ilp']:
                if f'{method}_theta' in df.columns:
                    count = df[f'{method}_theta'].notna().sum()
                    if count > 0:
                        methods_available.append(f"{method}: {count}")
            f.write(f"Methods evaluated: {', '.join(methods_available)}\n\n")
            
            # Runtime Performance
            f.write("RUNTIME PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            for method in ['chalupa', 'ilp', 'reduced_ilp']:
                time_col = f'{method}_time'
                if time_col in df.columns:
                    times = df[time_col].dropna()
                    if not times.empty:
                        f.write(f"\n{method.replace('_', ' ').title()}:\n")
                        f.write(f"  Mean: {times.mean():.3f}s\n")
                        f.write(f"  Median: {times.median():.3f}s\n")
                        f.write(f"  Min: {times.min():.3f}s\n")
                        f.write(f"  Max: {times.max():.3f}s\n")
            
            # Speedup analysis
            if 'chalupa_time' in df.columns and 'ilp_time' in df.columns:
                valid = df.dropna(subset=['chalupa_time', 'ilp_time'])
                if not valid.empty:
                    speedups = valid['ilp_time'] / valid['chalupa_time']
                    f.write(f"\nSpeedup (ILP/Chalupa):\n")
                    f.write(f"  Mean: {speedups.mean():.1f}x\n")
                    f.write(f"  Median: {speedups.median():.1f}x\n")
            
            # Solution Quality
            f.write("\nSOLUTION QUALITY\n")
            f.write("-"*40 + "\n")
            
            for method in ['chalupa', 'reduced_ilp']:
                ratio_col = f'{method}_quality_ratio'
                if ratio_col in df.columns:
                    ratios = df[ratio_col].dropna()
                    if not ratios.empty:
                        f.write(f"\n{method.replace('_', ' ').title()} Quality:\n")
                        f.write(f"  Mean ratio: {ratios.mean():.4f}\n")
                        f.write(f"  Median ratio: {ratios.median():.4f}\n")
                        f.write(f"  Std deviation: {ratios.std():.4f}\n")
                        
                        # Success rates
                        optimal = (ratios == 1.0).mean() * 100
                        within_5 = (ratios <= 1.05).mean() * 100
                        within_10 = (ratios <= 1.10).mean() * 100
                        within_20 = (ratios <= 1.20).mean() * 100
                        
                        f.write(f"  Optimal solutions: {optimal:.1f}%\n")
                        f.write(f"  Within 5% of optimal: {within_5:.1f}%\n")
                        f.write(f"  Within 10% of optimal: {within_10:.1f}%\n")
                        f.write(f"  Within 20% of optimal: {within_20:.1f}%\n")
            
            # Statistical results
            if stats_results:
                f.write("\nSTATISTICAL TESTS\n")
                f.write("-"*40 + "\n")
                
                if 'chalupa_ttest' in stats_results:
                    res = stats_results['chalupa_ttest']
                    f.write(f"\nChalupa vs Optimal (t-test):\n")
                    f.write(f"  t-statistic: {res['t_statistic']:.4f}\n")
                    f.write(f"  p-value: {res['p_value']:.6f}\n")
                    f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Saved text report to {report_path}")
    
    def generate_markdown_report(self, df, stats_results=None):
        """Generate markdown report with recommendations"""
        report_path = self.output_dir / f"wp1c_analysis_report_{self.timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# WP1c Analysis Report: Comprehensive Algorithm Comparison\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            # Determine best method based on data
            if 'chalupa_quality_ratio' in df.columns:
                chalupa_ratios = df['chalupa_quality_ratio'].dropna()
                if not chalupa_ratios.empty:
                    accuracy = (chalupa_ratios == 1.0).mean() * 100
                    mean_ratio = chalupa_ratios.mean()
                    
                    if accuracy >= 95 or mean_ratio <= 1.02:
                        recommendation = "**Excellence Level**: Chalupa achieves near-optimal solutions consistently"
                    elif accuracy >= 80 or mean_ratio <= 1.05:
                        recommendation = "**Production Ready**: Chalupa provides excellent trade-off between speed and quality"
                    elif accuracy >= 60 or mean_ratio <= 1.10:
                        recommendation = "**Good Performance**: Chalupa suitable for approximate solutions"
                    elif accuracy >= 40 or mean_ratio <= 1.15:
                        recommendation = "**Moderate Performance**: Consider using Chalupa for initial bounds only"
                    else:
                        recommendation = "**Limited Performance**: Prefer exact methods for these instances"
                    
                    f.write(f"### Key Finding: {recommendation}\n\n")
            
            # Summary Statistics Table
            f.write("## Summary Statistics\n\n")
            f.write("| Metric | Chalupa | ILP | Reduced ILP |\n")
            f.write("|--------|---------|-----|-------------|\n")
            
            metrics = {
                'Instances': lambda m: df[f'{m}_theta'].notna().sum() if f'{m}_theta' in df.columns else 0,
                'Avg Time (s)': lambda m: f"{df[f'{m}_time'].mean():.3f}" if f'{m}_time' in df.columns else "N/A",
                'Median Time (s)': lambda m: f"{df[f'{m}_time'].median():.3f}" if f'{m}_time' in df.columns else "N/A",
                'Optimal (%)': lambda m: f"{(df[f'{m}_quality_ratio'] == 1.0).mean()*100:.1f}" if f'{m}_quality_ratio' in df.columns else "N/A",
                'Within 10% (%)': lambda m: f"{(df[f'{m}_quality_ratio'] <= 1.10).mean()*100:.1f}" if f'{m}_quality_ratio' in df.columns else "N/A",
            }
            
            for metric_name, metric_func in metrics.items():
                row = f"| {metric_name} "
                for method in ['chalupa', 'ilp', 'reduced_ilp']:
                    value = metric_func(method)
                    row += f"| {value} "
                f.write(row + "|\n")
            
            # Performance Analysis
            f.write("\n## Performance Analysis\n\n")
            
            # Speedup analysis
            if 'chalupa_time' in df.columns and 'ilp_time' in df.columns:
                valid = df.dropna(subset=['chalupa_time', 'ilp_time'])
                if not valid.empty:
                    speedup = (valid['ilp_time'] / valid['chalupa_time']).mean()
                    f.write(f"### Speed Advantage\n")
                    f.write(f"- Chalupa is **{speedup:.1f}x faster** than ILP on average\n")
                    f.write(f"- Maximum speedup observed: **{(valid['ilp_time'] / valid['chalupa_time']).max():.1f}x**\n\n")
            
            # Quality analysis
            if 'chalupa_quality_ratio' in df.columns:
                ratios = df['chalupa_quality_ratio'].dropna()
                if not ratios.empty:
                    f.write(f"### Solution Quality\n")
                    f.write(f"- Average quality ratio: **{ratios.mean():.3f}**\n")
                    f.write(f"- Standard deviation: **{ratios.std():.3f}**\n")
                    f.write(f"- Worst case ratio: **{ratios.max():.3f}**\n\n")
            
            # Perturbation Impact
            if 'perturbation' in df.columns:
                pert_df = df.dropna(subset=['perturbation'])
                if not pert_df.empty and 'chalupa_quality_ratio' in pert_df.columns:
                    f.write("### Perturbation Impact\n\n")
                    
                    # Group by perturbation levels
                    pert_groups = pert_df.groupby(pd.cut(pert_df['perturbation'], 
                                                          bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]))
                    
                    f.write("| Perturbation Range | Avg Quality Ratio | Success Rate (%) |\n")
                    f.write("|-------------------|-------------------|------------------|\n")
                    
                    for pert_range, group in pert_groups:
                        if len(group) > 0:
                            ratio = group['chalupa_quality_ratio'].mean()
                            success = (group['chalupa_quality_ratio'] <= 1.1).mean() * 100
                            f.write(f"| {pert_range} | {ratio:.3f} | {success:.1f} |\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            # Based on performance analysis
            recommendations = []
            
            if 'chalupa_quality_ratio' in df.columns:
                accuracy = (df['chalupa_quality_ratio'] == 1.0).mean() * 100
                if accuracy >= 80:
                    recommendations.append("✅ **Use Chalupa as primary solver** for large-scale instances where speed is critical")
                elif accuracy >= 50:
                    recommendations.append("⚡ **Use Chalupa for warm-starting** ILP solvers to improve convergence")
                else:
                    recommendations.append("🔧 **Use Chalupa for initial bounds** in branch-and-bound frameworks")
            
            if 'reduced_ilp_time' in df.columns and 'ilp_time' in df.columns:
                valid = df.dropna(subset=['reduced_ilp_time', 'ilp_time'])
                if not valid.empty:
                    reduction_speedup = (valid['ilp_time'] / valid['reduced_ilp_time']).mean()
                    if reduction_speedup > 1.5:
                        recommendations.append(f"✅ **Always apply reductions** before ILP (average {reduction_speedup:.1f}x speedup)")
            
            recommendations.append("📊 **Monitor instance characteristics** to select appropriate algorithm")
            recommendations.append("🔄 **Consider hybrid approaches** combining heuristics with exact methods")
            
            for rec in recommendations:
                f.write(f"- {rec}\n")
            
            # Instance-specific recommendations
            f.write("\n### Instance-Specific Guidelines\n\n")
            
            if 'n_nodes' in df.columns:
                size_groups = df.groupby(pd.cut(df['n_nodes'], bins=[0, 50, 100, 200, float('inf')]))
                
                f.write("| Problem Size | Recommended Approach |\n")
                f.write("|--------------|---------------------|\n")
                
                for size_range, group in size_groups:
                    if len(group) > 0:
                        if 'chalupa_quality_ratio' in group.columns:
                            ratio = group['chalupa_quality_ratio'].mean()
                            if ratio <= 1.05:
                                approach = "Chalupa (excellent quality)"
                            elif ratio <= 1.15:
                                approach = "Chalupa with verification"
                            else:
                                approach = "ILP or Reduced ILP"
                        else:
                            approach = "ILP (no heuristic data)"
                        
                        f.write(f"| {size_range} nodes | {approach} |\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated by WP1c Analyzer - {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        print(f"Saved markdown report to {report_path}")
    
    def generate_latex_table(self, df):
        """Generate LaTeX table for paper inclusion"""
        latex_path = self.output_dir / f"latex_table_{self.timestamp}.tex"
        
        with open(latex_path, 'w') as f:
            f.write("% LaTeX table for WP1c results\n")
            f.write("% Copy this into your paper\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{WP1.c Results: Algorithm Comparison on Clique Cover Problem}\n")
            f.write("\\label{tab:wp1c_results}\n")
            f.write("\\begin{tabular}{lrrrrr}\n")
            f.write("\\toprule\n")
            f.write("Method & Instances & Optimal (\\%) & Within 10\\% (\\%) & Avg Time (s) & Speedup \\\\\n")
            f.write("\\midrule\n")
            
            # Calculate statistics for each method
            for method in ['Chalupa', 'ILP', 'Reduced ILP']:
                method_key = method.lower().replace(' ', '_')
                
                instances = df[f'{method_key}_theta'].notna().sum() if f'{method_key}_theta' in df.columns else 0
                
                if f'{method_key}_quality_ratio' in df.columns:
                    ratios = df[f'{method_key}_quality_ratio'].dropna()
                    optimal = f"{(ratios == 1.0).mean()*100:.1f}" if not ratios.empty else "-"
                    within_10 = f"{(ratios <= 1.10).mean()*100:.1f}" if not ratios.empty else "-"
                else:
                    optimal = "100.0" if method_key == 'ilp' else "-"
                    within_10 = "100.0" if method_key == 'ilp' else "-"
                
                if f'{method_key}_time' in df.columns:
                    times = df[f'{method_key}_time'].dropna()
                    avg_time = f"{times.mean():.3f}" if not times.empty else "-"
                else:
                    avg_time = "-"
                
                # Calculate speedup relative to ILP
                speedup = "-"
                if method_key != 'ilp' and f'{method_key}_time' in df.columns and 'ilp_time' in df.columns:
                    valid = df.dropna(subset=[f'{method_key}_time', 'ilp_time'])
                    if not valid.empty:
                        speedup_val = (valid['ilp_time'] / valid[f'{method_key}_time']).mean()
                        speedup = f"{speedup_val:.1f}$\\times$"
                elif method_key == 'ilp':
                    speedup = "1.0$\\times$"
                
                f.write(f"{method} & {instances} & {optimal} & {within_10} & {avg_time} & {speedup} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"Saved LaTeX table to {latex_path}")
    
    def save_processed_data(self, df):
        """Save processed data in multiple formats"""
        # Save as CSV
        csv_path = self.output_dir / f"processed_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved processed data to {csv_path}")
        
        # Save as JSON
        json_path = self.output_dir / f"processed_results_{self.timestamp}.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"Saved JSON data to {json_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print(" WP1c ANALYSIS - WORKING WITH EXISTING RESULTS")
        print("="*60)
        
        # Load all results
        print("\n1. Loading existing results...")
        self.load_all_results()
        
        if self.df is None or self.df.empty:
            print("❌ No results found. Please run tests first.")
            return
        
        # Create comparison dataframe
        comparison_df = self.create_comparison_dataframe()
        print(f"✓ Created comparison dataframe with {len(comparison_df)} test instances")
        
        # Generate visualizations
        print("\n2. Creating runtime analysis plots...")
        self.create_runtime_plots()
        
        print("\n3. Creating quality analysis plots...")
        self.create_quality_plots()
        
        print("\n4. Creating perturbation analysis plots...")
        self.create_perturbation_analysis()
        
        # Statistical analysis
        print("\n5. Performing statistical analysis...")
        stats_results = self.perform_statistical_analysis()
        
        # Generate reports
        print("\n6. Generating comprehensive reports...")
        self.generate_comprehensive_report(stats_results)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        # Print summary
        self.print_quick_summary(comparison_df)
    
    def print_quick_summary(self, df):
        """Print a quick summary of key findings"""
        print("\n📊 QUICK SUMMARY")
        print("-"*40)
        
        if 'chalupa_quality_ratio' in df.columns:
            ratios = df['chalupa_quality_ratio'].dropna()
            if not ratios.empty:
                print(f"Chalupa Performance:")
                print(f"  • Average quality ratio: {ratios.mean():.3f}")
                print(f"  • Finds optimal: {(ratios == 1.0).mean()*100:.1f}%")
                print(f"  • Within 10% of optimal: {(ratios <= 1.10).mean()*100:.1f}%")
        
        if 'chalupa_time' in df.columns and 'ilp_time' in df.columns:
            valid = df.dropna(subset=['chalupa_time', 'ilp_time'])
            if not valid.empty:
                speedup = (valid['ilp_time'] / valid['chalupa_time']).mean()
                print(f"  • Average speedup: {speedup:.1f}x faster than ILP")
        
        print("\nCheck the evaluation_results/ folder for detailed analysis!")


def main():
    """Main entry point"""
    analyzer = WP1Analyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()