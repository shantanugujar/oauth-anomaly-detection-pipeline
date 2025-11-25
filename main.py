import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import os
import logging
import time
import numpy as np
from preprocessing import UltraFastPreprocessor
from stage1 import BalancedIntelligentSampler
from stage2 import PhysicalValidator
from stage3 import SuperiorBalancedPipeline
from stage4 import FastStage4Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('pipeline_dashboard.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class OAuthAnomalyDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("OAuth Anomaly Detection Pipeline Dashboard")
        self.root.geometry("900x700")  # Adjusted for better layout

        # Variables
        self.input_file = tk.StringVar()
        self.output_files = {}
        self.results = {}
        self.anomaly_details = {}

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # File Selection
        self.file_frame = ttk.Frame(self.root)
        self.file_frame.pack(pady=5, fill="x")
        tk.Label(self.file_frame, text="Select RBA Dataset (rba-dataset.csv):").pack(side=tk.LEFT)
        tk.Button(self.file_frame, text="Browse", command=self.load_file).pack(side=tk.LEFT, padx=5)

        # Progress Frame
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(pady=5, fill="x")
        self.progress_label = tk.Label(self.progress_frame, text="Progress: Not Started")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=850, mode='determinate')
        self.progress_bar.pack()

        # Results Notebook for better organization
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=5, fill="both", expand=True)

        # Stage Results Frame
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Pipeline Results")

        # Anomaly Details Frame
        self.anomaly_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.anomaly_frame, text="Anomaly Details")

        # Start Button
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=5, fill="x")
        tk.Button(self.button_frame, text="Run Pipeline", command=self.run_pipeline).pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.input_file.set(file_path)
            logger.info(f"Selected file: {file_path}")
            tk.Label(self.file_frame, text=f"Selected: {os.path.basename(file_path)}").pack(side=tk.LEFT, padx=5)

    def update_progress(self, stage, total_stages, message):
        progress = (stage / total_stages) * 100
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"Progress: {message} ({progress:.0f}%)")
        self.root.update_idletasks()

    def run_pipeline(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select a dataset file first!")
            return

        total_stages = 5  # Preprocessing, Stage 1, Stage 2, Stage 3, Stage 4
        self.progress_bar['maximum'] = 100

        try:
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            for widget in self.anomaly_frame.winfo_children():
                widget.destroy()

            # Stage 0: Preprocessing (full dataset)
            self.update_progress(1, total_stages, "Preprocessing...")
            preprocessor = UltraFastPreprocessor(self.input_file.get(), 'preprocessed_rba_ultra.csv')
            output_file, stats = preprocessor.process_ultra_fast(chunk_size=200000)
            if output_file:
                self.output_files['preprocessed'] = output_file
                self.results['preprocessing'] = stats
                logger.info(f"Preprocessing completed in {stats['processing_time_minutes']:.1f} minutes")
                self.display_result("Preprocessing", f"Records: {stats['total_records_processed']:,}\nTime: {stats['processing_time_minutes']:.1f} min\nAnomaly Rate: {stats['anomaly_rate']:.1%}")

            # Stage 1: Balanced Intelligent Sampling
            self.update_progress(2, total_stages, "Stage 1: Balanced Sampling...")
            sampler = BalancedIntelligentSampler('preprocessed_rba_ultra.csv', 'stage1_balanced_sampled_rba.csv')
            output_file, metadata = sampler.run_balanced_sampling()
            if output_file:
                self.output_files['stage1'] = output_file
                self.results['stage1'] = metadata
                logger.info(f"Stage 1 completed in {metadata['processing_time_minutes']:.1f} minutes")
                self.display_result("Stage 1", f"Sampled: {metadata['sampled_records']:,}\nReduction: {metadata['reduction_percentage']:.1f}%\nAnomaly Ratio: {metadata['anomaly_ratio_achieved']:.1%}")

            # Stage 2: Physical Validation
            self.update_progress(3, total_stages, "Stage 2: Physical Validation...")
            validator = PhysicalValidator('stage1_balanced_sampled_rba.csv', 'stage2_validated_rba.csv')
            output_file, metadata = validator.run_physical_validation()
            if output_file:
                self.output_files['stage2'] = output_file
                self.results['stage2'] = metadata
                logger.info(f"Stage 2 completed in {metadata['processing_results']['processing_time_minutes']:.1f} minutes")
                self.display_result("Stage 2", f"Validated: {metadata['processing_results']['validated_records']:,}\nRate: {metadata['processing_results']['validation_rate']:.1%}\nTime: {metadata['processing_results']['processing_time_minutes']:.1f} min")

            # Stage 3: Superior Balanced Detection
            self.update_progress(4, total_stages, "Stage 3: Anomaly Detection...")
            pipeline = SuperiorBalancedPipeline('stage2_validated_rba.csv')
            stage3_results = pipeline.run_superior_pipeline()
            if stage3_results:
                self.output_files['stage3'] = 'enhanced_superior_performance_report.json'
                self.results['stage3'] = stage3_results['results']
                if 'results' in stage3_results:
                    table_text = "MODEL PERFORMANCE COMPARISON (WITH ROC-AUC)\n" + \
                               "────────────────────────────────────────────────────────────────────────────────────────\n" + \
                               f"{'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n" + \
                               "────────────────────────────────────────────────────────────────────────────────────────\n"
                    results = stage3_results['results']
                    target_metrics = ['accuracy', 'precision', 'recall', 'f1']
                    sorted_results = sorted(results.items(),
                                            key=lambda x: (sum(1 for m in target_metrics if x[1][m] >= 0.90),
                                                           np.mean([x[1][m] for m in target_metrics])),
                                            reverse=True)
                    for model_name, metrics in sorted_results:
                        acc_pct = f"{metrics['accuracy'] * 100:.1f}%"
                        prec_pct = f"{metrics['precision'] * 100:.1f}%"
                        rec_pct = f"{metrics['recall'] * 100:.1f}%"
                        f1_pct = f"{metrics['f1'] * 100:.1f}%"
                        auc_pct = f"{metrics['roc_auc'] * 100:.1f}%"
                        table_text += f"{model_name:<18} {acc_pct:<10} {prec_pct:<11} {rec_pct:<10} {f1_pct:<10} {auc_pct:<10}\n"
                    self.display_result("Stage 3 Model Performance", table_text)
                if 'data' in stage3_results:
                    top_10_anomalies = stage3_results['data'].nlargest(10, 'anomaly_score')
                    self.anomaly_details.update({
                        f"Anomaly {i+1}": {
                            "User ID": row['User ID'],
                            "IP Address": row['IP Address'],
                            "Country": row['Country'],
                            "Device Type": row['Device Type'],
                            "Anomaly Score": row['anomaly_score'],
                            "Risk Category": row['risk_category'],
                        } for i, row in top_10_anomalies.iterrows()
                    })
                    self.display_anomaly_details()

            # Stage 4: LLM Security Analysis
            self.update_progress(5, total_stages, "Stage 4: LLM Analysis...")
            pipeline = FastStage4Pipeline(top_anomalies=10, use_mock=True)
            results = pipeline.run_complete_pipeline()
            if results and results['success']:
                self.output_files['stage4'] = 'stage4_fast_llm_comprehensive_report.json'
                self.results['stage4'] = results
                self.display_result("Stage 4", f"Time: {results['processing_time_minutes']:.1f} min\nAnomalies: {results['anomalies_processed']:,}")
                if hasattr(pipeline, 'results') and pipeline.results:
                    for i, result in enumerate(pipeline.results[:10], 1):
                        anomaly_data = result['original_data']
                        alert = result['outputs']['alert_summarizer']['fine_tuned']
                        explanation = result['outputs']['risk_explainer']['fine_tuned']
                        playbook = result['outputs']['response_generator']['fine_tuned']
                        self.anomaly_details[f"Anomaly {i}"] = {
                            "User ID": anomaly_data.get('User ID', 'Unknown'),
                            "IP Address": anomaly_data.get('IP Address', 'Unknown'),
                            "Country": anomaly_data.get('Country', 'Unknown'),
                            "Device Type": anomaly_data.get('Device Type', 'Unknown'),
                            "Anomaly Score": anomaly_data.get('anomaly_score', 0),
                            "Risk Category": anomaly_data.get('risk_category', 'Unknown'),
                            "Alert": alert[:200] + "..." if len(alert) > 200 else alert,
                            "Explanation": explanation[:200] + "..." if len(explanation) > 200 else explanation,
                            "Playbook": playbook[:200] + "..." if len(playbook) > 200 else playbook
                        }
                    self.display_anomaly_details()

            self.update_progress(5, total_stages, "Pipeline Completed!")
            messagebox.showinfo("Success", "Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            messagebox.showerror("Error", f"Pipeline failed: {str(e)}")

    def display_result(self, stage, text):
        frame = ttk.Frame(self.results_frame)
        frame.pack(fill="x", pady=2)
        tk.Label(frame, text=f"{stage}:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(frame, text=text, wraplength=800, justify=tk.LEFT).pack(side=tk.LEFT, padx=5)

    def display_anomaly_details(self):
        for widget in self.anomaly_frame.winfo_children():
            widget.destroy()
        if self.anomaly_details:
            canvas = tk.Canvas(self.anomaly_frame)
            scrollbar = ttk.Scrollbar(self.anomaly_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            for anomaly_id, details in self.anomaly_details.items():
                anomaly_frame = ttk.Frame(scrollable_frame)
                anomaly_frame.pack(fill="x", pady=2, padx=5)
                tk.Label(anomaly_frame, text=f"{anomaly_id}:", font=("Arial", 10, "bold")).pack(anchor="w")
                tk.Label(anomaly_frame, text=f"User ID: {details['User ID']}").pack(anchor="w")
                tk.Label(anomaly_frame, text=f"IP Address: {details['IP Address']}").pack(anchor="w")
                tk.Label(anomaly_frame, text=f"Country: {details['Country']}").pack(anchor="w")
                tk.Label(anomaly_frame, text=f"Device Type: {details['Device Type']}").pack(anchor="w")
                tk.Label(anomaly_frame, text=f"Anomaly Score: {details['Anomaly Score']:.3f}").pack(anchor="w")
                tk.Label(anomaly_frame, text=f"Risk Category: {details['Risk Category']}").pack(anchor="w")
                if 'Alert' in details:
                    tk.Label(anomaly_frame, text=f"Alert: {details['Alert']}").pack(anchor="w")
                if 'Explanation' in details:
                    tk.Label(anomaly_frame, text=f"Explanation: {details['Explanation']}").pack(anchor="w")
                if 'Playbook' in details:
                    tk.Label(anomaly_frame, text=f"Playbook: {details['Playbook']}").pack(anchor="w")
                ttk.Separator(scrollable_frame, orient="horizontal").pack(fill="x", pady=2)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

if __name__ == "__main__":
    root = tk.Tk()
    app = OAuthAnomalyDashboard(root)
    root.mainloop()