# OAuth Anomaly Detection Pipeline

## Project Description

This capstone project implements a comprehensive machine learning system for detecting OAuth authentication anomalies in large-scale enterprise environments. The system processes millions of authentication records to identify suspicious patterns that may indicate security threats such as account takeovers, impossible travel sequences, and credential-based attacks.

## System Architecture

The pipeline consists of four sequential processing stages:

**Stage 0 - Data Preprocessing**: Performs high-speed data cleaning and feature engineering on raw authentication logs
**Stage 1 - Intelligent Sampling**: Applies entropy-guided sampling to reduce dataset size from 31 million to 255,000 records
**Stage 2 - Physical Validation**: Validates authentication sequences using geographic and temporal constraints
**Stage 3 - Machine Learning Detection**: Employs ensemble models to classify anomalous authentication patterns
**Stage 4 - Security Analysis**: Uses language models to generate threat assessments and response recommendations

## System Requirements

- Python version 3.8 or higher
- Minimum 8GB RAM (16GB recommended for large datasets)
- 5GB available disk space
- Operating System: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+

## Installation Instructions

### Step 1: Download Project Files
Extract the project archive to your desired directory or clone the repository.

### Step 2: Download RBA Dataset from Kaggle

#### Method 1: Direct Download via Web Browser
1. Visit the Kaggle RBA Dataset page: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
2. Create a Kaggle account if you do not have one
3. Click the "Download" button to download the dataset archive
4. Extract the downloaded archive and locate the file named "rba-dataset.csv"
5. Place the "rba-dataset.csv" file in your project directory

#### Method 2: Using Kaggle API (Command Line)
1. Install Kaggle API client:
   ```bash
   pip install kaggle
   ```

2. Create Kaggle API credentials:
   - Go to Kaggle Account settings: https://www.kaggle.com/account
   - Click "Create New API Token" to download kaggle.json
   - Place kaggle.json in your home directory under .kaggle folder

3. Download the dataset using API:
   ```bash
   kaggle datasets download -d whenamancodes/fraud-detection
   unzip fraud-detection.zip
   ```

4. Rename the dataset file to expected format:
   ```bash
   mv fraud_detection.csv rba-dataset.csv
   ```

### Step 3: Navigate to Project Directory
```bash
cd oauth-anomaly-detection-pipeline
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Alternative Installation with Virtual Environment
```bash
python -m venv oauth_env
source oauth_env/bin/activate  # On Windows: oauth_env\Scripts\activate
pip install -r requirements.txt
```

## Running the System

### Method 1: GUI Application (Recommended)
1. Execute the main application file:
   ```bash
   python main.py
   ```
2. The graphical interface will open automatically
3. Click "Browse" button to select your authentication dataset file
4. Click "Run Pipeline" to start the analysis process
5. Monitor progress through the interface dashboard
6. View results in the dedicated tabs when processing completes

### Method 2: Command Line Execution
Execute individual stages programmatically:

```bash
# Run preprocessing stage
python preprocessing.py

# Run sampling stage
python stage1.py

# Run validation stage  
python stage2.py

# Run machine learning detection
python stage3.py

# Run security analysis
python stage4.py
```

## Input Data Requirements

The system expects CSV files containing authentication records with the following structure:

### Required Columns
- Login Timestamp: Authentication event timestamp
- User ID: Unique identifier for each user
- Round-Trip Time [ms]: Network latency measurement
- IP Address: Source IP address of authentication attempt
- Country: Geographic location of authentication
- Device Type: Type of device used for authentication

### Sample Data Format
```
Login Timestamp,User ID,Round-Trip Time [ms],IP Address,Country,Device Type
2020-02-03 12:43:30.800,12345,150,192.168.1.1,US,desktop
2020-02-03 12:45:12.200,67890,280,10.0.0.1,CN,mobile
2020-02-03 12:47:05.150,11111,320,203.0.113.5,FR,tablet
```

## Configuration Options

### Sampling Configuration (Stage 1)
Modify the following parameters in stage1.py:
```python
target_sample_size = 255000          # Number of records in final sample
target_anomaly_ratio = 0.125         # Proportion of anomalous records
convergence_threshold = 0.0005       # Entropy convergence criteria
```

### Validation Configuration (Stage 2)
Adjust validation parameters in stage2.py:
```python
max_travel_speed_kmh = 1200          # Maximum feasible travel speed
min_time_threshold_minutes = 3       # Minimum time between locations
```

### Machine Learning Configuration (Stage 3)
Tune model parameters in stage3.py:
```python
n_estimators = 500                   # Number of trees for ensemble models
max_depth = 8                        # Maximum tree depth
learning_rate = 0.1                  # Learning rate for gradient boosting
```

## Output Files and Results

### Generated Files
- stage1_balanced_sampled_rba.csv: Processed sample dataset
- stage2_validated_rba.csv: Validated authentication records
- enhanced_superior_performance_report.json: Model performance metrics
- stage4_fast_llm_comprehensive_report.json: Security analysis results

### Performance Metrics
The system achieves the following benchmark performance:
- Processing Speed: 31 million records processed in approximately 15 minutes
- Data Reduction: 99.17% size reduction while preserving anomaly characteristics
- Classification Accuracy: Greater than 94% across precision, recall, and F1-score metrics
- ROC-AUC Score: Consistently above 98.3% for all ensemble models

## Technical Implementation Details

### Stage 1: Multi-Dimensional Entropy Sampling
Implements intelligent sampling across five entropy dimensions:
- Temporal patterns in authentication timing
- Geographic distribution across countries and regions  
- User behavior consistency and variation patterns
- Device type distribution and switching patterns
- Authentication outcome distribution balancing

### Stage 2: Physical Constraint Validation
Applies real-world feasibility checks including:
- Travel time calculations between geographic locations
- Network round-trip time consistency validation
- Device fingerprint consistency analysis
- Temporal sequence feasibility assessment

### Stage 3: Ensemble Machine Learning
Combines multiple algorithms for robust detection:
- XGBoost gradient boosting with optimized hyperparameters
- LightGBM efficient gradient boosting implementation
- Random Forest ensemble with feature importance ranking
- Neural network models with regularization techniques

### Stage 4: Language Model Analysis
Utilizes fine-tuned models for security interpretation:
- Alert summarization for technical security teams
- Risk explanation generation for management reporting
- Incident response playbook creation for operational teams

## Troubleshooting

### Common Issues and Solutions

**Memory Error During Processing:**
- Reduce chunk_size parameter in preprocessing.py
- Ensure minimum 8GB RAM availability
- Close other applications during execution

**CSV File Format Errors:**
- Verify column headers match required format exactly
- Check for special characters in data fields
- Ensure timestamp format consistency

**Dependency Installation Failures:**
- Update pip to latest version: pip install --upgrade pip
- Install packages individually if batch installation fails
- Use virtual environment to avoid conflicts

**GUI Not Opening:**
- Verify tkinter installation: python -m tkinter
- Check Python version compatibility (3.8+)
- Install tkinter separately if needed: sudo apt-get install python3-tk

## Project File Structure

```
oauth-anomaly-detection-pipeline/
├── main.py                 # Primary application entry point
├── preprocessing.py        # Stage 0 data preprocessing implementation
├── stage1.py              # Stage 1 entropy-guided sampling
├── stage2.py              # Stage 2 physical validation logic
├── stage3.py              # Stage 3 machine learning models
├── stage4.py              # Stage 4 security analysis engine
├── requirements.txt       # Python package dependencies
└── README.md             # Project documentation
```

## Security and Privacy Considerations

The system implements several privacy protection measures:
- User identifier anonymization using cryptographic hashing
- No persistent storage of sensitive authentication data
- Local processing without external data transmission
- Comprehensive audit logging for compliance verification

## Academic and Research Context

This implementation represents a capstone project in advanced cybersecurity, demonstrating:
- Large-scale data processing and analysis techniques
- Integration of traditional machine learning with modern language models
- Real-world application of entropy theory in security contexts
- Production-quality software engineering practices

## Technical References and Standards

The implementation follows established cybersecurity frameworks and standards:
- OAuth 2.0 Security Best Practices (RFC 6749, RFC 6750)
- NIST Cybersecurity Framework guidelines
- MITRE ATT&CK threat modeling methodology
- Industry-standard machine learning evaluation metrics

## Support and Documentation

For detailed technical information, refer to the inline code documentation and comments within each module. The system generates comprehensive logs during execution that can assist with troubleshooting and performance optimization.

This project demonstrates practical application of advanced machine learning techniques to real-world cybersecurity challenges, providing a foundation for enterprise-scale authentication anomaly detection systems.