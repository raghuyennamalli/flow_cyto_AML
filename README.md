# Comparison of anomaly detection and supervised learning for acute myeloid leukemia blast identification: A proof-of-concept study

Mezya Sezen1,*, Pavinap Priyaa1,*, Krishnapriya Vinod1, Atul Thatai2, Nitin Dayal2, Ragothaman M. Yennamalli3$, Rama S. Akondy1$.

,* These authors contributed equally to this manuscript.
$ Corresponding Authors

## Table of Contents
• Description • Contents • Materials and Methods • Results • Installation • Usage • Model Usage • File Information • Contact

## Description
This repository accompanies a proof-of-concept study comparing anomaly detection, unsupervised clustering, and supervised classification for identifying leukemic blast cells in multiparameter flow cytometry data from acute myeloid leukemia (AML) patients. We evaluate six computational approaches spanning three methodological families: four anomaly detection algorithms (Isolation Forest, One-Class SVM, Local Outlier Factor, Half-Space Trees), one supervised classifier (Random Forest), and one unsupervised clustering approach (UMAP-HDBSCAN). 

## Contents
- HST/ — Half-Space Trees: custom NumPy anomaly detector, training + external validation scripts 
- Isolation Forest/ — Isolation Forest anomaly detector, nested CV training script 
- LOF/— Local Outlier Factor anomaly detector, nested CV training script
 - Oneclass SVM/ — One-Class SVM anomaly detector, 80/10/10 split training + external test 
-Preprocessing/ — R scripts for margin/debris/doublet removal, logicle transformation, and scaling
 - Random Forest/ — Supervised Random Forest classifier, nested CV training + SHAP/feature importance
 - UMAP_HDBSCAN/— Unsupervised dimensionality reduction + clustering 

## Materials and Methods
Data source: BLAST110 and LAIP29 cohorts (Mocking et al., 2024), Zenodo (https://zenodo.org/records/11046402). BLAST110: 110 bone marrow samples (20 healthy, 30 diagnosis, 60 follow-up). LAIP29: 48 samples (28 diagnosis, 20 follow-up). Each sample measured across 4 antibody panels (P1–P4); 5 backbone markers common to all panels. 
Feature set: SSC-A, CD34, CD45, CD117, CD13 (5 backbone markers)
Model validation: 5-fold nested cross-validation with StratifiedGroupKFold, grouped by patient ID to prevent leakage. Inner loop: hyperparameter tuning. Outer loop: unbiased performance estimation. External validation on two independent LAIP29 samples (~41% and ~1% blast prevalence). 

Results

----------

## Contact
Dr. Rama S. Akondy, Department of Biology, Trivedi School of Biosciences, Ashoka University, Sonipat, India 131029. Email: rama.akondy@ashoka.edu.in

Dr. Ragothaman M. Yennamalli, School of Computational and Integrative Sciences, Jawaharlal Nehru University, New Delhi, India 110067. Email: ragothaman@jnu.ac.in

