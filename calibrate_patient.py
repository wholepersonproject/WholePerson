#!/usr/bin/env python3
"""
Patient-specific model calibration using wearable data
"""

from core.state import SimulationState
from core.entity_factory import EntityFactory, ProcessLoader
from engine.engine import PhysiologyEngine
from training.data_loader import PatientDataLoader
from training.calibrator import ModelCalibrator
from training.validator import SimulationValidator

def calibrate_patient(patient_id="patient_001"):
    print("="*70)
    print(f"PATIENT-SPECIFIC CALIBRATION: {patient_id}")
    print("="*70)
    print()
    
    # Load patient data
    loader = PatientDataLoader(patient_id)
    patient_data = loader.load_all()
    print()
    
    # Initialize simulation
    state = SimulationState()
    factory = EntityFactory("configs/anatomy.yaml")
    factory.initialize_simulation_state(state)
    print()
    
    engine = PhysiologyEngine(state)
    process_loader = ProcessLoader("configs/processes.yaml")
    process_loader.load_all_processes(engine)
    print()
    
    # Validate baseline
    print("BASELINE VALIDATION (before calibration):")
    validator = SimulationValidator(engine, patient_data)
    baseline_metrics = validator.validate_glucose()
    
    # Calibrate
    calibrator = ModelCalibrator(engine, patient_data)
    target_params = {
        'insulin_secretion': ['glucose_sensitivity', 'max_secretion'],
        'glucose_uptake_muscle': ['insulin_sensitivity']
    }
    calibrator.calibrate(target_params, max_iter=15)
    
    # Validate after calibration
    print("POST-CALIBRATION VALIDATION:")
    post_metrics = validator.validate_glucose()
    
    # Compare
    print("="*70)
    print("IMPROVEMENT:")
    print("="*70)
    improvement = ((baseline_metrics['rmse'] - post_metrics['rmse']) / baseline_metrics['rmse']) * 100
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} → {post_metrics['rmse']:.2f} mg/dL ({improvement:+.1f}%)")
    print("="*70)
    print()
    
    return engine, calibrator, validator

if __name__ == "__main__":
    engine, calibrator, validator = calibrate_patient("patient_001")