"""
Comprehensive Testing and Validation System for ASL Recognition
Implements advanced testing including confusion matrix analysis, multi-frame voting, and real-world testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import cv2
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    test_type: str
    predicted_letter: str
    actual_letter: str
    confidence: float
    processing_time: float
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class TestSession:
    """Complete testing session"""
    session_id: str
    session_type: str
    start_time: float
    end_time: Optional[float] = None
    total_tests: int = 0
    correct_predictions: int = 0
    results: List[TestResult] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    test_conditions: Dict[str, Any] = field(default_factory=dict)

class ConfusionMatrixAnalyzer:
    """Advanced confusion matrix analysis"""
    
    def __init__(self):
        self.letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.confusion_data = []
        
    def add_prediction(self, actual: str, predicted: str, confidence: float = 1.0):
        """Add a prediction result"""
        self.confusion_data.append({
            'actual': actual,
            'predicted': predicted,
            'confidence': confidence,
            'correct': actual == predicted
        })
    
    def generate_confusion_matrix(self, normalize: str = None) -> np.ndarray:
        """Generate confusion matrix"""
        if not self.confusion_data:
            return np.zeros((26, 26))
        
        actual_labels = [d['actual'] for d in self.confusion_data]
        predicted_labels = [d['predicted'] for d in self.confusion_data]
        
        cm = confusion_matrix(actual_labels, predicted_labels, 
                            labels=self.letters, normalize=normalize)
        return cm
    
    def plot_confusion_matrix(self, save_path: str = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot detailed confusion matrix"""
        cm = self.generate_confusion_matrix(normalize='true')
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.letters, yticklabels=self.letters,
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title('ASL Letter Recognition Confusion Matrix')
        plt.xlabel('Predicted Letter')
        plt.ylabel('Actual Letter')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze common errors and patterns"""
        if not self.confusion_data:
            return {}
        
        # Most confused letter pairs
        error_pairs = defaultdict(int)
        for data in self.confusion_data:
            if not data['correct']:
                pair = (data['actual'], data['predicted'])
                error_pairs[pair] += 1
        
        most_confused = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Per-letter accuracy
        letter_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for data in self.confusion_data:
            letter_stats[data['actual']]['total'] += 1
            if data['correct']:
                letter_stats[data['actual']]['correct'] += 1
        
        letter_accuracy = {
            letter: stats['correct'] / max(stats['total'], 1)
            for letter, stats in letter_stats.items()
        }
        
        # Confidence vs accuracy correlation
        correct_confidences = [d['confidence'] for d in self.confusion_data if d['correct']]
        incorrect_confidences = [d['confidence'] for d in self.confusion_data if not d['correct']]
        
        return {
            'most_confused_pairs': most_confused,
            'letter_accuracy': letter_accuracy,
            'worst_performing_letters': sorted(letter_accuracy.items(), key=lambda x: x[1])[:5],
            'best_performing_letters': sorted(letter_accuracy.items(), key=lambda x: x[1], reverse=True)[:5],
            'avg_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0,
            'avg_confidence_incorrect': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'confidence_threshold_analysis': self._analyze_confidence_thresholds()
        }
    
    def _analyze_confidence_thresholds(self) -> Dict[str, float]:
        """Analyze optimal confidence thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = {}
        
        for threshold in thresholds:
            filtered_data = [d for d in self.confusion_data if d['confidence'] >= threshold]
            if not filtered_data:
                continue
            
            accuracy = sum(d['correct'] for d in filtered_data) / len(filtered_data)
            coverage = len(filtered_data) / len(self.confusion_data)
            
            threshold_metrics[f"{threshold:.1f}"] = {
                'accuracy': accuracy,
                'coverage': coverage,
                'f1_score': 2 * accuracy * coverage / (accuracy + coverage) if (accuracy + coverage) > 0 else 0
            }
        
        return threshold_metrics

class MultiFrameVotingSystem:
    """Multi-frame voting mechanism for improved accuracy"""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.7,
                 consistency_threshold: float = 0.6):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.consistency_threshold = consistency_threshold
        self.prediction_window = []
        self.voting_history = []
        
    def add_prediction(self, letter: str, confidence: float, timestamp: float = None) -> Optional[Dict[str, Any]]:
        """Add prediction to voting window"""
        prediction = {
            'letter': letter,
            'confidence': confidence,
            'timestamp': timestamp or time.time()
        }
        
        self.prediction_window.append(prediction)
        
        # Maintain window size
        if len(self.prediction_window) > self.window_size:
            self.prediction_window.pop(0)
        
        # Perform voting if window is full
        if len(self.prediction_window) >= self.window_size:
            return self._perform_voting()
        
        return None
    
    def _perform_voting(self) -> Dict[str, Any]:
        """Perform multi-frame voting"""
        # Simple majority voting
        letter_votes = Counter(p['letter'] for p in self.prediction_window)
        confidence_avg = np.mean([p['confidence'] for p in self.prediction_window])
        
        # Weighted voting by confidence
        weighted_votes = defaultdict(float)
        total_weight = 0
        
        for pred in self.prediction_window:
            if pred['confidence'] >= self.confidence_threshold:
                weight = pred['confidence']
                weighted_votes[pred['letter']] += weight
                total_weight += weight
        
        # Determine final prediction
        if total_weight > 0:
            # Use weighted voting
            final_letter = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_confidence = weighted_votes[final_letter] / total_weight
        else:
            # Fallback to simple majority
            final_letter = letter_votes.most_common(1)[0][0]
            final_confidence = confidence_avg
        
        # Check consistency
        majority_count = letter_votes[final_letter]
        consistency = majority_count / len(self.prediction_window)
        
        voting_result = {
            'letter': final_letter,
            'confidence': final_confidence,
            'consistency': consistency,
            'is_reliable': consistency >= self.consistency_threshold,
            'vote_distribution': dict(letter_votes),
            'window_size': len(self.prediction_window),
            'timestamp': time.time()
        }
        
        self.voting_history.append(voting_result)
        return voting_result
    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """Get voting system statistics"""
        if not self.voting_history:
            return {}
        
        reliabilities = [v['consistency'] for v in self.voting_history]
        confidences = [v['confidence'] for v in self.voting_history]
        
        return {
            'total_votes': len(self.voting_history),
            'avg_consistency': np.mean(reliabilities),
            'avg_confidence': np.mean(confidences),
            'reliability_rate': sum(v['is_reliable'] for v in self.voting_history) / len(self.voting_history),
            'confidence_distribution': {
                'min': np.min(confidences),
                'max': np.max(confidences),
                'std': np.std(confidences)
            }
        }

class RealWorldTestingFramework:
    """Framework for testing under various real-world conditions"""
    
    def __init__(self):
        self.test_conditions = {
            'lighting': ['bright', 'normal', 'dim', 'variable'],
            'background': ['plain', 'complex', 'moving', 'cluttered'],
            'camera_angle': ['front', 'side', 'tilted', 'distant'],
            'hand_position': ['center', 'edge', 'partial', 'moving'],
            'user_variation': ['consistent', 'fast', 'slow', 'varied_style']
        }
        
        self.test_results = defaultdict(list)
        self.condition_combinations = []
        
    def generate_test_matrix(self) -> List[Dict[str, str]]:
        """Generate comprehensive test condition matrix"""
        import itertools
        
        # Generate all combinations (subset for practicality)
        conditions = list(self.test_conditions.keys())
        combinations = []
        
        # Full factorial would be too many, so we'll use a reduced set
        # focusing on key combinations
        key_combinations = [
            {'lighting': 'bright', 'background': 'plain', 'camera_angle': 'front', 
             'hand_position': 'center', 'user_variation': 'consistent'},
            {'lighting': 'dim', 'background': 'complex', 'camera_angle': 'side', 
             'hand_position': 'edge', 'user_variation': 'fast'},
            {'lighting': 'normal', 'background': 'moving', 'camera_angle': 'tilted', 
             'hand_position': 'partial', 'user_variation': 'slow'},
            {'lighting': 'variable', 'background': 'cluttered', 'camera_angle': 'distant', 
             'hand_position': 'moving', 'user_variation': 'varied_style'},
        ]
        
        # Add systematic variations
        for base_condition in key_combinations:
            combinations.append(base_condition.copy())
            
            # Vary one condition at a time
            for condition_type, values in self.test_conditions.items():
                for value in values:
                    if value != base_condition[condition_type]:
                        variation = base_condition.copy()
                        variation[condition_type] = value
                        combinations.append(variation)
        
        # Remove duplicates
        unique_combinations = []
        seen = set()
        for combo in combinations:
            combo_tuple = tuple(sorted(combo.items()))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(combo)
        
        self.condition_combinations = unique_combinations
        return unique_combinations
    
    def run_condition_test(self, condition: Dict[str, str], 
                          test_function: Callable, test_data: List[Any]) -> Dict[str, Any]:
        """Run test under specific conditions"""
        condition_results = []
        start_time = time.time()
        
        logger.info(f"Testing condition: {condition}")
        
        for test_item in test_data:
            try:
                # Apply condition modifications to test
                modified_test = self._apply_conditions(test_item, condition)
                
                # Run test
                result = test_function(modified_test)
                result['condition'] = condition
                condition_results.append(result)
                
            except Exception as e:
                logger.error(f"Error in condition test: {e}")
                continue
        
        end_time = time.time()
        
        # Analyze results
        condition_analysis = self._analyze_condition_results(condition_results)
        condition_analysis['test_duration'] = end_time - start_time
        condition_analysis['condition'] = condition
        
        self.test_results[str(condition)] = condition_analysis
        
        return condition_analysis
    
    def _apply_conditions(self, test_data: Any, condition: Dict[str, str]) -> Any:
        """Apply real-world conditions to test data"""
        # This would implement condition-specific modifications
        # For now, we'll simulate by adding metadata
        if hasattr(test_data, 'copy'):
            modified = test_data.copy()
        else:
            modified = test_data
            
        # Add condition metadata
        if isinstance(modified, dict):
            modified['test_conditions'] = condition
        
        return modified
    
    def _analyze_condition_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results for specific conditions"""
        if not results:
            return {'accuracy': 0, 'confidence': 0, 'total_tests': 0}
        
        correct = sum(1 for r in results if r.get('correct', False))
        total = len(results)
        confidences = [r.get('confidence', 0) for r in results]
        processing_times = [r.get('processing_time', 0) for r in results]
        
        return {
            'accuracy': correct / total,
            'avg_confidence': np.mean(confidences),
            'avg_processing_time': np.mean(processing_times),
            'total_tests': total,
            'confidence_std': np.std(confidences),
            'processing_time_std': np.std(processing_times),
            'error_rate': 1 - (correct / total)
        }
    
    def get_condition_ranking(self) -> List[Tuple[str, float]]:
        """Get conditions ranked by difficulty"""
        condition_scores = []
        
        for condition_str, results in self.test_results.items():
            # Difficulty score based on accuracy and confidence
            accuracy = results.get('accuracy', 0)
            confidence = results.get('avg_confidence', 0)
            difficulty_score = 1 - (accuracy * 0.7 + confidence * 0.3)
            
            condition_scores.append((condition_str, difficulty_score))
        
        return sorted(condition_scores, key=lambda x: x[1], reverse=True)
    
    def generate_condition_report(self, save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive condition testing report"""
        if not self.test_results:
            return {}
        
        # Overall statistics
        all_accuracies = [r['accuracy'] for r in self.test_results.values()]
        all_confidences = [r['avg_confidence'] for r in self.test_results.values()]
        
        # Condition analysis
        condition_ranking = self.get_condition_ranking()
        
        # Factor analysis (which factors impact performance most)
        factor_impact = self._analyze_factor_impact()
        
        report = {
            'overall_statistics': {
                'avg_accuracy': np.mean(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies),
                'accuracy_std': np.std(all_accuracies),
                'avg_confidence': np.mean(all_confidences),
                'conditions_tested': len(self.test_results)
            },
            'most_challenging_conditions': condition_ranking[:5],
            'easiest_conditions': condition_ranking[-5:],
            'factor_impact_analysis': factor_impact,
            'detailed_results': dict(self.test_results),
            'recommendations': self._generate_recommendations()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _analyze_factor_impact(self) -> Dict[str, float]:
        """Analyze which factors have the most impact on performance"""
        factor_impacts = {}
        
        for factor in self.test_conditions.keys():
            factor_results = defaultdict(list)
            
            # Group results by factor value
            for condition_str, results in self.test_results.items():
                condition = eval(condition_str)  # Convert string back to dict
                factor_value = condition.get(factor, 'unknown')
                factor_results[factor_value].append(results['accuracy'])
            
            # Calculate impact (variance in accuracy across factor values)
            if len(factor_results) > 1:
                all_accuracies = [np.mean(accs) for accs in factor_results.values()]
                factor_impacts[factor] = np.std(all_accuracies)
            else:
                factor_impacts[factor] = 0
        
        return factor_impacts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze factor impacts
        factor_impacts = self._analyze_factor_impact()
        most_impactful = max(factor_impacts.items(), key=lambda x: x[1])
        
        recommendations.append(
            f"Focus on improving robustness to {most_impactful[0]} variations "
            f"(impact score: {most_impactful[1]:.3f})"
        )
        
        # Analyze accuracy distribution
        all_accuracies = [r['accuracy'] for r in self.test_results.values()]
        if np.std(all_accuracies) > 0.1:
            recommendations.append(
                "High accuracy variance across conditions suggests need for "
                "more robust feature extraction or adaptive algorithms"
            )
        
        # Low-accuracy conditions
        low_accuracy_conditions = [
            condition for condition, score in self.get_condition_ranking()[:3]
        ]
        recommendations.append(
            f"Prioritize testing and improvement for: {', '.join(low_accuracy_conditions[:2])}"
        )
        
        return recommendations

class ComprehensiveValidator:
    """Main comprehensive validation system"""
    
    def __init__(self, model_predictor: Callable, feature_extractor: Callable):
        self.model_predictor = model_predictor
        self.feature_extractor = feature_extractor
        
        # Analysis components
        self.confusion_analyzer = ConfusionMatrixAnalyzer()
        self.voting_system = MultiFrameVotingSystem()
        self.real_world_tester = RealWorldTestingFramework()
        
        # Test sessions
        self.test_sessions = []
        self.current_session = None
        
    def start_test_session(self, session_type: str = "validation", 
                          test_conditions: Dict[str, Any] = None) -> str:
        """Start a new test session"""
        session_id = f"session_{int(time.time())}"
        
        self.current_session = TestSession(
            session_id=session_id,
            session_type=session_type,
            start_time=time.time(),
            test_conditions=test_conditions or {}
        )
        
        logger.info(f"Started test session: {session_id}")
        return session_id
    
    def run_single_test(self, image: np.ndarray, actual_letter: str, 
                       test_metadata: Dict[str, Any] = None) -> TestResult:
        """Run a single test case"""
        test_id = f"test_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Extract features
            features = self.feature_extractor(image)
            
            # Get prediction
            prediction = self.model_predictor(features)
            
            processing_time = time.time() - start_time
            
            # Create test result
            result = TestResult(
                test_id=test_id,
                test_type=self.current_session.session_type if self.current_session else "single",
                predicted_letter=prediction.get('letter', ''),
                actual_letter=actual_letter,
                confidence=prediction.get('confidence', 0.0),
                processing_time=processing_time,
                features=features,
                metadata=test_metadata or {}
            )
            
            # Add to current session
            if self.current_session:
                self.current_session.results.append(result)
                self.current_session.total_tests += 1
                if result.predicted_letter == result.actual_letter:
                    self.current_session.correct_predictions += 1
            
            # Add to confusion matrix
            self.confusion_analyzer.add_prediction(
                actual_letter, result.predicted_letter, result.confidence
            )
            
            # Add to voting system
            voting_result = self.voting_system.add_prediction(
                result.predicted_letter, result.confidence
            )
            
            result.metadata['voting_result'] = voting_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single test: {e}")
            # Return error result
            return TestResult(
                test_id=test_id,
                test_type="error",
                predicted_letter="",
                actual_letter=actual_letter,
                confidence=0.0,
                processing_time=time.time() - start_time,
                features=np.array([]),
                metadata={'error': str(e)}
            )
    
    def run_comprehensive_validation(self, test_dataset: List[Tuple[np.ndarray, str]], 
                                   save_results: bool = True, 
                                   results_dir: str = "validation_results") -> Dict[str, Any]:
        """Run comprehensive validation on test dataset"""
        session_id = self.start_test_session("comprehensive_validation")
        
        logger.info(f"Starting comprehensive validation with {len(test_dataset)} samples")
        
        # Run basic tests
        basic_results = []
        for i, (image, actual_letter) in enumerate(test_dataset):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(test_dataset)} samples")
                
            result = self.run_single_test(image, actual_letter)
            basic_results.append(result)
        
        # End session
        self.end_test_session()
        
        # Generate reports
        validation_report = self._generate_validation_report(basic_results)
        
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(exist_ok=True)
            
            # Save confusion matrix
            cm_path = results_path / "confusion_matrix.png"
            self.confusion_analyzer.plot_confusion_matrix(str(cm_path))
            
            # Save detailed report
            report_path = results_path / "validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {results_dir}")
        
        return validation_report
    
    def run_real_world_testing(self, test_function: Callable, test_data: List[Any],
                             save_results: bool = True) -> Dict[str, Any]:
        """Run real-world condition testing"""
        logger.info("Starting real-world testing")
        
        # Generate test conditions
        test_conditions = self.real_world_tester.generate_test_matrix()
        logger.info(f"Generated {len(test_conditions)} test conditions")
        
        # Run tests for each condition
        for i, condition in enumerate(test_conditions[:10]):  # Limit for demo
            logger.info(f"Testing condition {i+1}/{min(10, len(test_conditions))}")
            self.real_world_tester.run_condition_test(condition, test_function, test_data)
        
        # Generate comprehensive report
        report = self.real_world_tester.generate_condition_report()
        
        if save_results:
            report_path = "real_world_testing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Real-world testing report saved to {report_path}")
        
        return report
    
    def end_test_session(self):
        """End current test session"""
        if self.current_session:
            self.current_session.end_time = time.time()
            
            # Calculate session metrics
            if self.current_session.total_tests > 0:
                accuracy = self.current_session.correct_predictions / self.current_session.total_tests
                avg_confidence = np.mean([r.confidence for r in self.current_session.results])
                avg_processing_time = np.mean([r.processing_time for r in self.current_session.results])
                
                self.current_session.performance_metrics = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'avg_processing_time': avg_processing_time,
                    'total_duration': self.current_session.end_time - self.current_session.start_time
                }
            
            self.test_sessions.append(self.current_session)
            session_id = self.current_session.session_id
            self.current_session = None
            
            logger.info(f"Ended test session: {session_id}")
    
    def _generate_validation_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not results:
            return {}
        
        # Basic metrics
        correct = sum(1 for r in results if r.predicted_letter == r.actual_letter)
        total = len(results)
        accuracy = correct / total
        
        confidences = [r.confidence for r in results]
        processing_times = [r.processing_time for r in results]
        
        # Confusion matrix analysis
        error_analysis = self.confusion_analyzer.get_error_analysis()
        
        # Voting system statistics
        voting_stats = self.voting_system.get_voting_statistics()
        
        # Per-letter performance
        letter_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
        for result in results:
            letter_performance[result.actual_letter]['total'] += 1
            letter_performance[result.actual_letter]['confidences'].append(result.confidence)
            if result.predicted_letter == result.actual_letter:
                letter_performance[result.actual_letter]['correct'] += 1
        
        letter_stats = {}
        for letter, stats in letter_performance.items():
            letter_stats[letter] = {
                'accuracy': stats['correct'] / stats['total'],
                'total_samples': stats['total'],
                'avg_confidence': np.mean(stats['confidences']),
                'confidence_std': np.std(stats['confidences'])
            }
        
        return {
            'overall_metrics': {
                'accuracy': accuracy,
                'total_tests': total,
                'correct_predictions': correct,
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'avg_processing_time': np.mean(processing_times),
                'processing_time_std': np.std(processing_times)
            },
            'error_analysis': error_analysis,
            'voting_system_stats': voting_stats,
            'per_letter_performance': letter_stats,
            'test_session_info': {
                'session_id': self.current_session.session_id if self.current_session else 'completed',
                'session_type': self.current_session.session_type if self.current_session else 'validation',
                'timestamp': time.time()
            }
        }
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all testing"""
        return {
            'total_test_sessions': len(self.test_sessions),
            'confusion_matrix_summary': self.confusion_analyzer.get_error_analysis(),
            'voting_system_summary': self.voting_system.get_voting_statistics(),
            'real_world_testing_summary': self.real_world_tester.get_condition_ranking(),
            'session_summaries': [
                {
                    'session_id': session.session_id,
                    'session_type': session.session_type,
                    'performance_metrics': session.performance_metrics,
                    'total_tests': session.total_tests
                }
                for session in self.test_sessions
            ]
        }


if __name__ == "__main__":
    # Example usage
    def mock_feature_extractor(image):
        return np.random.rand(100)
    
    def mock_model_predictor(features):
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return {
            'letter': np.random.choice(letters),
            'confidence': np.random.rand()
        }
    
    # Initialize validator
    validator = ComprehensiveValidator(mock_model_predictor, mock_feature_extractor)
    
    # Generate mock test data
    test_data = []
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for _ in range(100):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        letter = np.random.choice(letters)
        test_data.append((image, letter))
    
    # Run comprehensive validation
    report = validator.run_comprehensive_validation(test_data)
    
    print("Validation completed!")
    print(f"Overall accuracy: {report['overall_metrics']['accuracy']:.3f}")
    print(f"Average confidence: {report['overall_metrics']['avg_confidence']:.3f}")