import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from environmental_analysis.analysis import clean_data, run_analysis
from environmental_analysis.sample import generate_sample


class AnalysisTests(unittest.TestCase):
    def test_quality_and_target_validation(self):
        frame = generate_sample(100)
        frame.loc[10, 'pm25_ug_m3'] = np.inf
        cleaned, quality = clean_data(frame, 'pm25_ug_m3')
        self.assertEqual(quality['duplicates_removed'], 3)
        self.assertEqual(quality['missing_or_invalid_targets_removed'], 1)
        self.assertEqual(len(cleaned), 99)
        self.assertGreater(cleaned.isna().sum().sum(), 0)
        with self.assertRaisesRegex(ValueError, 'Target column'):
            clean_data(frame, 'absent')

    def test_end_to_end_chronological(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = run_analysis(generate_sample(150), 'pm25_ug_m3', directory,
                                    date_column='date', synthetic=True)
            self.assertEqual(metadata['train_rows'], 120)
            self.assertEqual(metadata['test_rows'], 30)
            output = Path(directory)
            metrics = pd.read_csv(output / 'model_metrics.csv').set_index('model')
            self.assertLess(metrics.loc['ridge', 'rmse'], metrics.loc['mean_baseline', 'rmse'])
            self.assertTrue(np.isfinite(metrics.to_numpy()).all())
            predictions = pd.read_csv(output / 'predictions.csv')
            self.assertGreaterEqual(predictions.source_row.min(), 120)
            self.assertEqual(json.loads((output / 'analysis_metadata.json').read_text())['seed'], 42)
            self.assertTrue((output / 'evaluation.png').read_bytes().startswith(b'\x89PNG'))
            self.assertIn('Synthetic demonstration', (output / 'report.md').read_text())

    def test_predictor_missing_throughout_training(self):
        frame = generate_sample(100).drop_duplicates()
        frame['only_in_test'] = np.nan
        frame.loc[80:, 'only_in_test'] = 1000.0
        with tempfile.TemporaryDirectory() as directory:
            result = run_analysis(frame, 'pm25_ug_m3', directory, date_column='date')
            self.assertEqual(result['test_rows'], 20)
            metrics = pd.read_csv(Path(directory) / 'model_metrics.csv')
            self.assertTrue(np.isfinite(metrics[['mae', 'rmse', 'r2']]).all().all())

    def test_invalid_dates_rejected(self):
        frame = generate_sample(50)
        frame['date'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'invalid dates'):
            run_analysis(frame, 'pm25_ug_m3', '/tmp/unused-analysis', date_column='date')


if __name__ == '__main__':
    unittest.main()
