import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml.quality_checks import run_checks


class QualityCheckTests(unittest.TestCase):
    def test_empty_spreadsheet_rows_are_not_counted_as_transactions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "labeled.csv"
            output = Path(directory) / "quality.md"
            source.write_text(
                "date,raw_description,amount,main_category,sub_category,bank,account,\n"
                "2025-01-01,Store,-4,Food,Groceries,Bank,Checking,\n"
                "2025-01-02,Pay,10,Income,Pay,Bank,Checking,\n"
                ",,,,,,,\n",
                encoding="utf-8",
            )
            with patch("ml.quality_checks.raw_file_summary", return_value=pd.DataFrame()):
                run_checks("jillian", labeled_csv=str(source), out=str(output))
            report = output.read_text(encoding="utf-8")
            self.assertIn("Entirely empty rows excluded: **1**", report)
            self.assertIn("Labeled rows: **2 / 2**", report)
            self.assertNotIn("Unnamed:", report)


if __name__ == "__main__":
    unittest.main()
