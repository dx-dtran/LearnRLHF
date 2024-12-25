import unittest
import torch
from sft_data import collate_fn, SupervisedFineTuningDataset
import tempfile
import json


class TestSupervisedFineTuning(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        self.sample_data = [
            {"chosen": [1, 2, 3, 4]},
            {"chosen": [5, 6]},
            {"chosen": [7, 8, 9]},
        ]
        for entry in self.sample_data:
            self.temp_file.write(json.dumps(entry) + "\n")
        self.temp_file.close()

        self.dataset = SupervisedFineTuningDataset(jsonl_file=self.temp_file.name)

    def tearDown(self):
        import os

        os.unlink(self.temp_file.name)

    def test_collate_fn(self):
        batch = [self.dataset[i] for i in range(len(self.dataset))]
        collated = collate_fn(batch)

        expected_input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0], [7, 8, 9, 0]])
        expected_target_ids = torch.tensor(
            [[2, 3, 4, -100], [6, -100, -100, -100], [8, 9, -100, -100]]
        )
        expected_attention_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
                [True, True, True, False],
            ]
        )

        self.assertEqual(collated["input_ids"].shape, expected_input_ids.shape)
        self.assertEqual(collated["target_ids"].shape, expected_target_ids.shape)
        self.assertEqual(
            collated["attention_mask"].shape, expected_attention_mask.shape
        )

        self.assertTrue(torch.equal(collated["input_ids"], expected_input_ids))
        self.assertTrue(torch.equal(collated["target_ids"], expected_target_ids))
        self.assertTrue(
            torch.equal(collated["attention_mask"], expected_attention_mask)
        )


if __name__ == "__main__":
    unittest.main()
