import unittest

from nn_scratch.datasets import(
    load_article_prediction_examples,
    load_article_training_data,
    load_easy_points_data,
)


class TestDatasets(unittest.TestCase):
    def test_people_training_data_loads(self):
        X, y, names = load_article_training_data()

        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4,))
        self.assertEqual(len(names), 4)
        self.assertIn("Alice",names)


    def test_people_prediction_examples_load(self):
        X, names = load_article_prediction_examples()


        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(len(names), 2)
        self.assertIn("Emily", names)
        self.assertIn("Frank", names)

    def test_easy_points_data_loads(self):
        X, y, names = load_easy_points_data()


        self.assertEqual(X.shape, (10, 2))
        self.assertEqual(y.shape, (10,))
        self.assertEqual(len(names), 10)


if __name__ == "__main__":
    unittest.main()