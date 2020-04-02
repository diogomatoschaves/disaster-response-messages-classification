import pytest

import pandas as pd

from disaster_response_pipeline.data import clean_data


class TestProcessData:
    @pytest.mark.parametrize(
        "messages,categories,expected_df",
        [
            pytest.param(
                pd.DataFrame({"id": [1, 2], "content": ["xyz", "abc"]}),
                pd.DataFrame(
                    {"id": [1, 2], "categories": ["cat1-1;cat2-0", "cat1-0;cat2-1"]}
                ),
                pd.DataFrame(
                    {
                        "id": [1, 2],
                        "content": ["xyz", "abc"],
                        "cat1": [1, 0],
                        "cat2": [0, 1],
                    }
                ),
                id="it_splits_categories",
            ),
            pytest.param(
                pd.DataFrame({"id": [1, 2, 2], "content": ["xyz", "abc", "abc"]}),
                pd.DataFrame(
                    {
                        "id": [1, 2, 2],
                        "categories": [
                            "cat1-1;cat2-0",
                            "cat1-0;cat2-1",
                            "cat1-0;cat2-1",
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "id": [1, 2],
                        "content": ["xyz", "abc"],
                        "cat1": [1, 0],
                        "cat2": [0, 1],
                    }
                ),
                id="it_drops_duplicates",
            ),
            pytest.param(
                pd.DataFrame({"id": [1, 2, 3], "content": ["xyz", "abc", "def"]}),
                pd.DataFrame(
                    {"id": [1, 2], "categories": ["cat1-1;cat2-0", "cat1-0;cat2-1"]}
                ),
                pd.DataFrame(
                    {
                        "id": [1, 2],
                        "content": ["xyz", "abc"],
                        "cat1": [1, 0],
                        "cat2": [0, 1],
                    }
                ),
                id="it_drops_messages_with_no_categories",
            ),
            pytest.param(
                pd.DataFrame({"id": [1, 2], "content": ["xyz", "abc"]}),
                pd.DataFrame(
                    {
                        "id": [1, 2, 3],
                        "categories": [
                            "cat1-1;cat2-0",
                            "cat1-0;cat2-1",
                            "cat1-1;cat2-1",
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "id": [1, 2],
                        "content": ["xyz", "abc"],
                        "cat1": [1, 0],
                        "cat2": [0, 1],
                    }
                ),
                id="it_drops_categories_with_no_messages",
            ),
        ],
    )
    def test_clean_data_method(self, messages, categories, expected_df):

        result = clean_data(messages, categories)

        assert expected_df.equals(result)
