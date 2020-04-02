import pytest


from disaster_response_pipeline.models import tokenize, build_model


class TestTrainClassifier:
    @pytest.mark.parametrize(
        "text,include_numbers,expected_result",
        [
            pytest.param(
                "Weather update - a cold front from Cuba that could pass over Haiti.",
                True,
                ["weather", "updat", "cold", "front", "cuba", "could", "pas", "haiti"],
                id="tokenizing_removes_punctuation",
            ),
            pytest.param(
                "More information on the 4636 number in order for me to participate. ( To see if I can use it )",
                False,
                ["more", "inform", "number", "order", "particip", "to", "see", "i", "use"],
                id="tokenizing_without_numbers",
            ),
            pytest.param(
                "More information on the 4636 number in order for me to participate. ( To see if I can use it )",
                True,
                ["more", "inform", "4636", "number", "order", "particip", "to", "see", "i", "use"],
                id="tokenizing_with_numbers",
            ),
            pytest.param(
                "Please check http://www.example.com, you'll find the information you need there",
                True,
                ["pleas", "check", "urlplacehold", "find", "inform", "need"],
                id="removes_urls",
            ),
        ],
    )
    def test_tokenize(self, text, include_numbers, expected_result):

        print(tokenize(text, include_numbers))

        assert tokenize(text, include_numbers) == expected_result

    @pytest.mark.parametrize(
        "clf_name,grid_search,classifier_params,grid_search_params,expected_classifier,expected_params",
        [
            pytest.param(
                None,
                False,
                None,
                None,
                'KNeighborsClassifier',
                {'n_neighbors': 10},
                id="default_classifier_is_KNeighborsClassifier"
            ),
            pytest.param(
                'KNeighborsClassifier',
                False,
                {'leaf_size': 20, 'n_neighbors': 50},
                None,
                'KNeighborsClassifier',
                {'leaf_size': 20, 'n_neighbors': 50},
                id="KNeighborsClassifier;leaf_size=20;n_neighbors=50"
            ),
            pytest.param(
                'RandomForestClassifier',
                False,
                None,
                None,
                'RandomForestClassifier',
                {"n_estimators": 100, "max_depth": 50},
                id="RandomForestClassifier;default_params"
            )
        ],
    )
    def test_build_model(
        self,
        clf_name,
        grid_search,
        classifier_params,
        grid_search_params,
        expected_classifier,
        expected_params,
    ):

        args = [clf_name, grid_search]
        if classifier_params:
            args.append(classifier_params)

        if grid_search_params:
            args.append(grid_search_params)

        model = build_model(*args)

        clf = list(filter(lambda x: x[0] == "clf", model.steps))[0][1].estimator

        assert clf.__class__.__name__ == expected_classifier

        for attribute, attr_value in expected_params.items():
            assert getattr(clf, attribute) == attr_value

    @pytest.mark.parametrize(
        "clf_name,grid_search,classifier_params,grid_search_params,expected_classifier,expected_params",
        [
            pytest.param(
                None,
                True,
                None,
                {'vect__tokenizer': [1], 'tfidf__smooth_idf': [1], 'clf__estimator__n_neighbors': [1]},
                'KNeighborsClassifier',
                {'vect__tokenizer': [1], 'tfidf__smooth_idf': [1], 'clf__estimator__n_neighbors': [1]},
                id="search_grid_params_are_present_in_model"
            ),
        ],
    )
    def test_build_model_with_grid_search(
        self,
        clf_name,
        grid_search,
        classifier_params,
        grid_search_params,
        expected_classifier,
        expected_params,
    ):

        args = [clf_name, grid_search, classifier_params]

        if grid_search_params:
            args.append(grid_search_params)

        model = build_model(*args)

        clf = list(filter(lambda x: x[0] == "clf", model.estimator.steps))[0][1].estimator

        param_grid = model.param_grid

        assert clf.__class__.__name__ == expected_classifier

        print(param_grid)

        for attribute, attr_value in expected_params.items():
            assert param_grid.get(attribute, '') == attr_value
