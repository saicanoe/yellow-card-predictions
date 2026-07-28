from xgboost import XGBClassifier, XGBRegressor

from features import FEATURE_COLUMNS


def train_models(training_data):
    """Train the regression and over-4.5 classifier with existing parameters."""
    x_train = training_data[FEATURE_COLUMNS]

    reg_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    reg_model.fit(x_train, training_data["total_cards"])

    clf_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    clf_model.fit(x_train, training_data["over_4_5"])

    return reg_model, clf_model


def add_predictions(fixtures, reg_model, clf_model):
    """Add predicted cards and over-4.5 probability to upcoming fixtures."""
    fixtures = fixtures.copy()
    x_upcoming = fixtures[FEATURE_COLUMNS]
    fixtures["predicted_cards"] = reg_model.predict(x_upcoming)
    fixtures["over_4_5_prob"] = clf_model.predict_proba(x_upcoming)[:, 1]
    return fixtures
