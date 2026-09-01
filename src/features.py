import pandas as pd


MAIN_TITLES = [
    "Mr",
    "Mrs",
    "Miss",
    "Master",
]


def extract_title(
    name: pd.Series,
) -> pd.Series:
    title = name.str.extract(
        r",\s*([^.]+)\.",
        expand=False,
    )

    return title.where(
        title.isin(MAIN_TITLES),
        "Other",
    )


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    # Age 결측 여부
    result["AgeWasMissing"] = (
        result["Age"]
        .isna()
        .astype(int)
    )

    # Cabin 기록 여부
    result["CabinWasMissing"] = (
        result["Cabin"]
        .isna()
        .astype(int)
    )

    # Name에서 Title 추출
    result["Title"] = extract_title(
        result["Name"]
    )

    # 본인을 포함한 가족 수
    result["FamilySize"] = (
        result["SibSp"]
        + result["Parch"]
        + 1
    )

    # 혼자 탑승했는지 여부
    result["IsAlone"] = (
        result["FamilySize"] == 1
    ).astype(int)

    return result