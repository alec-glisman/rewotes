from pathlib import Path
from src.data_load import MaterialData


def main() -> None:
    file_path = Path(__file__).resolve().parent

    # API key is not included in the code for security reasons
    with open(file_path / "api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    # Load data
    data = MaterialData(api_key, band_gap=(0.0, 1000.0))
    x_train, x_test, y_train, y_test, _ = data.split_data()

    # Train models

    print("complete!")


if __name__ == "__main__":
    main()
