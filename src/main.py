import pandas as pd
from config import DEVICE, PATH, FEATURES, FEATURE_INDEX, MODEL_NAME
from data import process_data
from model import get_model, train_model, evaluate_model
from inference import predict_preop_dose


def main():
    print("Using device:", DEVICE)

    # load data
    df = pd.read_csv(PATH)

    # process data
    X_train, y_train, X_test, y_test, stats = process_data(df, device=DEVICE)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("X_train first row:", X_train[0])
    print("y_train first row:", y_train[0])

    print("X_train column means:", X_train.mean(dim=0))
    print("X_train column stds: ", X_train.std(dim=0))
    print("y_train mean:", y_train.mean().item(), "std:", y_train.std().item())
    # train model
    model = get_model(MODEL_NAME, input_dim=len(FEATURES)).to(DEVICE)
    model = train_model(model, X_train, y_train, epochs=2000, lr=0.2, weight_decay=0)

    # evaluate model
    mae, mape = evaluate_model(model, X_test, y_test)
    print(f"Test MAE:  {mae:.6f}")
    print(f"Test MAPE: {mape:.3f}%")

    # user interaction loop
    while True:
        try:
            print("\nEnter patient details to predict preoperative dose:")
            # get patient, sex, age, weight, sbp, dbp
            patientInfo = input("Format - sex (0/1), age, weight, sbp, dbp: ")
            patientSex, patientAge, patientWeight, patientSbp, patientDbp = patientInfo.split(",")
            pred, warnings = predict_preop_dose(
                                sex=int(patientSex), age=float(patientAge), weight=float(patientWeight), sbp=float(patientSbp), dbp=float(patientDbp),
                                model=model,
                                stats=stats,
                                feature_index=FEATURE_INDEX,
                            )
            print(pred)
        except (KeyboardInterrupt, ValueError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
