import pandas as pd
from config import DEVICE, PATH, FEATURES, FEATURE_INDEX
from data import process_data
from model import PreoperativeANN, train_model, evaluate_model
from inference import predict_preop_dose


def main():
    print("Using device:", DEVICE)

    # load data
    df = pd.read_csv(PATH)

    # process data
    X_train, y_train, X_test, y_test, stats = process_data(df, device=DEVICE)

    # train model
    model = PreoperativeANN(input_dim=len(FEATURES)).to(DEVICE)
    model = train_model(model, X_train, y_train, epochs=1500, lr=1e-3)

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
