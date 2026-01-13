import pandas as pd
from config import DEVICE, PATH, FEATURES, FEATURE_INDEX, MODEL_NAME
from data import process_data
from model import bucketed_metrics_by_dose, get_model, train_model, evaluate_model, get_test_predictions
from inference import predict_preop_dose


def main():
    print("Using device:", DEVICE)

    # load data
    df = pd.read_csv(PATH)

    # process data
    X_train, y_train, X_test, y_test, x_stats, y_stats = process_data(df, device=DEVICE)

    # train model
    model = get_model(MODEL_NAME, input_dim=len(FEATURES)).to(DEVICE)
    model = train_model(model, X_train, y_train, epochs=100, lr=1e-3, weight_decay=3e-3)

    # evaluate model
    mae, mape = evaluate_model(model, X_test, y_test, y_stats)
    print(f"Test MAE:  {mae:.6f}")
    print(f"Test MAPE: {mape:.3f}%")

    y_true, y_pred = get_test_predictions(model, X_test, y_test, y_stats)
    bucketed_metrics_by_dose(y_true, y_pred)

    # user interaction loop
    while True:
        try:
            print("\nEnter patient details to predict preoperative dose:")
            # get patient, sex, age, weight, sbp, dbp
            patientInfo = input("Format - sex (0/1), age, weight, sbp, dbp: ")
            patientSex, patientAge, patientWeight, patientSbp, patientDbp = patientInfo.split(",")
            pred, warnings = predict_preop_dose(
                                sex=int(patientSex),
                                age=float(patientAge),
                                weight=float(patientWeight),
                                sbp=float(patientSbp),
                                dbp=float(patientDbp),
                                model=model,
                                stats=x_stats,
                                y_stats=y_stats,
                                feature_index=FEATURE_INDEX,
                                device=DEVICE,
                            )

            print(pred)
        except (KeyboardInterrupt, ValueError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
