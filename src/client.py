import sys
import os
import numpy as np
import xgboost as xgb
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Status, Code
from sklearn.metrics import f1_score

class BankXGBClient(fl.client.Client):
    def __init__(self, bank_id):
        self.bank_id = bank_id
        data = np.load(f"data/processed/bank_{bank_id}_data.npz")
        self.X_train = data["X"]
        self.y_train = data["y"]
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.local_model_path = f"models/bank_{bank_id}_local.json"

    def get_parameters(self, ins):
        return fl.common.GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[], tensor_type="bytes"),
        )

    def fit(self, ins):
        print(f"\n[BANK {self.bank_id}] Starting Local XGBoost Training on {len(self.X_train)} records...")
        
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42
        }

        # Agar pehle round ka global model exist karta hai, toh uske aage train karein (Incremental Boosting)
        if os.path.exists("models/global_xgb_model.json"):
            bst = xgb.train(params, self.dtrain, num_boost_round=15, xgb_model="models/global_xgb_model.json")
        else:
            bst = xgb.train(params, self.dtrain, num_boost_round=15)

        bst.save_model(self.local_model_path)

        # Model file ko bytes me convert karke server ko bhejna (Raw data nahi jayega)
        with open(self.local_model_path, "rb") as f:
            model_bytes = f.read()

        print(f"[BANK {self.bank_id}] Local training done. Sending encrypted/serialized model updates to Server...")
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensors=[model_bytes], tensor_type="bytes"),
            num_examples=len(self.X_train),
            metrics={},
        )

    def evaluate(self, ins):
        bst = xgb.Booster()
        bst.load_model("models/global_xgb_model.json")
        
        preds = bst.predict(self.dtrain)
        pred_labels = (preds > 0.5).astype(int)
        f1 = float(f1_score(self.y_train, pred_labels))
        loss = float(np.mean((self.y_train - preds) ** 2))

        print(f"[BANK {self.bank_id}] Evaluating Global Model on Local Data -> F1-Score: {f1:.4f}")
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=len(self.X_train),
            metrics={"f1_score": f1},
        )

if __name__ == "__main__":
    bank_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    print(f"[BANK {bank_id}] Connecting to Central Federated Server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=BankXGBClient(bank_id).to_client(),
    )
